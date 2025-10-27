# -*- coding: utf-8 -*-
"""
deepseek_binance.py
A Binance USDT-M futures trading bot using ccxt + DeepSeek LLM.
Modifications:
- Fetch multi-timeframe OHLCV (15m, 1h, 4h) in one shot
- For each timeframe, compute volume profile (POC/VAH/VAL), vol(sum), realized volatility
- Keep only the last N candles per timeframe + simple stats (last/mean/stdev/slope/min/max + atr14_last)

ENV (.env) required:
  BINANCE_API_KEY=
  BINANCE_SECRET=
  DEEPSEEK_API_KEY=
Optional:
  SYMBOL=BNB/USDT
  TIMEFRAME=3m
  LEVERAGE=10
  RISK_FRACTION=0.02
  ATR_MULTIPLIER=1.5
  MIN_RRR=3.0
  POSITION_MODE=ONE_WAY  # or HEDGE (your account setting)
  MARGIN_MODE=cross      # or isolated

Run:
  python deepseek_binance.py -t 15m
"""

from __future__ import annotations
import os
import re
import json
import math
import time
import argparse
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from datetime import datetime, timezone
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import requests
import ccxt

# ------------------------- Logging -------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("bot")

# ------------------------- Config --------------------------------------------
load_dotenv()

SYMBOL = os.getenv("SYMBOL", "BNB/USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "3m")   # main tick cadence
MARGIN_MODE = os.getenv("MARGIN_MODE", "cross")
POSITION_MODE = os.getenv("POSITION_MODE", "ONE_WAY")  # or HEDGE
LEVERAGE = int(os.getenv("LEVERAGE", "10"))

RISK_FRACTION = float(os.getenv("RISK_FRACTION", "0.02"))
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", "1.5"))
MIN_RRR = float(os.getenv("MIN_RRR", "3.0"))

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

MULTI_TFS = ["15m", "1h", "4h"]
LOOKBACK = 400           # candles per timeframe
LAST_N = 5               # last N candles kept in prompt
VP_BINS = 48             # volume profile bins

# cooldown for reversals on 3m cadence
COOLDOWN_BARS_3M = 5

# ------------------------- Utils ---------------------------------------------

def now_iso_tz(tokyo: bool = True) -> str:
    if tokyo:
        # Asia/Tokyo is UTC+9 without DST; we keep ISO in UTC for simplicity
        return datetime.now(timezone.utc).isoformat()
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()


def clean_json_from_text(txt: str) -> str:
    """Extract first JSON object from text and return it."""
    # Try a simple brace matching heuristic
    start = txt.find("{")
    end = txt.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = txt[start:end + 1]
        return candidate
    return txt


def slope_per_bar(arr: np.ndarray) -> float:
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = arr.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    slope = ((x - x_mean) * (arr - y_mean)).sum() / denom
    return float(slope)

# ---------------------- Indicator/Feature Helpers ----------------------------

def to_df(ohlcv: List[List[float]]) -> pd.DataFrame:
    if not ohlcv:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    arr = np.array(ohlcv, dtype=float)
    df = pd.DataFrame(arr, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(close: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, period=14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


@dataclass
class VolumeProfileResult:
    poc_price: float
    vah: float
    val: float
    bins_price: np.ndarray
    bins_volume: np.ndarray
    coverage: float


def volume_profile(df: pd.DataFrame, n_bins: int = 48) -> VolumeProfileResult:
    if df.empty:
        return VolumeProfileResult(0.0,0.0,0.0,np.array([]),np.array([]),0.0)
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].to_numpy(dtype=float)
    pmin, pmax = float(df["low"].min()), float(df["high"].max())
    if pmax <= pmin:
        pmax = pmin + 1e-6
    bins = np.linspace(pmin, pmax, n_bins + 1)
    mids = (bins[:-1] + bins[1:]) / 2.0
    hist = np.zeros(n_bins, dtype=float)
    idx = np.digitize(hlc3.to_numpy(dtype=float), bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    for i, v in zip(idx, vol):
        hist[i] += v
    total = hist.sum()
    poc_idx = int(np.argmax(hist))
    if total <= 0:
        m = float(mids[poc_idx])
        return VolumeProfileResult(m, m, m, mids, hist, 0.0)
    target = 0.7 * total
    acc = hist[poc_idx]
    L = R = poc_idx
    while acc < target and (L > 0 or R < n_bins - 1):
        left_next = hist[L-1] if L > 0 else -1
        right_next = hist[R+1] if R < n_bins-1 else -1
        if right_next >= left_next and R < n_bins-1:
            R += 1
            acc += hist[R]
        elif L > 0:
            L -= 1
            acc += hist[L]
        else:
            break
    poc = float(mids[poc_idx])
    vah = float(mids[R])
    val = float(mids[L])
    coverage = float(acc / total)
    return VolumeProfileResult(poc, vah, val, mids, hist, coverage)


def realized_vol(close: pd.Series) -> float:
    if len(close) < 2:
        return 0.0
    r = np.diff(np.log(close.to_numpy(dtype=float)))
    return float(np.std(r, ddof=1))


def simple_stats(series: pd.Series) -> Dict[str, float]:
    arr = series.to_numpy(dtype=float)
    if len(arr) == 0:
        return {"last": None, "mean": None, "stdev": None, "slope": None, "min": None, "max": None}
    return {
        "last": float(arr[-1]),
        "mean": float(np.mean(arr)),
        "stdev": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "slope": slope_per_bar(arr),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

# ---------------------- Multi-timeframe fetch & features ---------------------

def fetch_ohlcv_multi(exchange, symbol: str, timeframes: List[str], limit: int = 400, since_ms: int | None = None) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, since=since_ms, limit=limit)
            out[tf] = to_df(ohlcv)
        except Exception as e:
            log.error(f"fetch_ohlcv failed for {tf}: {e}")
            out[tf] = pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    return out


def build_timeframe_features(data: Dict[str, pd.DataFrame], last_n: int = 5, n_bins: int = 48) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for tf, df in data.items():
        if df.empty:
            result[tf] = {"empty": True}
            continue
        df = df.sort_values("timestamp").reset_index(drop=True).copy()
        close = df["close"]
        ema20 = ema(close, 20)
        macd_line, sig, hist = macd(close, 12, 26, 9)
        # RSI14
        delta = close.diff()
        up = delta.clip(lower=0.0)
        down = -delta.clip(upper=0.0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / (roll_down.replace(0, np.nan))
        rsi14 = 100.0 - (100.0 / (1.0 + rs))
        rsi14 = rsi14.fillna(method="bfill").fillna(50.0)
        atr14 = atr(df, 14)

        vpr = volume_profile(df, n_bins=n_bins)
        tail = df.tail(last_n).copy()
        last_candles = [
            {
                "t": c.timestamp.isoformat(),
                "o": float(c.open),
                "h": float(c.high),
                "l": float(c.low),
                "c": float(c.close),
                "v": float(c.volume),
            } for c in tail.itertuples()
        ]

        stats_close = simple_stats(close.tail(last_n))
        stats_ema20 = simple_stats(ema20.tail(last_n))
        stats_macd = simple_stats(macd_line.tail(last_n))
        stats_rsi = simple_stats(rsi14.tail(last_n))

        result[tf] = {
            "empty": False,
            "as_of": tail["timestamp"].iloc[-1].isoformat(),
            "last_n": last_n,
            "last_candles": last_candles,
            "stats": {
                "close": stats_close,
                "ema20": stats_ema20,
                "macd": stats_macd,
                "rsi14": stats_rsi,
                "atr14_last": float(atr14.iloc[-1]) if not atr14.empty else None
            },
            "volume_profile": {
                "poc": vpr.poc_price,
                "vah": vpr.vah,
                "val": vpr.val,
                "coverage": vpr.coverage
            },
            "aliases": {
                "voh": vpr.vah
            },
            "aggregates": {
                "vol": float(df["volume"].sum()),
                "volatility": realized_vol(close)
            }
        }
    return result

# ---------------------- LLM (DeepSeek) ---------------------------------------

class DeepSeekClient:
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.deepseek.com/chat/completions"

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(self.url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content

# ---------------------- Exchange init / account ------------------------------

def init_exchange() -> ccxt.binance:
    api_key = os.getenv("BINANCE_API_KEY")
    secret = os.getenv("BINANCE_SECRET")
    if not api_key or not secret:
        raise RuntimeError("Missing BINANCE_API_KEY / BINANCE_SECRET")

    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",  # USDT-M futures
        }
    })
    exchange.load_markets()

    # set margin mode + leverage
    try:
        market = exchange.market(SYMBOL)
        exchange.set_margin_mode(MARGIN_MODE, SYMBOL, params={"leverage": LEVERAGE})
    except Exception as e:
        log.warning(f"set_margin_mode failed: {e}")
    try:
        exchange.set_leverage(LEVERAGE, SYMBOL)
    except Exception as e:
        log.warning(f"set_leverage failed: {e}")
    return exchange


def fetch_position_info(exchange) -> Dict[str, Any]:
    """Return simplified position info (one-way mode)."""
    try:
        positions = exchange.fetch_positions([SYMBOL])
        pos = next((p for p in positions if p.get("symbol") == SYMBOL), None)
        if not pos:
            return {"side": None}
        side = "LONG" if float(pos.get("contracts", 0)) > 0 and float(pos.get("entryPrice") or 0) <= float(pos.get("markPrice") or 0) else \
               ("SHORT" if float(pos.get("contracts", 0)) > 0 else None)
        qty = float(pos.get("contracts") or 0)
        entry = float(pos.get("entryPrice") or 0)
        liq = float(pos.get("liquidationPrice") or 0)
        mark = float(pos.get("markPrice") or 0)
        return {
            "raw": pos,
            "side": side,
            "qty": qty,
            "entry_price": entry,
            "liquidation_price": liq,
            "current_price": mark
        }
    except Exception as e:
        log.warning(f"fetch_positions error: {e}")
        return {"side": None}


def fetch_equity(exchange) -> float:
    try:
        bal = exchange.fetch_balance(params={"type": "future"})
        # USDT equity:
        total = bal.get("USDT", {}).get("total")
        if total is None:
            total = bal.get("total", {}).get("USDT")
        return float(total or 0.0)
    except Exception as e:
        log.warning(f"fetch_balance error: {e}")
        return 0.0

# ---------------------- Order & risk helpers ---------------------------------

def compute_rrr(entry: float, sl: float, tp: float, side: str) -> float:
    if side == "LONG":
        risk = abs(entry - sl)
        reward = abs(tp - entry)
    else:
        risk = abs(sl - entry)
        reward = abs(entry - tp)
    if risk <= 0:
        return 0.0
    return reward / risk


def atr_position_size(equity_usd: float, entry: float, sl: float, atr_val: float, price_precision: int = 2, risk_fraction: float = 0.02, atr_multiplier: float = 1.5) -> float:
    # effective stop >= max(|entry - SL|, ATR * multiplier)
    eff_stop = max(abs(entry - sl), atr_val * atr_multiplier)
    max_risk = equity_usd * risk_fraction
    if eff_stop <= 0 or max_risk <= 0:
        return 0.0
    qty = max_risk / eff_stop
    return float(qty)


def place_order_with_protection(exchange, side: str, qty: float, entry_type: str, sl_price: float, tp_price: float) -> Dict[str, Any]:
    ord_side = "buy" if side == "LONG" else "sell"
    params = {"reduceOnly": False}
    try:
        # Market entry
        order = exchange.create_order(SYMBOL, "market", ord_side, qty, None, params)
        log.info(f"ENTRY filled: {order.get('id')} qty={qty}")
    except Exception as e:
        log.error(f"Entry order failed: {e}")
        return {"ok": False, "error": str(e)}

    # Protective orders (reduceOnly)
    try:
        if side == "LONG":
            sl_side = "sell"
            tp_side = "sell"
        else:
            sl_side = "buy"
            tp_side = "buy"
        # STOP_MARKET
        sl = exchange.create_order(SYMBOL, "STOP_MARKET", sl_side, qty, None, {
            "stopPrice": sl_price,
            "reduceOnly": True
        })
        # TAKE_PROFIT_MARKET
        tp = exchange.create_order(SYMBOL, "TAKE_PROFIT_MARKET", tp_side, qty, None, {
            "stopPrice": tp_price,
            "reduceOnly": True
        })
        log.info(f"SL={sl.get('id')} TP={tp.get('id')}")
        return {"ok": True, "entry": order, "sl": sl, "tp": tp}
    except Exception as e:
        log.error(f"Protective orders failed: {e}")
        return {"ok": False, "error": str(e)}

# ---------------------- Prompt build -----------------------------------------

SYSTEM_PROMPT = """You are a trading decision engine. Follow these HARD CONSTRAINTS in order:
1) Output ONLY valid JSON matching the schema. No prose.
2) Obey risk rules and constraints. If constraints fail, return HOLD with reasons.
3) No hindsight; use only provided data.
"""

def build_user_prompt(multi: Dict[str, Any], equity_usd: float, position: Dict[str, Any]) -> str:
    # Keep the prompt compact, pointing the model to use stats & volume profile.
    payload = {
        "market_metadata": {
            "exchange": "BINANCE",
            "symbol": SYMBOL,
            "instrument_type": "PERP",
            "position_mode": POSITION_MODE,
            "timeframes": MULTI_TFS,
            "timezone": "Asia/Tokyo",
            "as_of": now_iso_tz()
        },
        "risk_policy": {
            "risk_fraction_max": RISK_FRACTION,
            "atr_multiplier_for_sl": ATR_MULTIPLIER,
            "min_rrr": MIN_RRR,
            "cooldown_bars_3m": COOLDOWN_BARS_3M
        },
        "account": {
            "equity_usd": equity_usd
        },
        "position": position,
        "multi_timeframe": multi,
        "rules": [
            "Prefer ADJUST over REVERSE when a position exists.",
            "REVERSE needs confidence>=0.7, better RRR by >=20% and cooldown satisfied.",
            "No contradictory TP/SL. If violation would occur, HOLD."
        ],
        "output_schema": {
            "action": "OPEN|ADJUST|CLOSE|REVERSE|HOLD",
            "side": "LONG|SHORT|null",
            "confidence": "0.0-1.0",
            "entry": {"type": "market|limit", "price": "number|null", "offset_bps": "number|null"},
            "stop_loss": {"type": "STOP_MARKET", "price": "number"},
            "take_profit": {"type": "TAKE_PROFIT_MARKET", "price": "number"},
            "rrr": "number",
            "risk_usd": "number",
            "qty": "number (base asset)",
            "cooldown_ok": "bool",
            "reasons": ["strings..."],
            "invalidation": "string"
        }
    }
    return (
        "Return ONLY valid JSON for the decision per schema.\n"
        + json.dumps(payload, ensure_ascii=False)
    )

# ---------------------- Main tick --------------------------------------------

def tick(exchange, llm: DeepSeekClient):
    # Multi-timeframe OHLCV
    mt_ohlcv = fetch_ohlcv_multi(exchange, SYMBOL, MULTI_TFS, limit=LOOKBACK)
    mt_features = build_timeframe_features(mt_ohlcv, last_n=LAST_N, n_bins=VP_BINS)

    # Equity & position
    equity = fetch_equity(exchange)
    position = fetch_position_info(exchange)

    # Build prompt
    user_prompt = build_user_prompt(mt_features, equity, position)

    # LLM completion
    try:
        content = llm.complete(SYSTEM_PROMPT, user_prompt, temperature=0.2, max_tokens=700)
        log.info(f"LLM raw:\n{content}")
    except Exception as e:
        log.error(f"DeepSeek error: {e}")
        return

    # Parse JSON
    try:
        jtxt = clean_json_from_text(content)
        decision = json.loads(jtxt)
    except Exception as e:
        log.error(f"JSON parse error: {e}")
        return

    action = decision.get("action", "HOLD")
    side = decision.get("side")
    confidence = float(decision.get("confidence", 0.0))
    entry_block = decision.get("entry", {}) or {}
    sl_block = decision.get("stop_loss", {}) or {}
    tp_block = decision.get("take_profit", {}) or {}
    rrr = float(decision.get("rrr", 0.0))
    qty_suggested = float(decision.get("qty", 0.0))
    reasons = decision.get("reasons", [])

    if action == "HOLD":
        log.info(f"HOLD: reasons={reasons}")
        return

    # Risk checks
    if rrr < MIN_RRR:
        log.info(f"RRR {rrr:.2f} < MIN_RRR {MIN_RRR:.2f}; HOLD")
        return

    # Use latest 15m ATR for sizing if available
    atr15 = mt_features.get("15m", {}).get("stats", {}).get("atr14_last", None)
    ticker = exchange.fetch_ticker(SYMBOL)
    current_price = float(ticker["last"])
    entry_price = current_price if entry_block.get("type") == "market" or not entry_block else float(entry_block.get("price") or current_price)

    sl_price = float(sl_block.get("price", 0))
    tp_price = float(tp_block.get("price", 0))
    if side == "LONG" and not (tp_price > entry_price > sl_price):
        log.info("Invalid LONG TP/Entry/SL relation; HOLD")
        return
    if side == "SHORT" and not (tp_price < entry_price < sl_price):
        log.info("Invalid SHORT TP/Entry/SL relation; HOLD")
        return

    if atr15 is None:
        log.info("No ATR value; HOLD to avoid blind risk.")
        return

    qty = qty_suggested
    if qty <= 0:
        qty = atr_position_size(
            equity_usd=equity,
            entry=entry_price,
            sl=sl_price,
            atr_val=atr15,
            risk_fraction=RISK_FRACTION,
            atr_multiplier=ATR_MULTIPLIER
        )
        # round quantity by market precision if needed
        market = exchange.market(SYMBOL)
        step = market.get("limits", {}).get("amount", {}).get("min", 0.001) or 0.001
        qty = max(step, math.floor(qty / step) * step)

    if qty <= 0:
        log.info("Computed qty <= 0; HOLD")
        return

    # Place order + protections
    res = place_order_with_protection(
        exchange=exchange,
        side=side,
        qty=qty,
        entry_type=entry_block.get("type", "market"),
        sl_price=sl_price,
        tp_price=tp_price
    )
    if res.get("ok"):
        log.info(f"{action} {side} done. qty={qty} entry≈{entry_price} SL={sl_price} TP={tp_price}")
    else:
        log.error(f"Order failed: {res.get('error')}")

# ---------------------- CLI / main -------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--timeframe", default=TIMEFRAME, help="main loop timeframe cadence, e.g., 3m/15m")
    args = parser.parse_args()
    cadence = args.timeframe

    exchange = init_exchange()
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY")

    llm = DeepSeekClient(DEEPSEEK_API_KEY, model="deepseek-chat")

    log.info(f"Start bot: SYMBOL={SYMBOL} cadence={cadence} multi_tfs={MULTI_TFS}, LAST_N={LAST_N}")

    # run once at start, then sleep according to cadence length
    tf_to_sec = {
        "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "4h": 14400
    }
    interval = tf_to_sec.get(cadence, 180)

    while True:
        try:
            tick(exchange, llm)
        except Exception as e:
            log.exception(f"tick error: {e}")
        time.sleep(interval)

if __name__ == "__main__":
    main()
