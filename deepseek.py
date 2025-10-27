# -*- coding: utf-8 -*-
"""
deepseek_binance.py
Binance USDT-M futures trading bot (ccxt + DeepSeek LLM)

What’s inside:
- Multi-timeframe OHLCV (15m, 1h, 4h)
- Volume profile per timeframe: POC / VAH / VAL (+ alias VOH=VAH)
- Aggregates: vol (sum of volume), realized volatility
- Compact last-N bars per timeframe + simple stats (last/mean/stdev/slope/min/max + atr14_last)
- Pandas fillna FutureWarning fix
- Feasibility floor: compute minimum equity required from exchange limits + your risk params
  → If equity < floor, HOLD before calling LLM (save tokens)
  → Pass the floor into LLM to avoid made-up thresholds
- LIMIT ENTRY SUPPORT:
  - LLM can set entry.type="limit" with entry.price or entry.offset_bps
  - Post-only limit orders to avoid taker/slippage
  - Wait for fill (configurable timeout); place SL/TP only after filled

ENV (.env):
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
  POSITION_MODE=ONE_WAY          # or HEDGE
  MARGIN_MODE=cross              # or isolated
  LAST_N=5
  LOOKBACK=400
  VP_BINS=48
  # Limit-order behavior:
  ENTRY_DEFAULT_OFFSET_BPS=5     # when LLM provides no price/offset
  LIMIT_WAIT_SEC=90              # max seconds to wait for fill before action
  LIMIT_POLL_SEC=2               # poll interval
  CANCEL_ON_TIMEOUT=true         # cancel limit order if not fully filled within timeout

Run:
  python deepseek_binance.py -t 15m
"""
from __future__ import annotations
import os, json, math, time, argparse, logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import ccxt
from dotenv import load_dotenv

# ------------------------- Logging -------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bot")

# ------------------------- Config --------------------------------------------
load_dotenv()

SYMBOL = os.getenv("SYMBOL", "BNB/USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "3m")
MARGIN_MODE = os.getenv("MARGIN_MODE", "cross")
POSITION_MODE = os.getenv("POSITION_MODE", "ONE_WAY")
LEVERAGE = int(os.getenv("LEVERAGE", "10"))

RISK_FRACTION = float(os.getenv("RISK_FRACTION", "0.02"))
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", "1.5"))
MIN_RRR = float(os.getenv("MIN_RRR", "3.0"))

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

MULTI_TFS = ["15m", "1h", "4h"]
LOOKBACK = int(os.getenv("LOOKBACK", "400"))
LAST_N = int(os.getenv("LAST_N", "5"))
VP_BINS = int(os.getenv("VP_BINS", "48"))

COOLDOWN_BARS_3M = 5

# Limit order behavior
ENTRY_DEFAULT_OFFSET_BPS = float(os.getenv("ENTRY_DEFAULT_OFFSET_BPS", "5"))
LIMIT_WAIT_SEC = int(os.getenv("LIMIT_WAIT_SEC", "90"))
LIMIT_POLL_SEC = int(os.getenv("LIMIT_POLL_SEC", "2"))
CANCEL_ON_TIMEOUT = os.getenv("CANCEL_ON_TIMEOUT", "true").lower() in ("1","true","yes")

# ------------------------- Utils ---------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def clean_json_from_text(txt: str) -> str:
    i, j = txt.find("{"), txt.rfind("}")
    return txt[i:j+1] if (i != -1 and j != -1 and j > i) else txt

def slope_per_bar(arr: np.ndarray) -> float:
    n = len(arr)
    if n < 2: return 0.0
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), arr.mean()
    denom = ((x-xm)**2).sum()
    return 0.0 if denom == 0 else float(((x-xm)*(arr-ym)).sum()/denom)

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
    ef, es = ema(close, fast), ema(close, slow)
    m = ef - es
    s = ema(m, signal)
    h = m - s
    return m, s, h

def atr(df: pd.DataFrame, period=14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
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
    hlc3 = (df["high"]+df["low"]+df["close"])/3.0
    vol = df["volume"].to_numpy(dtype=float)
    pmin, pmax = float(df["low"].min()), float(df["high"].max())
    if pmax <= pmin: pmax = pmin + 1e-6
    bins = np.linspace(pmin, pmax, n_bins+1)
    mids = (bins[:-1] + bins[1:]) / 2.0
    hist = np.zeros(n_bins, dtype=float)
    idx = np.digitize(hlc3.to_numpy(dtype=float), bins) - 1
    idx = np.clip(idx, 0, n_bins-1)
    for i, v in zip(idx, vol):
        hist[i] += v
    total = hist.sum()
    poc_idx = int(np.argmax(hist))
    if total <= 0:
        m = float(mids[poc_idx])
        return VolumeProfileResult(m, m, m, mids, hist, 0.0)
    # expand 70% value area
    target, acc = 0.7*total, hist[poc_idx]
    L = R = poc_idx
    while acc < target and (L>0 or R<n_bins-1):
        left_next  = hist[L-1] if L>0 else -1
        right_next = hist[R+1] if R<n_bins-1 else -1
        if right_next >= left_next and R < n_bins-1:
            R += 1; acc += hist[R]
        elif L > 0:
            L -= 1; acc += hist[L]
        else:
            break
    return VolumeProfileResult(float(mids[poc_idx]), float(mids[R]), float(mids[L]), mids, hist, float(acc/total))

def realized_vol(close: pd.Series) -> float:
    if len(close) < 2: return 0.0
    r = np.diff(np.log(close.to_numpy(dtype=float)))
    return float(np.std(r, ddof=1))

def simple_stats(series: pd.Series) -> Dict[str, float]:
    arr = series.to_numpy(dtype=float)
    if len(arr) == 0:
        return {"last": None, "mean": None, "stdev": None, "slope": None, "min": None, "max": None}
    return {
        "last": float(arr[-1]),
        "mean": float(np.mean(arr)),
        "stdev": float(np.std(arr, ddof=1)) if len(arr)>1 else 0.0,
        "slope": slope_per_bar(arr),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

# ---------------------- Multi-timeframe --------------------------------------
def fetch_ohlcv_multi(exchange, symbol: str, timeframes: List[str], limit: int = 400,
                      since_ms: int | None = None) -> Dict[str, pd.DataFrame]:
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
        e20 = ema(close, 20)
        m, s, h = macd(close, 12, 26, 9)
        # RSI14 with bfill (fix FutureWarning)
        delta = close.diff()
        up, dn = delta.clip(lower=0.0), -delta.clip(upper=0.0)
        rs = up.rolling(14).mean() / (dn.rolling(14).mean().replace(0, np.nan))
        rsi14 = 100.0 - (100.0 / (1.0 + rs))
        rsi14 = rsi14.bfill().fillna(50.0)

        a14 = atr(df, 14)
        vpr = volume_profile(df, n_bins=n_bins)

        tail = df.tail(last_n).copy()
        last_candles = [{
            "t": c.timestamp.isoformat(),
            "o": float(c.open), "h": float(c.high),
            "l": float(c.low),  "c": float(c.close),
            "v": float(c.volume)
        } for c in tail.itertuples()]

        result[tf] = {
            "empty": False,
            "as_of": tail["timestamp"].iloc[-1].isoformat(),
            "last_n": last_n,
            "last_candles": last_candles,
            "stats": {
                "close": simple_stats(close.tail(last_n)),
                "ema20": simple_stats(e20.tail(last_n)),
                "macd":  simple_stats(m.tail(last_n)),
                "rsi14": simple_stats(rsi14.tail(last_n)),
                "atr14_last": float(a14.iloc[-1]) if not a14.empty else None
            },
            "volume_profile": {
                "poc": vpr.poc_price, "vah": vpr.vah, "val": vpr.val, "coverage": vpr.coverage
            },
            "aliases": {"voh": vpr.vah},
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

    def complete(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.2, max_tokens: int = 700) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = requests.post(self.url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

# ---------------------- Exchange init / account ------------------------------
def init_exchange() -> ccxt.binance:
    ak, sk = os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_SECRET")
    if not ak or not sk: raise RuntimeError("Missing BINANCE_API_KEY / BINANCE_SECRET")
    ex = ccxt.binance({
        "apiKey": ak, "secret": sk, "enableRateLimit": True,
        "options": {"defaultType": "future"}
    })
    ex.load_markets()
    try: ex.set_margin_mode(os.getenv("MARGIN_MODE","cross"), SYMBOL, params={"leverage": int(os.getenv("LEVERAGE","10"))})
    except Exception as e: log.warning(f"set_margin_mode failed: {e}")
    try: ex.set_leverage(int(os.getenv("LEVERAGE","10")), SYMBOL)
    except Exception as e: log.warning(f"set_leverage failed: {e}")
    return ex

def fetch_position_info(exchange) -> Dict[str, Any]:
    try:
        positions = exchange.fetch_positions([SYMBOL])
        pos = next((p for p in positions if p.get("symbol")==SYMBOL), None)
        if not pos: return {"side": None}
        qty = float(pos.get("contracts") or 0)
        entry = float(pos.get("entryPrice") or 0)
        liq   = float(pos.get("liquidationPrice") or 0)
        mark  = float(pos.get("markPrice") or 0)
        side = None
        if qty > 0:
            side = "LONG" if entry <= mark else "SHORT"  # heuristic for one-way
        return {"raw":pos, "side":side, "qty":qty, "entry_price":entry, "liquidation_price":liq, "current_price":mark}
    except Exception as e:
        log.warning(f"fetch_positions error: {e}")
    return {"side": None}

def fetch_equity(exchange) -> float:
    try:
        bal = exchange.fetch_balance(params={"type":"future"})
        usdt_total = bal.get("USDT",{}).get("total") or bal.get("total",{}).get("USDT")
        usdt_free  = bal.get("USDT",{}).get("free")  or bal.get("free",{}).get("USDT")
        log.info(f"Futures balance: total={usdt_total} free={usdt_free}")
        return float(usdt_total or 0.0)
    except Exception as e:
        log.warning(f"fetch_balance error: {e}")
        return 0.0

# ---------------------- Risk & precision helpers -----------------------------
def compute_rrr(entry: float, sl: float, tp: float, side: str) -> float:
    if side=="LONG":
        risk = abs(entry - sl); reward = abs(tp - entry)
    else:
        risk = abs(sl - entry); reward = abs(entry - tp)
    return 0.0 if risk<=0 else reward/risk

def atr_position_size(equity_usd: float, entry: float, sl: float, atr_val: float,
                      risk_fraction: float, atr_multiplier: float) -> float:
    eff_stop = max(abs(entry - sl), atr_val * atr_multiplier)
    max_risk = equity_usd * max(risk_fraction, 1e-6)
    if eff_stop<=0 or max_risk<=0: return 0.0
    return float(max_risk / eff_stop)

def market_info_rounders(exchange, symbol):
    m = exchange.market(symbol)
    # amount step
    amt_min = (m.get("limits",{}).get("amount",{}).get("min") or 0.0) or 0.001
    amt_step = amt_min  # conservative; for exact step you could parse m['info']['filters']
    # price tick
    price_prec = m.get("precision",{}).get("price")
    if price_prec is not None:
        tick = 10**(-price_prec)
    else:
        tick = (m.get("limits",{}).get("price",{}).get("min") or 0.01)
    def round_qty(q): return max(amt_min, math.floor(q/amt_step)*amt_step)
    def round_price(p): return round(p / tick) * tick
    return round_qty, round_price, m

def compute_min_equity_required(exchange, symbol, entry_price: float, atr_val: float,
                                risk_fraction: float, atr_multiplier: float) -> dict:
    m = exchange.market(symbol)
    min_notional = (m.get("limits",{}).get("cost",{}).get("min")
                    or m.get("info",{}).get("minNotional") or 0.0)
    min_qty = (m.get("limits",{}).get("amount",{}).get("min")
               or float(m.get("info",{}).get("minQty") or 0.0) or 0.0)
    stop_dist = max(atr_val * atr_multiplier, 1e-9)
    rf = max(risk_fraction, 1e-6)
    eq_from_notional = (min_notional * stop_dist)/(rf * max(entry_price,1e-9)) if min_notional else 0.0
    eq_from_qty      = (min_qty * stop_dist)/rf if min_qty else 0.0
    equity_min = max(eq_from_notional, eq_from_qty)
    return {"equity_min": float(equity_min),
            "limits": {"min_notional": float(min_notional), "min_qty": float(min_qty), "stop_dist": float(stop_dist)}}

# ---------------------- Order placement (market & limit) ---------------------
def compute_limit_price(side: str, last: float, entry_block: Dict[str,Any]) -> float:
    # prefer explicit price
    price = entry_block.get("price")
    if price:
        return float(price)
    # else use offset_bps
    bps = entry_block.get("offset_bps")
    if bps is None:
        bps = ENTRY_DEFAULT_OFFSET_BPS
    bps = float(bps)
    if side=="LONG":
        # buy below last
        return last * (1.0 - bps/10000.0)
    else:
        # sell above last
        return last * (1.0 + bps/10000.0)

def place_market_with_protection(exchange, side: str, qty: float, sl_price: float, tp_price: float) -> Dict[str,Any]:
    ord_side = "buy" if side=="LONG" else "sell"
    try:
        entry = exchange.create_order(SYMBOL, "market", ord_side, qty, None, {"reduceOnly": False})
        log.info(f"ENTRY filled (market): id={entry.get('id')} qty={qty}")
    except Exception as e:
        log.error(f"Market entry failed: {e}")
        return {"ok": False, "error": str(e)}
    try:
        exit_side = "sell" if side=="LONG" else "buy"
        sl = exchange.create_order(SYMBOL, "STOP_MARKET", exit_side, qty, None, {"stopPrice": sl_price, "reduceOnly": True})
        tp = exchange.create_order(SYMBOL, "TAKE_PROFIT_MARKET", exit_side, qty, None, {"stopPrice": tp_price, "reduceOnly": True})
        log.info(f"SL={sl.get('id')} TP={tp.get('id')}")
        return {"ok": True, "entry": entry, "sl": sl, "tp": tp}
    except Exception as e:
        log.error(f"Protective orders failed: {e}")
        return {"ok": False, "error": str(e)}

def place_limit_postonly(exchange, side: str, qty: float, price: float) -> Dict[str,Any]:
    ord_side = "buy" if side=="LONG" else "sell"
    params = {"postOnly": True, "timeInForce": "GTX"}  # GTX ensures maker-only on Binance futures via ccxt
    return exchange.create_order(SYMBOL, "limit", ord_side, qty, price, params)

def wait_fill_and_place_protection(exchange, side: str, order, sl_price: float, tp_price: float,
                                   max_wait_sec: int, poll_sec: int, cancel_on_timeout: bool) -> Dict[str,Any]:
    order_id = order.get("id")
    deadline = time.time() + max_wait_sec
    filled_qty = 0.0
    while time.time() < deadline:
        try:
            o = exchange.fetch_order(order_id, SYMBOL)
            status = o.get("status")
            filled = float(o.get("filled") or 0.0)
            remaining = float(o.get("remaining") or 0.0)
            log.info(f"[LIMIT] status={status} filled={filled} remaining={remaining}")
            if status in ("closed","filled") or (filled > 0 and remaining <= 0):
                filled_qty = filled
                break
            time.sleep(poll_sec)
        except Exception as e:
            log.warning(f"fetch_order error: {e}")
            time.sleep(poll_sec)

    if filled_qty <= 0:
        if cancel_on_timeout:
            try:
                exchange.cancel_order(order_id, SYMBOL)
                log.info(f"[LIMIT] Cancelled unfilled order {order_id}")
            except Exception as e:
                log.warning(f"Cancel error: {e}")
        return {"ok": False, "error": "limit_not_filled"}

    # place protection for the filled amount only
    try:
        exit_side = "sell" if side=="LONG" else "buy"
        sl = exchange.create_order(SYMBOL, "STOP_MARKET", exit_side, filled_qty, None, {"stopPrice": sl_price, "reduceOnly": True})
        tp = exchange.create_order(SYMBOL, "TAKE_PROFIT_MARKET", exit_side, filled_qty, None, {"stopPrice": tp_price, "reduceOnly": True})
        log.info(f"[LIMIT] Filled qty={filled_qty}. SL={sl.get('id')} TP={tp.get('id')}")
        return {"ok": True, "entry": order, "sl": sl, "tp": tp}
    except Exception as e:
        log.error(f"Protective orders after limit fill failed: {e}")
        return {"ok": False, "error": str(e), "partial_filled_qty": filled_qty}

# ---------------------- Prompt build -----------------------------------------
SYSTEM_PROMPT = (
    "You are a trading decision engine. Follow HARD CONSTRAINTS:\n"
    "1) Output ONLY valid JSON per schema (no prose).\n"
    "2) Obey risk rules; if constraints fail, return HOLD with reasons.\n"
    "3) No hindsight; use only provided data.\n"
    "4) Prefer LIMIT entry with maker liquidity when it improves execution.\n"
)

def build_user_prompt(multi: Dict[str, Any], equity_usd: float, position: Dict[str, Any], feasibility: Dict[str,Any]) -> str:
    payload = {
        "market_metadata": {
            "exchange": "BINANCE", "symbol": SYMBOL, "instrument_type": "PERP",
            "position_mode": POSITION_MODE, "timeframes": MULTI_TFS,
            "timezone": "Asia/Tokyo", "as_of": now_iso()
        },
        "risk_policy": {
            "risk_fraction_max": RISK_FRACTION,
            "atr_multiplier_for_sl": ATR_MULTIPLIER,
            "min_rrr": MIN_RRR,
            "cooldown_bars_3m": COOLDOWN_BARS_3M,
            "min_equity_required": feasibility.get("equity_min", 0.0),
            "min_equity_required_passed": float(equity_usd) >= float(feasibility.get("equity_min", 0.0)),
            "exchange_limits": feasibility.get("limits", {})
        },
        "account": {"equity_usd": equity_usd},
        "position": position,
        "multi_timeframe": multi,
        "execution_hints": {
            "prefer_limit": True,
            "post_only": True,
            "entry_default_offset_bps": ENTRY_DEFAULT_OFFSET_BPS
        },
        "rules": [
            "If a position exists, prefer ADJUST over REVERSE.",
            "REVERSE only with confidence>=0.7, better RRR by >=20%, cooldown satisfied.",
            "No contradictory TP/SL; else HOLD.",
            "When proposing LIMIT, provide either price or offset_bps (bps=1/100 of 1%)."
        ],
        "output_schema": {
            "action": "OPEN|ADJUST|CLOSE|REVERSE|HOLD",
            "side": "LONG|SHORT|null",
            "confidence": "0.0-1.0",
            "entry": {"type": "market|limit", "price": "number|null", "offset_bps": "number|null"},
            "stop_loss": {"type": "STOP_MARKET", "price": "number"},
            "take_profit": {"type": "TAKE_PROFIT_MARKET", "price": "number"},
            "rrr": "number", "risk_usd": "number", "qty": "number (base)",
            "cooldown_ok": "bool", "reasons": ["..."], "invalidation": "string"
        }
    }
    return "Return ONLY valid JSON:\n" + json.dumps(payload, ensure_ascii=False)

# ---------------------- Main tick --------------------------------------------
def tick(exchange, llm: DeepSeekClient):
    # Multi-timeframe features
    mt_ohlcv = fetch_ohlcv_multi(exchange, SYMBOL, MULTI_TFS, limit=LOOKBACK)
    mt_features = build_timeframe_features(mt_ohlcv, last_n=LAST_N, n_bins=VP_BINS)
    atr15 = mt_features.get("15m",{}).get("stats",{}).get("atr14_last", None)
    if atr15 is None:
        log.info("No ATR (15m) -> HOLD")
        return

    equity = fetch_equity(exchange)
    position = fetch_position_info(exchange)

    ticker = exchange.fetch_ticker(SYMBOL)
    last_price = float(ticker["last"])
    floor = compute_min_equity_required(exchange, SYMBOL, last_price, atr15, RISK_FRACTION, ATR_MULTIPLIER)
    log.info(f"Feasibility: min_equity≈{floor['equity_min']:.4f} "
             f"(min_notional={floor['limits']['min_notional']}, min_qty={floor['limits']['min_qty']}, stop≈{floor['limits']['stop_dist']:.6f})")
    if floor["equity_min"] and equity < floor["equity_min"]:
        log.info(f"HOLD: equity {equity:.2f} < required {floor['equity_min']:.2f}")
        return

    user_prompt = build_user_prompt(mt_features, equity, position, feasibility=floor)

    # LLM decision
    try:
        content = llm.complete(SYSTEM_PROMPT, user_prompt, temperature=0.2, max_tokens=700)
        log.info(f"LLM raw:\n{content}")
        decision = json.loads(clean_json_from_text(content))
    except Exception as e:
        log.error(f"LLM/JSON error: {e}")
        return

    action = decision.get("action","HOLD"); side = decision.get("side")
    entry_block = decision.get("entry",{}) or {}
    sl_block = decision.get("stop_loss",{}) or {}
    tp_block = decision.get("take_profit",{}) or {}
    rrr = float(decision.get("rrr",0.0))
    qty_suggested = float(decision.get("qty",0.0))
    reasons = decision.get("reasons", [])

    if action=="HOLD":
        log.info(f"HOLD: {reasons}")
        return
    if rrr < MIN_RRR:
        log.info(f"HOLD: RRR {rrr:.2f} < MIN_RRR {MIN_RRR:.2f}")
        return

    sl_price = float(sl_block.get("price",0) or 0)
    tp_price = float(tp_block.get("price",0) or 0)
    # sanity relations
    if side=="LONG" and not (tp_price > last_price > sl_price):
        log.info("Invalid LONG TP/Entry/SL relation -> HOLD"); return
    if side=="SHORT" and not (tp_price < last_price < sl_price):
        log.info("Invalid SHORT TP/Entry/SL relation -> HOLD"); return

    # qty sizing (ATR-based) if LLM didn't provide
    round_qty, round_price, market = market_info_rounders(exchange, SYMBOL)
    qty = qty_suggested
    if qty <= 0:
        qty = atr_position_size(equity, last_price, sl_price, atr15, RISK_FRACTION, ATR_MULTIPLIER)
    qty = round_qty(qty)
    if qty <= 0:
        log.info("Computed qty <= 0 -> HOLD"); return

    entry_type = (entry_block.get("type") or "market").lower()
    if entry_type == "market":
        res = place_market_with_protection(exchange, side, qty, sl_price, tp_price)
        if res.get("ok"):
            log.info(f"{action} {side} via MARKET ok. qty={qty} SL={sl_price} TP={tp_price}")
        else:
            log.error(f"Market path failed: {res.get('error')}")
        return

    # ----- LIMIT ENTRY path -----
    # compute limit price from LLM price or offset_bps
    limit_price_raw = compute_limit_price(side, last_price, entry_block)
    limit_price = round_price(limit_price_raw)

    try:
        order = place_limit_postonly(exchange, side, qty, limit_price)
        log.info(f"Placed LIMIT postOnly: id={order.get('id')} side={side} qty={qty} px={limit_price}")
    except Exception as e:
        log.error(f"Limit placement failed: {e}")
        return

    res = wait_fill_and_place_protection(
        exchange=exchange, side=side, order=order,
        sl_price=sl_price, tp_price=tp_price,
        max_wait_sec=LIMIT_WAIT_SEC, poll_sec=LIMIT_POLL_SEC,
        cancel_on_timeout=CANCEL_ON_TIMEOUT
    )
    if res.get("ok"):
        log.info(f"{action} {side} via LIMIT ok. qty≈(filled) SL={sl_price} TP={tp_price}")
    else:
        log.warning(f"LIMIT not filled or protection failed: {res.get('error')}")

# ---------------------- CLI / main -------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--timeframe", default=TIMEFRAME, help="main loop cadence, e.g., 3m/15m")
    args = parser.parse_args()
    cadence = args.timeframe

    exchange = init_exchange()
    if not DEEPSEEK_API_KEY: raise RuntimeError("Missing DEEPSEEK_API_KEY")
    llm = DeepSeekClient(DEEPSEEK_API_KEY, model="deepseek-chat")

    log.info(f"Start bot: SYMBOL={SYMBOL} cadence={cadence} multi_tfs={MULTI_TFS}, LAST_N={LAST_N}")

    # cadence sleep map
    tf_to_sec = {"1m":60,"3m":180,"5m":300,"15m":900,"30m":1800,"1h":3600,"4h":14400}
    interval = tf_to_sec.get(cadence, 180)

    while True:
        try:
            tick(exchange, llm)
        except Exception as e:
            log.exception(f"tick error: {e}")
        time.sleep(interval)

if __name__ == "__main__":
    main()
