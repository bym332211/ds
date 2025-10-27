# -*- coding: utf-8 -*-
from __future__ import annotations
import os, time, argparse, json, logging
from typing import Any, Dict

import ccxt
from dotenv import load_dotenv

from bot.data import fetch_ohlcv_multi
from bot.indicators import build_timeframe_features
from bot.llm import DeepSeekClient, build_user_prompt
from bot.exchange import init_exchange, fetch_position_info, fetch_equity
from bot.risk import atr_position_size, compute_min_equity_required, market_info_rounders
from bot.orders import compute_limit_price, place_market_with_protection, place_limit_postonly, wait_fill_and_place_protection

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bot")

SYMBOL = os.getenv("SYMBOL", "BNB/USDT")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MULTI_TFS = ["15m", "1h", "4h"]
LOOKBACK = int(os.getenv("LOOKBACK", "400"))
LAST_N = int(os.getenv("LAST_N", "5"))
VP_BINS = int(os.getenv("VP_BINS", "48"))
RISK_FRACTION = float(os.getenv("RISK_FRACTION", "0.02"))
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", "1.5"))
MIN_RRR = float(os.getenv("MIN_RRR", "3.0"))
ENTRY_DEFAULT_OFFSET_BPS = float(os.getenv("ENTRY_DEFAULT_OFFSET_BPS", "5"))
LIMIT_WAIT_SEC = int(os.getenv("LIMIT_WAIT_SEC", "90"))
LIMIT_POLL_SEC = int(os.getenv("LIMIT_POLL_SEC", "2"))
CANCEL_ON_TIMEOUT = os.getenv("CANCEL_ON_TIMEOUT", "true").lower() in ("1","true","yes")

SYSTEM_PROMPT = (
    "You are a trading decision engine. Follow HARD CONSTRAINTS:\n"
    "1) Output ONLY valid JSON per schema (no prose).\n"
    "2) Obey risk rules; if constraints fail, return HOLD with reasons.\n"
    "3) No hindsight; use only provided data.\n"
    "4) Prefer LIMIT entry with maker liquidity when it improves execution.\n"
)


def now_iso():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def clean_json_from_text(txt: str) -> str:
    i, j = txt.find("{"), txt.rfind("}")
    return txt[i:j+1] if (i != -1 and j != -1 and j > i) else txt


## build_user_prompt moved to bot.llm for reuse


def tick(exchange: ccxt.binance, llm: DeepSeekClient):
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
    log.info(f"Feasibility: min_equity={floor['equity_min']:.4f} "
             f"(min_notional={floor['limits']['min_notional']}, min_qty={floor['limits']['min_qty']}, stop={floor['limits']['stop_dist']:.6f})")
    if floor["equity_min"] and equity < floor["equity_min"]:
        log.info(f"HOLD: equity {equity:.2f} < required {floor['equity_min']:.2f}")
        return

    user_prompt = build_user_prompt(
        symbol=SYMBOL,
        multi_tfs=MULTI_TFS,
        entry_default_offset_bps=ENTRY_DEFAULT_OFFSET_BPS,
        equity_usd=equity,
        position=position,
        multi=mt_features,
        feasibility=floor,
        position_mode=os.getenv("POSITION_MODE","ONE_WAY"),
        min_rrr=MIN_RRR,
        risk_fraction=RISK_FRACTION,
        atr_multiplier=ATR_MULTIPLIER,
        cooldown_bars_3m=int(os.getenv("COOLDOWN_BARS_3M","5"))
    )

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
    if side=="LONG" and not (tp_price > last_price > sl_price):
        log.info("Invalid LONG TP/Entry/SL relation -> HOLD"); return
    if side=="SHORT" and not (tp_price < last_price < sl_price):
        log.info("Invalid SHORT TP/Entry/SL relation -> HOLD"); return

    round_qty, round_price, _ = market_info_rounders(exchange, SYMBOL)
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
        log.info(f"{action} {side} via LIMIT ok. SL={sl_price} TP={tp_price}")
    else:
        log.warning(f"LIMIT not filled or protection failed: {res.get('error')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--timeframe", default=os.getenv("TIMEFRAME","3m"), help="main loop cadence, e.g., 3m/15m")
    args = parser.parse_args()
    cadence = args.timeframe

    exchange = init_exchange()
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY")
    llm = DeepSeekClient(DEEPSEEK_API_KEY, model="deepseek-chat")

    log.info(f"Start bot: SYMBOL={SYMBOL} cadence={cadence} multi_tfs={MULTI_TFS}, LAST_N={LAST_N}")

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
