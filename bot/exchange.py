from __future__ import annotations
from typing import Any, Dict
import os
import ccxt


def init_exchange() -> ccxt.binance:
    ak, sk = os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_SECRET")
    if not ak or not sk:
        raise RuntimeError("Missing BINANCE_API_KEY / BINANCE_SECRET")
    ex = ccxt.binance({
        "apiKey": ak, "secret": sk, "enableRateLimit": True,
        "options": {"defaultType": "future"}
    })
    ex.load_markets()
    symbol = os.getenv("SYMBOL", "BNB/USDT")
    try:
        ex.set_margin_mode(os.getenv("MARGIN_MODE","cross"), symbol, params={"leverage": int(os.getenv("LEVERAGE","10"))})
    except Exception:
        pass
    try:
        ex.set_leverage(int(os.getenv("LEVERAGE","10")), symbol)
    except Exception:
        pass
    return ex


def fetch_position_info(exchange) -> Dict[str, Any]:
    symbol = os.getenv("SYMBOL", "BNB/USDT")
    try:
        positions = exchange.fetch_positions([symbol])
        pos = next((p for p in positions if p.get("symbol")==symbol), None)
        if not pos:
            return {"side": None}
        qty = float(pos.get("contracts") or 0)
        entry = float(pos.get("entryPrice") or 0)
        liq   = float(pos.get("liquidationPrice") or 0)
        mark  = float(pos.get("markPrice") or 0)
        side = None
        if qty > 0:
            side = "LONG" if entry <= mark else "SHORT"  # heuristic for one-way
        return {"raw":pos, "side":side, "qty":qty, "entry_price":entry, "liquidation_price":liq, "current_price":mark}
    except Exception:
        return {"side": None}


def fetch_equity(exchange) -> float:
    try:
        bal = exchange.fetch_balance(params={"type":"future"})
        usdt_total = bal.get("USDT",{}).get("total") or bal.get("total",{}).get("USDT")
        return float(usdt_total or 0.0)
    except Exception:
        return 0.0

