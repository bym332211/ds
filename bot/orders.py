from __future__ import annotations
from typing import Any, Dict
import os, time


def _symbol() -> str:
    return os.getenv("SYMBOL", "BNB/USDT")


def compute_limit_price(side: str, last: float, entry_block: Dict[str,Any]) -> float:
    # prefer explicit price
    price = entry_block.get("price")
    if price:
        return float(price)
    # else use offset_bps
    bps = entry_block.get("offset_bps")
    if bps is None:
        # fallback to env default if not provided
        try:
            from os import getenv
            bps = float(getenv("ENTRY_DEFAULT_OFFSET_BPS", "5"))
        except Exception:
            bps = 5.0
    bps = float(bps)
    if side=="LONG":
        # buy below last
        return last * (1.0 - bps/10000.0)
    else:
        # sell above last
        return last * (1.0 + bps/10000.0)


def place_market_with_protection(exchange, side: str, qty: float, sl_price: float, tp_price: float) -> Dict[str,Any]:
    symbol = _symbol()
    ord_side = "buy" if side=="LONG" else "sell"
    try:
        entry = exchange.create_order(symbol, "market", ord_side, qty, None, {"reduceOnly": False})
        print(f"ENTRY filled (market): id={entry.get('id')} qty={qty}")
    except Exception as e:
        print(f"Market entry failed: {e}")
        return {"ok": False, "error": str(e)}
    try:
        exit_side = "sell" if side=="LONG" else "buy"
        sl = exchange.create_order(symbol, "STOP_MARKET", exit_side, qty, None, {"stopPrice": sl_price, "reduceOnly": True})
        tp = exchange.create_order(symbol, "TAKE_PROFIT_MARKET", exit_side, qty, None, {"stopPrice": tp_price, "reduceOnly": True})
        print(f"SL={sl.get('id')} TP={tp.get('id')}")
        return {"ok": True, "entry": entry, "sl": sl, "tp": tp}
    except Exception as e:
        print(f"Protective orders failed: {e}")
        return {"ok": False, "error": str(e)}


def place_limit_postonly(exchange, side: str, qty: float, price: float) -> Dict[str,Any]:
    symbol = _symbol()
    ord_side = "buy" if side=="LONG" else "sell"
    params = {"postOnly": True, "timeInForce": "GTX"}
    return exchange.create_order(symbol, "limit", ord_side, qty, price, params)


def wait_fill_and_place_protection(exchange, side: str, order, sl_price: float, tp_price: float,
                                   max_wait_sec: int, poll_sec: int, cancel_on_timeout: bool) -> Dict[str,Any]:
    symbol = _symbol()
    order_id = order.get("id")
    deadline = time.time() + max_wait_sec
    filled_qty = 0.0
    last_o = None
    while time.time() < deadline:
        try:
            o = exchange.fetch_order(order_id, symbol)
            status = o.get("status")
            filled = float(o.get("filled") or 0.0)
            remaining = float(o.get("remaining") or 0.0)
            print(f"[LIMIT] status={status} filled={filled} remaining={remaining}")
            if status in ("closed","filled") or (filled > 0 and remaining <= 0):
                filled_qty = filled
                last_o = o
                break
            time.sleep(poll_sec)
        except Exception as e:
            print(f"fetch_order error: {e}")

    if filled_qty <= 0:
        if cancel_on_timeout:
            try:
                exchange.cancel_order(order_id, symbol)
                print(f"[LIMIT] Cancelled unfilled order {order_id}")
            except Exception as e:
                print(f"Cancel error: {e}")
        return {"ok": False, "error": "limit_not_filled"}

    # place protection for the filled amount only
    try:
        exit_side = "sell" if side=="LONG" else "buy"
        sl = exchange.create_order(symbol, "STOP_MARKET", exit_side, filled_qty, None, {"stopPrice": sl_price, "reduceOnly": True})
        tp = exchange.create_order(symbol, "TAKE_PROFIT_MARKET", exit_side, filled_qty, None, {"stopPrice": tp_price, "reduceOnly": True})
        print(f"[LIMIT] Filled qty={filled_qty}. SL={sl.get('id')} TP={tp.get('id')}")
        avg = 0.0
        try:
            avg = float((last_o or {}).get("average") or 0.0)
        except Exception:
            pass
        return {"ok": True, "entry": last_o or order, "sl": sl, "tp": tp, "filled_qty": filled_qty, "avg": avg}
    except Exception as e:
        print(f"Protective orders after limit fill failed: {e}")
        return {"ok": False, "error": str(e), "partial_filled_qty": filled_qty}

