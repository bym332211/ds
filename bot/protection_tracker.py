from __future__ import annotations
from typing import Dict, Any

from .telegram_notify import send_telegram_text


_tracked: Dict[str, Dict[str, Dict[str, Any]]] = {}


def register_protection(symbol: str, side: str, sl_order: Dict[str, Any] | None, tp_order: Dict[str, Any] | None) -> None:
    bucket = _tracked.setdefault(symbol, {})

    def _put(kind: str, o: Dict[str, Any] | None) -> None:
        if not o:
            return
        oid = str(o.get("id"))
        if not oid:
            return
        bucket[oid] = {
            "kind": kind,  # "SL" or "TP"
            "side": side,
            "notified": False,
        }

    _put("SL", sl_order)
    _put("TP", tp_order)


def poll_and_notify(exchange) -> None:
    """Poll tracked protection orders; notify when they are filled and then untrack."""
    for symbol, orders in list(_tracked.items()):
        for oid, meta in list(orders.items()):
            try:
                o = exchange.fetch_order(oid, symbol)
            except Exception:
                continue

            status = (o or {}).get("status")
            if status in ("closed", "filled") and not meta.get("notified"):
                kind = meta.get("kind")
                side = meta.get("side")
                filled = (o or {}).get("filled")
                avg = (o or {}).get("average") or (o or {}).get("price")
                try:
                    send_telegram_text(
                        f"[Protection Triggered]\n"
                        f"TYPE: {kind}\nSIDE: {side}\nSYMBOL: {symbol}\nQTY: {filled}\nAVG: {avg}\nORDER_ID: {oid}"
                    )
                except Exception:
                    pass
                meta["notified"] = True
                # remove from tracking once notified
                try:
                    del orders[oid]
                except Exception:
                    pass
        if not orders:
            try:
                del _tracked[symbol]
            except Exception:
                pass

