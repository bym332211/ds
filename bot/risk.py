from __future__ import annotations
from typing import Dict, Any, Tuple
import math


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
    if eff_stop<=0 or max_risk<=0:
        return 0.0
    return float(max_risk / eff_stop)


def market_info_rounders(exchange, symbol: str):
    m = exchange.market(symbol)
    # amount step
    amt_min = (m.get("limits",{}).get("amount",{}).get("min") or 0.0) or 0.001
    amt_step = amt_min  # conservative
    # price tick
    price_prec = m.get("precision",{}).get("price")
    if price_prec is not None:
        tick = 10**(-price_prec)
    else:
        tick = (m.get("limits",{}).get("price",{}).get("min") or 0.01)
    def round_qty(q): return max(amt_min, math.floor(q/amt_step)*amt_step)
    def round_price(p): return round(p / tick) * tick
    return round_qty, round_price, m


def compute_min_equity_required(exchange, symbol: str, entry_price: float, atr_val: float,
                                risk_fraction: float, atr_multiplier: float) -> Dict[str, Any]:
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

