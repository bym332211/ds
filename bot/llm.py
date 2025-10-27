from __future__ import annotations
import requests
import json
import logging
from datetime import datetime, timezone

log = logging.getLogger("bot")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_user_prompt(symbol: str,
                      multi_tfs: list[str],
                      entry_default_offset_bps: float,
                      equity_usd: float,
                      position: dict,
                      multi: dict,
                      feasibility: dict,
                      position_mode: str = "ONE_WAY",
                      min_rrr: float = 3.0,
                      risk_fraction: float = 0.02,
                      atr_multiplier: float = 1.5,
                      cooldown_bars_3m: int = 5) -> str:
    payload = {
        "market_metadata": {
            "exchange": "BINANCE", "symbol": symbol, "instrument_type": "PERP",
            "position_mode": position_mode, "timeframes": multi_tfs,
            "timezone": "Asia/Tokyo", "as_of": now_iso()
        },
        "risk_policy": {
            "risk_fraction_max": risk_fraction,
            "atr_multiplier_for_sl": atr_multiplier,
            "min_rrr": min_rrr,
            "cooldown_bars_3m": cooldown_bars_3m,
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
            "entry_default_offset_bps": entry_default_offset_bps
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


class DeepSeekClient:
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.deepseek.com/chat/completions"

    def complete(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.2, max_tokens: int = 700) -> str:
        # log prompts being sent
        try:
            log.info(f"LLM system prompt:\n{system_prompt}")
            log.info(f"LLM user prompt:\n{user_prompt}")
        except Exception:
            pass
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
