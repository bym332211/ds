from __future__ import annotations
import os
from typing import Iterable, List
import requests


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
_chat_ids = os.getenv("TELEGRAM_CHAT_ID", "").strip()


def _chat_id_list() -> List[str]:
    if not _chat_ids:
        return []
    return [c.strip() for c in _chat_ids.split(",") if c.strip()]


def _split(text: str, limit: int = 4000) -> List[str]:
    chunks: List[str] = []
    t = text or ""
    while t:
        if len(t) <= limit:
            chunks.append(t)
            break
        cut = t.rfind("\n\n", 0, limit)
        if cut == -1:
            cut = t.rfind(". ", 0, limit)
        if cut == -1:
            cut = limit
        chunks.append(t[:cut].strip())
        t = t[cut:].lstrip()
    return [c for c in chunks if c]


def send_telegram_text(message: str) -> None:
    """Send a text message to configured Telegram chat(s).

    No-op if token or chat id(s) are missing.
    """
    token = TELEGRAM_BOT_TOKEN
    chat_ids = _chat_id_list()
    if not token or not chat_ids:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for cid in chat_ids:
        for chunk in _split(message or ""):
            try:
                requests.post(url, data={"chat_id": cid, "text": chunk}, timeout=15)
            except Exception:
                # Silently ignore to avoid breaking trading loop
                pass

