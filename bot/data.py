from __future__ import annotations
from typing import Dict, List
import pandas as pd
import numpy as np


def to_df(ohlcv: List[List[float]]) -> pd.DataFrame:
    if not ohlcv:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    arr = np.array(ohlcv, dtype=float)
    df = pd.DataFrame(arr, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def fetch_ohlcv_multi(exchange, symbol: str, timeframes: List[str], limit: int = 400,
                      since_ms: int | None = None) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, since=since_ms, limit=limit)
            out[tf] = to_df(ohlcv)
        except Exception:
            out[tf] = pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    return out

