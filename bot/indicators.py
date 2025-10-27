from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd


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


def slope_per_bar(arr: np.ndarray) -> float:
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), arr.mean()
    denom = ((x-xm)**2).sum()
    return 0.0 if denom == 0 else float(((x-xm)*(arr-ym)).sum()/denom)


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
    if pmax <= pmin:
        pmax = pmin + 1e-6
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
        "stdev": float(np.std(arr, ddof=1)) if len(arr)>1 else 0.0,
        "slope": slope_per_bar(arr),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


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
        # RSI14 with bfill
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

