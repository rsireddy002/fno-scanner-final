"""
rvol_atr_baseline.py

Builds a ONCE-PER-DAY cache of per-stock ATR% and average daily volume,
so the main live poll (every 15s, 208 stocks) can compute RVOL%, Stop Loss,
and Target WITHOUT fetching daily history on every single poll cycle --
that would mean 208 extra API calls every 15 seconds, which would blow
through Upstox's rate limits immediately.

Pattern mirrors fno_universe.py's daily JSON cache.

Usage:
    from rvol_atr_baseline import load_baseline
    baseline = load_baseline(universe, access_token)
    # {trading_symbol: {"avg_daily_volume": ..., "atr_pct": ...}}
"""

import os
import json
import time
import numpy as np

from upstox_downloads import download_daily_history

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, "rvol_atr_baseline.json")
CACHE_MAX_AGE_SECONDS = 24 * 60 * 60  # rebuild once a day

ATR_PERIOD = 14
RVOL_BASELINE_DAYS = 20

TS, O, H, L, C, V, OI = 0, 1, 2, 3, 4, 5, 6


def _to_sorted_numeric(candles):
    if not candles:
        return None
    rows = sorted(candles, key=lambda r: r[TS])
    return np.array([[float(r[i]) for i in range(1, 7)] for r in rows])  # o,h,l,c,v,oi


def _calc_atr_percent(daily, period=ATR_PERIOD):
    if daily is None or len(daily) < period + 1:
        return None
    high, low, close = daily[:, 1], daily[:, 2], daily[:, 3]
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = tr[-period:].mean()
    last_close = close[-1]
    return (atr / last_close) * 100 if last_close else None


def _calc_avg_daily_volume(daily, days=RVOL_BASELINE_DAYS):
    if daily is None or len(daily) < days:
        return None
    return float(daily[-days:, 4].mean())


def build_baseline(universe: dict, access_token: str, progress_callback=None) -> dict:
    """Fetch daily history once per stock, compute ATR% + avg volume, cache to disk."""
    baseline = {}
    total = len(universe)
    for i, (symbol, keys) in enumerate(universe.items()):
        try:
            equity_key = keys["equity_key"] if isinstance(keys, dict) else keys
            raw_daily = download_daily_history(equity_key, access_token, lookback_days=90)
            daily = _to_sorted_numeric(raw_daily)
            baseline[symbol] = {
                "avg_daily_volume": _calc_avg_daily_volume(daily),
                "atr_pct": _calc_atr_percent(daily),
            }
        except Exception as e:
            print(f"[rvol_atr_baseline] skip {symbol}: {e}")
            baseline[symbol] = {"avg_daily_volume": None, "atr_pct": None}
        if progress_callback:
            progress_callback((i + 1) / total)

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({"generated_at": time.time(), "baseline": baseline}, f, indent=2)
    return baseline


def load_baseline(universe: dict, access_token: str, force_refresh: bool = False,
                   progress_callback=None) -> dict:
    """Load from cache if fresh (<24h old), else rebuild."""
    if not force_refresh and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        age = time.time() - cached.get("generated_at", 0)
        if age < CACHE_MAX_AGE_SECONDS:
            return cached["baseline"]
    return build_baseline(universe, access_token, progress_callback)
