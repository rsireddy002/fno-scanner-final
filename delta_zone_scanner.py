"""
delta_zone_scanner.py

Screens the F&O universe for stocks where LTP is trading above ALL of:
    - POC (fixed for the day: VWAP of the first 5 one-minute candles, held
      constant once computed -- NOT a continuously updating day VWAP)
    - Session VWAP (continuously updating across today's candles so far)
    - Locked green delta-support zone (prior "strong buy" day's low + buffer)
    - Locked red delta-resistance zone (prior "strong sell" day's high + buffer)

Mirrors the Pine indicator you shared (Merged Value Areas + Volume Delta Zones).

Also computes, per stock, reusing the same fetched data (no extra API calls):
    - RVOL % (time-of-day-adjusted volume pace proxy vs 20-day average)
    - ATR % (actual 14-day Average True Range as % of price)
    - Suggested Stop Loss / Target (ATR%-sized, per Stage 4 of the strategy:
      stop = 1.25x ATR%, target = 1.75x that stop distance). LONG-only sizing,
      since this scan's trigger condition is a bullish breakout.

This is a HEAVIER scan than the RVOL scanner: for each stock it needs
(1) today's intraday candles and (2) ~90 days of daily candles, so two
Upstox calls per stock. Run it on a button / longer interval, not every
10s like the main quote poll.
"""

import numpy as np
import streamlit as st

from upstox_downloads import download_intraday_candles, download_daily_history

# ---------------- Config (mirrors Pine inputs) ----------------
FIRST_CANDLE_WINDOW = 5  # number of 1-min candles that form the "first 5 minutes" POC anchor
DELTA_LOOKBACK = 50
STRENGTH_THRESHOLD = 0.8
SMOOTH_LEN = 5
ZONE_WIDTH = 0.001
INTRADAY_INTERVAL = "1minute"  # '5minute' returns HTTP 400 on Upstox's intraday endpoint

ATR_PERIOD = 14
RVOL_BASELINE_DAYS = 20          # trading days to average for the RVOL volume baseline
SESSION_MINUTES = 375            # 9:15 - 15:30 IST
STOP_ATR_MULT = 1.25             # Stage 4: stop = 1-1.5x ATR%, using the midpoint
TARGET_R_MULTIPLE = 1.75         # Stage 4: target = 1.5-2x stop distance, using the midpoint

# Upstox candle row shape: [timestamp, open, high, low, close, volume, oi]
TS, O, H, L, C, V, OI = 0, 1, 2, 3, 4, 5, 6


def _to_sorted_array(candles):
    """Upstox candles aren't guaranteed sorted -- sort ascending by timestamp."""
    if not candles:
        return None
    arr = np.array(candles, dtype=object)
    arr = arr[np.argsort(arr[:, TS])]
    # numeric columns only (drop timestamp string for math)
    numeric = arr[:, 1:].astype(float)
    return numeric  # columns now: o, h, l, c, v, oi  (shifted by 1 vs TS,O,H,L,C,V,OI)


def _compute_session_vwap(today_5m: np.ndarray) -> float:
    close = today_5m[:, C - 1]   # -1 because timestamp column was dropped
    vol = today_5m[:, V - 1]
    if vol.sum() == 0:
        return float(close[-1])
    return float((close * vol).sum() / vol.sum())


def _compute_first_candle_poc(today_1m: np.ndarray, num_candles: int = FIRST_CANDLE_WINDOW) -> float:
    """
    POC = VWAP of the first `num_candles` one-minute candles of the session,
    computed ONCE and held fixed for the rest of the day -- this is the same
    "first 5-minute candle" anchor used in the chart-level support/resistance
    rule, not a continuously-updating day VWAP. If fewer than `num_candles`
    have printed yet (very start of session), uses whatever is available.
    """
    n = min(num_candles, len(today_1m))
    opening_window = today_1m[:n]
    close = opening_window[:, C - 1]
    vol = opening_window[:, V - 1]
    if vol.sum() == 0:
        return float(close[-1])
    return float((close * vol).sum() / vol.sum())


def _compute_cumulative_delta(daily: np.ndarray, smooth_len=SMOOTH_LEN):
    o, c, v = daily[:, O - 1], daily[:, C - 1], daily[:, V - 1]
    raw_delta = (c - o) * v
    return np.convolve(raw_delta, np.ones(smooth_len), mode="full")[: len(raw_delta)]


def calculate_atr_percent(daily: np.ndarray, period: int = ATR_PERIOD) -> float:
    """
    Actual ATR (true range, simple-averaged), expressed as % of latest close.
    Reuses the SAME daily array already fetched for the delta-zone lookback --
    no extra API call needed.
    """
    if daily is None or len(daily) < period + 1:
        return None
    high, low, close = daily[:, H - 1], daily[:, L - 1], daily[:, C - 1]
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = tr[-period:].mean()
    last_close = close[-1]
    return (atr / last_close) * 100 if last_close else None


def calculate_rvol_percent(daily: np.ndarray, today: np.ndarray,
                            baseline_days: int = RVOL_BASELINE_DAYS) -> float:
    """
    Time-of-day-adjusted RVOL, as a percentage (100% = exactly average pace).

    Proxy method (no extra API call -- reuses daily history already fetched
    for ATR/delta-zones, rather than your live scanner's full per-minute
    time-bucketed baseline): average full-day volume over the last
    `baseline_days`, scaled down to "expected volume by this point in the
    session" using elapsed-minutes / SESSION_MINUTES, then compared against
    today's actual cumulative volume so far.

    This is a coarser proxy than your dedicated RVOL scanner's true
    time-bucketed baseline -- treat it as directional, not precise.
    """
    if daily is None or len(daily) < baseline_days:
        return None
    avg_daily_volume = daily[-baseline_days:, V - 1].mean()
    if avg_daily_volume == 0 or today is None or len(today) == 0:
        return None

    elapsed_minutes = len(today)  # 1 row per minute since INTRADAY_INTERVAL = "1minute"
    elapsed_fraction = min(elapsed_minutes / SESSION_MINUTES, 1.0)
    expected_volume_by_now = avg_daily_volume * elapsed_fraction
    if expected_volume_by_now == 0:
        return None

    cumulative_volume_today = today[:, V - 1].sum()
    return (cumulative_volume_today / expected_volume_by_now) * 100


def _find_locked_zones(daily: np.ndarray, last_price: float,
                        lookback=DELTA_LOOKBACK, threshold=STRENGTH_THRESHOLD, zone_width=ZONE_WIDTH):
    if daily is None or len(daily) < lookback + 1:
        return None, None

    cum_delta = _compute_cumulative_delta(daily)
    support_low, resistance_high = None, None

    for i in range(lookback, len(daily)):
        window = cum_delta[i - lookback:i]
        max_d, min_d = window.max(), window.min()
        rng = max_d - min_d
        if rng == 0:
            continue
        if cum_delta[i] > (min_d + rng * threshold):
            support_low = daily[i, L - 1]
        if cum_delta[i] < (max_d - rng * threshold):
            resistance_high = daily[i, H - 1]

    support_zone_top = (support_low + last_price * zone_width) if support_low is not None else None
    resistance_zone_top = (resistance_high + last_price * zone_width) if resistance_high is not None else None
    return support_zone_top, resistance_zone_top


def evaluate_stock(symbol, instrument_key, access_token):
    """Returns a result dict, or None if data unavailable / conditions not met."""
    raw_intraday = download_intraday_candles(instrument_key, INTRADAY_INTERVAL, access_token)
    today = _to_sorted_array(raw_intraday)
    if today is None or len(today) == 0:
        return None

    raw_daily = download_daily_history(instrument_key, access_token, lookback_days=90)
    daily = _to_sorted_array(raw_daily)

    ltp = float(today[-1, C - 1])

    vwap = _compute_session_vwap(today)          # continuously updating, whole day so far
    poc = _compute_first_candle_poc(today)        # fixed once from first 5 one-min candles
    support_zone_top, resistance_zone_top = _find_locked_zones(daily, last_price=ltp)

    above_poc = ltp > poc
    above_vwap = ltp > vwap
    above_support = (ltp > support_zone_top) if support_zone_top is not None else True
    above_resistance = (ltp > resistance_zone_top) if resistance_zone_top is not None else True
    all_conditions_met = above_poc and above_vwap and above_support and above_resistance

    atr_pct = calculate_atr_percent(daily)
    rvol_pct = calculate_rvol_percent(daily, today)

    # Stage 4 sizing: stop = 1.25x ATR% from LTP, target = 1.75x that stop
    # distance. This scan is bullish-breakout-only (see module docstring),
    # so stop/target are always computed for a LONG -- stop below LTP,
    # target above. Only meaningful once ATR% is available.
    stop_loss, target = None, None
    if atr_pct is not None:
        stop_distance = ltp * (atr_pct / 100) * STOP_ATR_MULT
        stop_loss = ltp - stop_distance
        target = ltp + stop_distance * TARGET_R_MULTIPLE

    result = {
        "Symbol": symbol,
        "LTP": round(ltp, 2),
        "RVOL %": round(rvol_pct, 1) if rvol_pct is not None else None,
        "ATR %": round(atr_pct, 2) if atr_pct is not None else None,
        "Stop Loss": round(stop_loss, 2) if stop_loss is not None else None,
        "Target": round(target, 2) if target is not None else None,
        "POC": round(poc, 2),
        "VWAP": round(vwap, 2),
        "Support Zone Top": round(support_zone_top, 2) if support_zone_top else None,
        "Resistance Zone Top": round(resistance_zone_top, 2) if resistance_zone_top else None,
        "All Conditions Met": all_conditions_met,
    }
    return result


def run_scan(universe: dict, access_token: str, progress_callback=None) -> "pd.DataFrame":
    import pandas as pd
    import time
    rows = []
    total = len(universe)
    for i, (symbol, keys) in enumerate(universe.items()):
        try:
            equity_key = keys["equity_key"] if isinstance(keys, dict) else keys
            res = evaluate_stock(symbol, equity_key, access_token)
            if res:
                rows.append(res)
        except Exception as e:
            print(f"[delta_zone_scanner] skip {symbol}: {e}")
        if progress_callback:
            progress_callback((i + 1) / total)
        # Small pacing delay -- this scan makes 2 calls/stock (~400+ total),
        # and runs concurrently with the 10s live poll's own calls. Without
        # this, both compete for the same rate-limit budget and the live
        # poll starts throwing 429s.
        time.sleep(0.15)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[df["All Conditions Met"]].reset_index(drop=True)
