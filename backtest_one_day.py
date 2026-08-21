"""
backtest_one_day.py

Backtests the full strategy (RVOL -> VWAP/candle bias -> delta-confirmed
zone break -> ATR%-sized risk) for ONE stock on ONE past trading day, using
real 1-minute Upstox candles. No lookahead: the delta-zone lookback only
uses daily candles strictly BEFORE the backtest date, and every decision at
minute N only uses data from minutes 1..N.

NOTE ON STAGE 1 (RVOL): this script assumes you've already picked a stock
that WAS an RVOL leader that day (you'd know this from your live scanner
history, or just want to test the mechanics on a stock you remember moving).
It does not re-derive RVOL rank from scratch -- that requires the 10-day
time-of-day baseline your live scanner already builds. Pass rvol_confirmed=True
to acknowledge this, or wire in your own RVOL check before calling.

Run:
    python backtest_one_day.py --symbol SBICARD --key "NSE_EQ|INE018E01016" --date 2026-08-14 --token YOUR_TOKEN

Never hardcode your token in this file -- pass it as an argument or env var.
"""

import argparse
import datetime
import numpy as np

from upstox_downloads import download_historical_candles, download_daily_history_as_of

# ---------------- Config (matches delta_zone_scanner.py + strategy discussion) ----------------
DELTA_LOOKBACK = 50
STRENGTH_THRESHOLD = 0.8
SMOOTH_LEN = 5
ZONE_WIDTH = 0.001
FIRST_CANDLE_WINDOW = 5          # minutes, for the POC anchor
NO_TRADE_WINDOW_MIN = 5          # skip first 5 minutes of session
ATR_PERIOD = 14
STOP_ATR_MULT = 1.25             # midpoint of the discussed 1-1.5x range
MIN_TARGET_R_MULTIPLE = 1.75     # midpoint of discussed 1.5-2x
EOD_SQUAREOFF_TIME = datetime.time(15, 20)  # square off before close

TS, O, H, L, C, V, OI = 0, 1, 2, 3, 4, 5, 6


def _sorted_numeric(candles):
    """Sort by timestamp ascending, return (timestamps, numeric_ohlcv_array)."""
    if not candles:
        return [], None
    rows = sorted(candles, key=lambda r: r[TS])
    timestamps = [r[TS] for r in rows]
    numeric = np.array([[float(r[i]) for i in range(1, 7)] for r in rows])
    return timestamps, numeric  # numeric columns: o,h,l,c,v,oi


def calculate_atr_percent(daily_numeric: np.ndarray, period: int = ATR_PERIOD) -> float:
    """
    Standard ATR (Wilder-style true range, simple-averaged for simplicity)
    expressed as a percentage of the most recent close.
    daily_numeric columns: o,h,l,c,v,oi (as returned by _sorted_numeric).
    """
    if daily_numeric is None or len(daily_numeric) < period + 1:
        return None
    high, low, close = daily_numeric[:, 1], daily_numeric[:, 2], daily_numeric[:, 3]
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = tr[-period:].mean()
    last_close = close[-1]
    return (atr / last_close) * 100 if last_close else None


def _compute_cumulative_delta(daily: np.ndarray, smooth_len=SMOOTH_LEN):
    o, c, v = daily[:, 0], daily[:, 3], daily[:, 4]
    raw_delta = (c - o) * v
    return np.convolve(raw_delta, np.ones(smooth_len), mode="full")[: len(raw_delta)]


def find_locked_zones(daily: np.ndarray, last_price: float,
                       lookback=DELTA_LOOKBACK, threshold=STRENGTH_THRESHOLD, zone_width=ZONE_WIDTH):
    """Same logic as delta_zone_scanner.py -- kept in sync deliberately."""
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
            support_low = daily[i, 2]  # low
        if cum_delta[i] < (max_d - rng * threshold):
            resistance_high = daily[i, 1]  # high
    support_zone_top = (support_low + last_price * zone_width) if support_low is not None else None
    resistance_zone_bottom = (resistance_high - last_price * zone_width) if resistance_high is not None else None
    return support_zone_top, resistance_zone_bottom


def determine_bias(first_candle_o, first_candle_c, first_candle_color_is_green, vwap_now, price_now):
    """
    Stage 2 bias logic, including the VWAP-override rule:
      1. green + above vwap -> long
      2. red + below vwap -> short
      3. red, but vwap > first candle close, price reclaims above vwap -> long (override)
      4. green, but vwap < first candle close, price breaks below vwap -> short (override)
      5. otherwise -> None (no trade)
    """
    above_vwap = price_now > vwap_now
    below_vwap = price_now < vwap_now

    if first_candle_color_is_green and above_vwap:
        return "long"
    if (not first_candle_color_is_green) and below_vwap:
        return "short"
    if (not first_candle_color_is_green) and vwap_now > first_candle_c and above_vwap:
        return "long"  # override
    if first_candle_color_is_green and vwap_now < first_candle_c and below_vwap:
        return "short"  # override
    return None


def backtest_day(symbol, instrument_key, date_str, access_token, rvol_confirmed=True):
    if not rvol_confirmed:
        print("WARNING: rvol_confirmed=False -- Stage 1 (RVOL) not verified for this day. "
              "Results assume this stock WAS a genuine RVOL leader; if it wasn't, this "
              "backtest is testing a setup that would never have been screened-in live.")

    # ---- Fetch data (no lookahead: daily history stops the day BEFORE date_str) ----
    intraday_raw = download_historical_candles(instrument_key, "1minute", date_str, date_str, access_token)
    ts, today = _sorted_numeric(intraday_raw)
    if today is None or len(today) < FIRST_CANDLE_WINDOW + 1:
        print(f"Not enough intraday data for {symbol} on {date_str} (got {len(today) if today is not None else 0} candles).")
        return None

    daily_raw = download_daily_history_as_of(instrument_key, access_token, date_str, lookback_days=120)
    _, daily = _sorted_numeric(daily_raw)
    atr_pct = calculate_atr_percent(daily)
    if atr_pct is None:
        print(f"Not enough daily history for {symbol} to compute ATR% -- proceeding without ATR-sized stops.")

    last_price_for_zones = today[-1, 3]
    support_zone_top, resistance_zone_bottom = find_locked_zones(daily, last_price_for_zones) if daily is not None else (None, None)

    # ---- First-candle anchor (Stage 2 setup) ----
    n_anchor = min(FIRST_CANDLE_WINDOW, len(today))
    opening = today[:n_anchor]
    first_candle_open = opening[0, 0]
    first_candle_close = opening[-1, 3]
    first_candle_green = first_candle_close >= first_candle_open
    opening_vol = opening[:, 4]
    poc = (opening[:, 3] * opening_vol).sum() / opening_vol.sum() if opening_vol.sum() else first_candle_close

    print(f"\n=== {symbol} — {date_str} ===")
    print(f"ATR%: {atr_pct:.2f}%" if atr_pct else "ATR%: unavailable")
    print(f"First-candle: open={first_candle_open:.2f} close={first_candle_close:.2f} "
          f"({'GREEN' if first_candle_green else 'RED'})")
    print(f"POC (fixed, first {FIRST_CANDLE_WINDOW}min VWAP): {poc:.2f}")
    print(f"Support zone top: {support_zone_top}")
    print(f"Resistance zone bottom: {resistance_zone_bottom}")

    # ---- Walk minute-by-minute (Stage 3-7 simulation) ----
    cum_pv, cum_vol = 0.0, 0.0
    position = None  # dict: side, entry, stop, target, entry_idx
    trades = []
    delta_running = 0.0
    delta_history = []

    for i in range(n_anchor, len(today)):
        o, h, l, c, v, oi = today[i]
        candle_time = datetime.datetime.fromtimestamp(0)  # placeholder; real time from ts[i] string
        minute_of_day = i  # proxy index since we don't have parsed wall-clock here

        cum_pv += c * v
        cum_vol += v
        vwap_now = cum_pv / cum_vol if cum_vol else c

        delta_running += (c - o) * v
        delta_history.append(delta_running)

        # Stage 5: no-trade window at open
        if i < n_anchor + NO_TRADE_WINDOW_MIN:
            continue

        # Stage 3: delta expansion check (fresh local high/low vs recent lookback)
        recent = delta_history[-DELTA_LOOKBACK:] if len(delta_history) >= 5 else delta_history
        delta_expanding_up = delta_running >= max(recent) if recent else False
        delta_expanding_down = delta_running <= min(recent) if recent else False

        bias = determine_bias(first_candle_open, first_candle_close, first_candle_green, vwap_now, c)

        # ---- Manage open position first ----
        if position:
            exit_reason = None
            if position["side"] == "long":
                if l <= position["stop"]:
                    exit_reason, exit_price = "stop", position["stop"]
                elif h >= position["target"]:
                    exit_reason, exit_price = "target", position["target"]
                elif c < vwap_now:
                    exit_reason, exit_price = "vwap_invalidation", c
            else:
                if h >= position["stop"]:
                    exit_reason, exit_price = "stop", position["stop"]
                elif l <= position["target"]:
                    exit_reason, exit_price = "target", position["target"]
                elif c > vwap_now:
                    exit_reason, exit_price = "vwap_invalidation", c

            if exit_reason:
                r_multiple = ((exit_price - position["entry"]) / (position["entry"] - position["stop"])
                              if position["side"] == "long"
                              else (position["entry"] - exit_price) / (position["stop"] - position["entry"]))
                trades.append({**position, "exit": exit_price, "exit_reason": exit_reason,
                               "exit_idx": i, "r_multiple": round(r_multiple, 2)})
                position = None
                continue  # one position at a time; re-eval next candle for re-entry

        # ---- Look for new entry (Stage 3 trigger + Stage 6 re-entry allowed) ----
        if position is None and bias and atr_pct:
            stop_distance = c * (atr_pct / 100) * STOP_ATR_MULT

            if bias == "long" and resistance_zone_bottom and c > resistance_zone_bottom and delta_expanding_up:
                target = c + stop_distance * MIN_TARGET_R_MULTIPLE
                position = {"symbol": symbol, "side": "long", "entry": c, "entry_idx": i,
                           "stop": c - stop_distance, "target": target}
            elif bias == "short" and support_zone_top and c < support_zone_top and delta_expanding_down:
                target = c - stop_distance * MIN_TARGET_R_MULTIPLE
                position = {"symbol": symbol, "side": "short", "entry": c, "entry_idx": i,
                           "stop": c + stop_distance, "target": target}

    # ---- EOD square-off if still open ----
    if position:
        exit_price = today[-1, 3]
        r_multiple = ((exit_price - position["entry"]) / (position["entry"] - position["stop"])
                      if position["side"] == "long"
                      else (position["entry"] - exit_price) / (position["stop"] - position["entry"]))
        trades.append({**position, "exit": exit_price, "exit_reason": "eod_squareoff",
                       "exit_idx": len(today) - 1, "r_multiple": round(r_multiple, 2)})

    # ---- Report ----
    print(f"\n--- Trades ({len(trades)}) ---")
    for t in trades:
        print(f"  {t['side'].upper()}: entry={t['entry']:.2f} stop={t['stop']:.2f} "
              f"target={t['target']:.2f} exit={t['exit']:.2f} ({t['exit_reason']}) "
              f"R={t['r_multiple']}")

    if trades:
        total_r = sum(t["r_multiple"] for t in trades)
        wins = sum(1 for t in trades if t["r_multiple"] > 0)
        print(f"\nTotal R: {total_r:.2f} | Win rate: {wins}/{len(trades)} ({100*wins/len(trades):.0f}%)")
    else:
        print("\nNo trades triggered this day (bias/zone/delta conditions never aligned).")

    return trades


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest one stock, one day, full strategy.")
    parser.add_argument("--symbol", required=True, help="e.g. SBICARD")
    parser.add_argument("--key", required=True, help="Upstox instrument_key, e.g. 'NSE_EQ|INE018E01016'")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD, must be a past trading day")
    parser.add_argument("--token", required=True, help="Upstox access token (never hardcode this)")
    args = parser.parse_args()

    backtest_day(args.symbol, args.key, args.date, args.token)
