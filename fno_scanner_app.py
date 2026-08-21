"""
fno_scanner_app.py

F&O scanner using the DYNAMIC Upstox instrument universe (fno_universe.py)
instead of a hardcoded stock list. Polls Full Market Quotes via REST every
10 seconds in a background thread (the pattern you already confirmed
working on the office PC after the WebSocket issues), and renders a live
Streamlit table.

Run:
    streamlit run fno_scanner_app.py

Requires:
    UPSTOX_ACCESS_TOKEN set as an environment variable, OR pasted into the
    sidebar at runtime. Never hardcode tokens in this file.
"""

import os
import time
import threading
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st

from fno_universe import load_fno_universe
from upstox_downloads import download_full_quotes, BATCH_SIZE
from delta_zone_scanner import run_scan as run_delta_zone_scan
from rvol_atr_baseline import load_baseline
from paper_trader import PaperTradeState, manage_exits, find_new_entries

IST = ZoneInfo("Asia/Kolkata")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POLL_INTERVAL_SECONDS = 15

SESSION_START = (9, 15)   # IST
SESSION_MINUTES = 375     # 9:15 - 15:30
STOP_ATR_MULT = 1.25      # Stage 4: stop = 1.25x ATR%
TARGET_R_MULTIPLE = 1.75  # Stage 4: target = 1.75x that stop distance


def _elapsed_session_minutes() -> float:
    """Minutes elapsed since 9:15 IST today, capped at a full session."""
    import datetime
    now = datetime.datetime.now(IST)
    session_open = now.replace(hour=SESSION_START[0], minute=SESSION_START[1], second=0, microsecond=0)
    elapsed = (now - session_open).total_seconds() / 60
    return max(0.0, min(elapsed, SESSION_MINUTES))

st.set_page_config(page_title="F&O Scanner", layout="wide")


# ---------------------------------------------------------------------------
# Background polling worker
# ---------------------------------------------------------------------------
class ScannerState:
    """
    Thread-safe container for the latest scan results.
    Using a plain dict + lock (not st.session_state) inside the background
    thread avoids the deadlock issue you hit before with the scoring thread
    touching Streamlit state directly.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.df = pd.DataFrame()
        self.last_update = None
        self.last_error = None
        self.running = False
        self.debug_sample = None
        self.bad_universe_entries = []

    def set_result(self, df):
        with self.lock:
            self.df = df
            self.last_update = time.time()
            self.last_error = None

    def set_error(self, msg):
        with self.lock:
            self.last_error = msg

    def get(self):
        with self.lock:
            return self.df.copy(), self.last_update, self.last_error


def score_row(equity_quote, futures_quote, baseline_entry, elapsed_minutes):
    """
    Columns: LTP, Previous Close, % Chg, VWAP, RVOL %, SL, Target, Volume, OI, Score.

    LTP / % Chg / VWAP / Volume come from the live EQUITY quote (cheap, every poll).
    OI comes from the live FUTURES quote.
    RVOL % / SL / Target use the ONCE-PER-DAY cached baseline (ATR%, avg volume)
    combined with today's live volume/LTP -- no extra API call per poll.
    """
    ltp = equity_quote.get("last_price", 0) or 0
    net_change = equity_quote.get("net_change", 0) or 0
    previous_close = ltp - net_change
    volume = equity_quote.get("volume", 0) or 0
    vwap = equity_quote.get("average_price", 0) or 0  # Upstox's average_price IS VWAP

    oi = 0
    if futures_quote:
        oi = futures_quote.get("oi", 0) or 0

    pct_change = (net_change / previous_close * 100) if previous_close else 0

    # RVOL %: today's live volume vs expected-by-now volume from cached
    # 20-day average, scaled by elapsed session time.
    rvol_pct = None
    avg_daily_volume = (baseline_entry or {}).get("avg_daily_volume")
    if avg_daily_volume and elapsed_minutes > 0:
        expected_by_now = avg_daily_volume * (elapsed_minutes / SESSION_MINUTES)
        if expected_by_now > 0:
            rvol_pct = (volume / expected_by_now) * 100

    # SL / Target: Stage 4 sizing off the cached ATR%. Long-side sizing
    # (stop below LTP, target above) -- matches the Delta Zone scan's
    # bullish-breakout convention.
    stop_loss, target = None, None
    atr_pct = (baseline_entry or {}).get("atr_pct")
    if atr_pct and ltp:
        stop_distance = ltp * (atr_pct / 100) * STOP_ATR_MULT
        stop_loss = ltp - stop_distance
        target = ltp + stop_distance * TARGET_R_MULTIPLE

    vwap_pct = ((ltp - vwap) / vwap * 100) if vwap else 0
    score = pct_change + (0.3 * vwap_pct)

    return {
        "LTP": round(ltp, 2),
        "Previous Close": round(previous_close, 2),
        "% Chg": round(pct_change, 2),
        "VWAP": round(vwap, 2),
        "RVOL %": round(rvol_pct, 1) if rvol_pct is not None else None,
        "SL": round(stop_loss, 2) if stop_loss is not None else None,
        "Target": round(target, 2) if target is not None else None,
        "Volume": int(volume),
        "OI": int(oi),
        "Score": round(score, 2),
    }


def poll_loop(state: ScannerState, universe: dict, access_token: str, stop_event: threading.Event,
              baseline: dict, pt_state: PaperTradeState):
    state.running = True

    # Defensive: only keep entries with the expected shape. If universe ever
    # contains a malformed entry (e.g. stale cache from an older schema, or
    # a stray non-dict value), skip it instead of crashing the whole poll
    # with "sequence item 0: expected str instance, dict found".
    bad_entries = []
    equity_keys, futures_keys = [], []
    equity_key_to_symbol, fut_key_by_symbol = {}, {}
    for symbol, v in universe.items():
        if not isinstance(v, dict) or not isinstance(v.get("equity_key"), str):
            bad_entries.append(symbol)
            continue
        equity_keys.append(v["equity_key"])
        equity_key_to_symbol[v["equity_key"]] = symbol
        fk = v.get("futures_key")
        if isinstance(fk, str):
            futures_keys.append(fk)
            fut_key_by_symbol[symbol] = fk

    if bad_entries:
        print(f"[poll_loop] Skipping {len(bad_entries)} malformed universe entries: {bad_entries[:10]}"
              f"{'...' if len(bad_entries) > 10 else ''}")
    state.bad_universe_entries = bad_entries

    while not stop_event.is_set():
        try:
            equity_quotes = download_full_quotes(equity_keys, access_token)
            futures_quotes = download_full_quotes(futures_keys, access_token) if futures_keys else {}

            # Upstox's outer response dict key isn't reliably the pipe-format
            # instrument_key we requested with -- but the 'instrument_token'
            # field INSIDE each quote body is. Re-index on that instead.
            equity_by_token = {
                q.get("instrument_token"): q for q in equity_quotes.values() if q.get("instrument_token")
            }
            futures_by_token = {
                q.get("instrument_token"): q for q in futures_quotes.values() if q.get("instrument_token")
            }

            elapsed_minutes = _elapsed_session_minutes()

            rows = []
            debug_sample = None
            for symbol, keys in universe.items():
                equity_quote = equity_by_token.get(keys["equity_key"])
                if not equity_quote:
                    continue  # no live quote for this stock this cycle

                fut_key = fut_key_by_symbol.get(symbol)
                futures_quote = futures_by_token.get(fut_key) if fut_key else None
                baseline_entry = baseline.get(symbol)

                row = {"Symbol": symbol}
                row.update(score_row(equity_quote, futures_quote, baseline_entry, elapsed_minutes))
                rows.append(row)

                if debug_sample is None:
                    debug_sample = {"symbol": symbol, "equity_quote": equity_quote, "futures_quote": futures_quote}

            df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
            df.index = df.index + 1
            df.index.name = "Sl No"
            state.set_result(df)
            state.debug_sample = debug_sample

            # ---- Paper trading: exits are free (reuse this cycle's quotes),
            # entries are bounded (top RVOL candidates only, capped count) ----
            import datetime
            now_ist = datetime.datetime.now(IST)
            manage_exits(pt_state, equity_by_token, universe, now_ist)
            find_new_entries(pt_state, df, universe, access_token, now_ist)

        except Exception as e:
            # Keep the last good dataframe visible on screen rather than
            # blanking it -- a single rate-limited cycle (e.g. while the
            # heavy Delta Zone scan is also running) will self-recover on
            # the next 10s tick, so don't alarm the user or lose the table.
            state.set_error(str(e))

        stop_event.wait(POLL_INTERVAL_SECONDS)
    state.running = False


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.title("F&O Scanner — Dynamic Universe")

    with st.sidebar:
        st.header("Setup")
        default_token = os.environ.get("UPSTOX_ACCESS_TOKEN", "")
        access_token = st.text_input(
            "Upstox Access Token",
            value=default_token,
            type="password",
            help="Set UPSTOX_ACCESS_TOKEN env var to avoid pasting this every run. "
                 "Regenerate the token after any accidental sharing.",
        )
        refresh_universe = st.button("Refresh F&O universe now")
        refresh_baseline = st.button("Refresh RVOL/ATR baseline now (slow, ~1x/day needed)")
        st.caption(f"Polling every {POLL_INTERVAL_SECONDS}s · batches of {BATCH_SIZE}")

    if not access_token:
        st.warning("Enter your Upstox access token in the sidebar to start scanning.")
        st.stop()

    # Load (and optionally force-refresh) the dynamic F&O universe
    universe = load_fno_universe(force_refresh=refresh_universe)
    st.caption(f"Tracking {len(universe)} F&O-eligible stocks (auto-detected, not hardcoded).")

    # Load (and optionally force-refresh) the once-per-day RVOL/ATR baseline.
    # This is the SLOW part (one daily-history call per stock) -- cached to
    # disk so it only actually runs once a day, not on every app restart.
    if "baseline_cache" not in st.session_state or refresh_baseline:
        with st.spinner("Loading RVOL/ATR baseline (cached daily, first load is slower)..."):
            st.session_state.baseline_cache = load_baseline(
                universe, access_token, force_refresh=refresh_baseline
            )
    baseline = st.session_state.baseline_cache

    # Set up background polling thread once per session
    if "scanner_state" not in st.session_state:
        st.session_state.scanner_state = ScannerState()
        st.session_state.pt_state = PaperTradeState()
        st.session_state.stop_event = threading.Event()
        thread = threading.Thread(
            target=poll_loop,
            args=(st.session_state.scanner_state, universe, access_token, st.session_state.stop_event,
                  baseline, st.session_state.pt_state),
            daemon=True,
        )
        thread.start()
        st.session_state.scanner_thread = thread

    state: ScannerState = st.session_state.scanner_state
    df, last_update, error = state.get()

    status_col, _ = st.columns([3, 1])
    with status_col:
        if error:
            st.error(f"Last poll error: {error}")
        elif last_update:
            import datetime
            ist_time = datetime.datetime.fromtimestamp(last_update, tz=IST)
            st.success(f"Last updated: {ist_time.strftime('%H:%M:%S')} IST")
        else:
            st.info("Waiting for first poll...")

    if not df.empty:
        st.dataframe(df, width='stretch', height=700)
    else:
        st.info("No data yet — first poll can take a few seconds.")

    with st.expander("Debug: raw quote sample (check this if % Chg or OI still show 0)"):
        if state.debug_sample:
            st.json(state.debug_sample)
        else:
            st.caption("No sample captured yet — wait for the first poll.")
        if state.bad_universe_entries:
            st.warning(
                f"{len(state.bad_universe_entries)} universe entries were skipped for having "
                f"the wrong shape: {state.bad_universe_entries[:15]}"
            )

    # -----------------------------------------------------------------
    # Delta Zone Breakout scan (on-demand — 2 API calls per stock, so
    # this is NOT run inside the 10s poll loop)
    # -----------------------------------------------------------------
    st.divider()
    st.subheader("Delta Zone Breakout Scan")
    st.caption(
        "Flags stocks currently above POC, VWAP, the locked green support zone, "
        "and the locked red resistance zone — run manually, not every 10s."
    )
    if st.button("Run Delta Zone Scan"):
        progress_bar = st.progress(0.0)
        breakout_df = run_delta_zone_scan(
            universe, access_token, progress_callback=progress_bar.progress
        )
        progress_bar.empty()
        if breakout_df.empty:
            st.info("No stocks currently meet all four conditions.")
        else:
            st.success(f"{len(breakout_df)} stock(s) above POC, VWAP, and both delta zones:")
            st.dataframe(breakout_df, width='stretch')

    # -----------------------------------------------------------------
    # Paper Trading (simulated -- no real orders). Runs inside the same
    # poll cycle: exits are free, entries check only top-RVOL candidates.
    # State resets on app reboot/redeploy (Streamlit Cloud filesystem is
    # ephemeral) -- this is a same-session log, not a permanent journal.
    # -----------------------------------------------------------------
    st.divider()
    st.subheader("Paper Trading (Simulated)")
    st.caption(
        f"Long-only, opens on the same delta-zone trigger as the scan above. "
        f"Max {2} open positions, checks top RVOL candidates each cycle. "
        f"No real orders are ever placed. Resets if the app reboots."
    )

    pt_state: PaperTradeState = st.session_state.pt_state
    open_positions, closed_trades, enabled = pt_state.get()

    toggle_col, _ = st.columns([2, 3])
    with toggle_col:
        new_enabled = st.toggle("Enable paper trading", value=enabled)
        if new_enabled != enabled:
            pt_state.set_enabled(new_enabled)

    if open_positions:
        st.write("**Open Positions**")
        st.dataframe(pd.DataFrame(open_positions.values()), width='stretch')
    else:
        st.caption("No open positions.")

    if closed_trades:
        st.write("**Closed Trades**")
        trades_df = pd.DataFrame(closed_trades)
        st.dataframe(trades_df, width='stretch')
        total_r = trades_df["r_multiple"].sum()
        wins = (trades_df["r_multiple"] > 0).sum()
        st.success(f"Total R: {total_r:.2f} | Win rate: {wins}/{len(trades_df)} ({100*wins/len(trades_df):.0f}%)")
    else:
        st.caption("No closed trades yet.")

    # Lightweight auto-refresh of the UI (data itself refreshes in the background thread)
    time.sleep(2)
    st.rerun()


if __name__ == "__main__":
    main()
