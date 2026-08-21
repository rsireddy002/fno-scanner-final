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
import requests
import pandas as pd
import streamlit as st

from fno_universe import load_fno_universe

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUOTE_URL = "https://api.upstox.com/v2/market-quote/quotes"
BATCH_SIZE = 500          # Upstox hard limit per request
POLL_INTERVAL_SECONDS = 10

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


def chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def fetch_quotes(instrument_keys, access_token):
    """Fetch full market quotes for a list of instrument_keys, batched at 500."""
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    all_data = {}
    for batch in chunk_list(instrument_keys, BATCH_SIZE):
        params = {"instrument_key": ",".join(batch)}
        resp = requests.get(QUOTE_URL, headers=headers, params=params, timeout=15)
        if resp.status_code == 429:
            # Rate limited -- back off and retry once
            time.sleep(1.5)
            resp = requests.get(QUOTE_URL, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json().get("data", {})
        all_data.update(payload)
    return all_data


def score_row(quote):
    """
    Lightweight composite score from fields available on the Full Market
    Quote response. Swap this out for your fuller stock_dashboard.py model
    (RSI/ADX/SMA50/200/OI quadrants) once you wire in historical candles --
    this REST quote endpoint only gives live snapshot fields, not history.
    """
    ltp = quote.get("last_price", 0) or 0
    close = quote.get("ohlc", {}).get("close", 0) or 0
    volume = quote.get("volume", 0) or 0
    oi = quote.get("oi", 0) or 0
    avg_price = quote.get("average_price", 0) or 0

    pct_change = ((ltp - close) / close * 100) if close else 0
    vwap_pct = ((ltp - avg_price) / avg_price * 100) if avg_price else 0

    # simple momentum-style score: price change weighted, nudged by VWAP side
    score = pct_change + (0.3 * vwap_pct)

    return {
        "LTP": round(ltp, 2),
        "% Chg": round(pct_change, 2),
        "Volume": int(volume),
        "OI": int(oi),
        "VWAP %": round(vwap_pct, 2),
        "Score": round(score, 2),
    }


def poll_loop(state: ScannerState, universe: dict, access_token: str, stop_event: threading.Event):
    state.running = True
    while not stop_event.is_set():
        try:
            quotes = fetch_quotes(list(universe.values()), access_token)

            rows = []
            key_to_symbol = {v: k for k, v in universe.items()}
            for inst_key, quote in quotes.items():
                symbol = key_to_symbol.get(inst_key, quote.get("symbol", inst_key))
                row = {"Symbol": symbol}
                row.update(score_row(quote))
                rows.append(row)

            df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
            state.set_result(df)

        except Exception as e:
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
        st.caption(f"Polling every {POLL_INTERVAL_SECONDS}s · batches of {BATCH_SIZE}")

    if not access_token:
        st.warning("Enter your Upstox access token in the sidebar to start scanning.")
        st.stop()

    # Load (and optionally force-refresh) the dynamic F&O universe
    universe = load_fno_universe(force_refresh=refresh_universe)
    st.caption(f"Tracking {len(universe)} F&O-eligible stocks (auto-detected, not hardcoded).")

    # Set up background polling thread once per session
    if "scanner_state" not in st.session_state:
        st.session_state.scanner_state = ScannerState()
        st.session_state.stop_event = threading.Event()
        thread = threading.Thread(
            target=poll_loop,
            args=(st.session_state.scanner_state, universe, access_token, st.session_state.stop_event),
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
            st.success(f"Last updated: {time.strftime('%H:%M:%S', time.localtime(last_update))} IST")
        else:
            st.info("Waiting for first poll...")

    if not df.empty:
        st.dataframe(df, use_container_width=True, height=700)
    else:
        st.info("No data yet — first poll can take a few seconds.")

    # Lightweight auto-refresh of the UI (data itself refreshes in the background thread)
    time.sleep(2)
    st.rerun()


if __name__ == "__main__":
    main()
