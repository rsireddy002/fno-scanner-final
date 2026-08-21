"""
upstox_downloads.py

All network/download calls to Upstox live here in one place:
  - Instrument master file (for the dynamic F&O universe)
  - Full Market Quotes (for the live scanner polling)

Keeping every download function in one module means:
  - one place to fix rate-limiting / retry / timeout behaviour
  - fno_universe.py and fno_scanner_app.py just import + call, no
    duplicated requests logic
  - easy to add new downloads later (e.g. historical candles for
    RSI/ADX/SMA scoring) without touching the scanner or universe files
"""

import gzip
import json
import time
import requests

INSTRUMENT_MASTER_URL = (
    "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
)
FULL_QUOTE_URL = "https://api.upstox.com/v2/market-quote/quotes"
HISTORICAL_CANDLE_URL = "https://api.upstox.com/v2/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"
INTRADAY_CANDLE_URL = "https://api.upstox.com/v2/historical-candle/intraday/{instrument_key}/{interval}"

BATCH_SIZE = 500          # Upstox hard limit per quotes request
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0


def _chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _get_with_retry(url, headers=None, params=None, timeout=REQUEST_TIMEOUT):
    """Shared GET wrapper with basic 429 backoff, used by all downloads below."""
    last_exc = None
    last_status = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            if resp.status_code == 429:
                last_status = 429
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            last_exc = e
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))

    # If we get here, all MAX_RETRIES+1 attempts failed. If it was ONLY ever
    # 429s (never an actual RequestException), last_exc stays None -- raising
    # that directly crashes with "exceptions must derive from BaseException".
    # Raise a real, informative exception instead.
    if last_exc is not None:
        raise last_exc
    raise requests.exceptions.RequestException(
        f"Exhausted {MAX_RETRIES + 1} attempts, all rate-limited (HTTP 429): {url}"
    )


# ---------------------------------------------------------------------------
# 1. Instrument master download (used by fno_universe.py)
# ---------------------------------------------------------------------------
def download_instrument_master():
    """
    Download and parse the full Upstox instrument master (gzip JSON).
    Returns a list of instrument dicts covering every exchange/segment.
    """
    resp = _get_with_retry(INSTRUMENT_MASTER_URL)
    raw = gzip.decompress(resp.content)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# 2. Full market quotes download (used by fno_scanner_app.py)
# ---------------------------------------------------------------------------
def download_full_quotes(instrument_keys, access_token):
    """
    Fetch Full Market Quotes for a list of instrument_keys, batched at 500
    per request (Upstox's hard limit). Returns the merged {instrument_key: quote}
    dict across all batches.
    """
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    all_data = {}
    for batch in _chunk_list(instrument_keys, BATCH_SIZE):
        params = {"instrument_key": ",".join(batch)}
        resp = _get_with_retry(FULL_QUOTE_URL, headers=headers, params=params, timeout=15)
        payload = resp.json().get("data", {})
        all_data.update(payload)
    return all_data


# ---------------------------------------------------------------------------
# 3. Historical candles download (for future RSI/ADX/SMA scoring)
# ---------------------------------------------------------------------------
def download_historical_candles(instrument_key, interval, from_date, to_date, access_token):
    """
    Fetch historical OHLC candles for one instrument.
    interval: 'day' | '30minute' | '15minute' | etc. (per Upstox docs)
    from_date / to_date: 'YYYY-MM-DD'

    Not called yet by the scanner -- wire this in when you're ready to add
    RSI/ADX/SMA50/200 scoring on top of the live quote data.
    """
    headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
    url = HISTORICAL_CANDLE_URL.format(
        instrument_key=instrument_key, interval=interval, to_date=to_date, from_date=from_date
    )
    resp = _get_with_retry(url, headers=headers, timeout=30)
    return resp.json().get("data", {}).get("candles", [])


# ---------------------------------------------------------------------------
# 4. Today's intraday candles (for VWAP/POC — historical endpoint above does
#    NOT include today's incomplete session, this one does)
# ---------------------------------------------------------------------------
def download_intraday_candles(instrument_key, interval, access_token):
    """
    Fetch TODAY's candles so far for one instrument.
    interval: '1minute' | '5minute' | '30minute' etc. (matches historical endpoint options)

    Used for same-day VWAP/POC calculation in the delta-zone breakout scan.
    Returns candles in Upstox's native (usually newest-first) order --
    caller must sort by timestamp ascending before use, same as the
    historical endpoint.
    """
    headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
    url = INTRADAY_CANDLE_URL.format(instrument_key=instrument_key, interval=interval)
    resp = _get_with_retry(url, headers=headers, timeout=30)
    return resp.json().get("data", {}).get("candles", [])


# ---------------------------------------------------------------------------
# 5. Daily history convenience wrapper (for the delta-zone lookback window)
# ---------------------------------------------------------------------------
def download_daily_history(instrument_key, access_token, lookback_days=90):
    """
    Fetch the last `lookback_days` calendar days of DAILY candles for one
    instrument (excludes today, since today isn't closed yet). 90 calendar
    days comfortably covers 50+ trading days for the delta-zone lookback.
    """
    import datetime
    to_date = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    from_date = (datetime.date.today() - datetime.timedelta(days=lookback_days)).isoformat()
    return download_historical_candles(instrument_key, "day", from_date, to_date, access_token)


def download_daily_history_as_of(instrument_key, access_token, as_of_date, lookback_days=90):
    """
    Same as download_daily_history, but anchored to a PAST date instead of
    today -- for backtesting. Only returns daily candles strictly BEFORE
    as_of_date, so the delta-zone lookback can't see data from the future
    relative to the day being backtested (lookahead bias).

    as_of_date: 'YYYY-MM-DD' -- the day being backtested.
    """
    import datetime
    as_of = datetime.date.fromisoformat(as_of_date)
    to_date = (as_of - datetime.timedelta(days=1)).isoformat()
    from_date = (as_of - datetime.timedelta(days=lookback_days)).isoformat()
    return download_historical_candles(instrument_key, "day", from_date, to_date, access_token)


if __name__ == "__main__":
    # Quick smoke test for the instrument master download only
    # (quotes/candles need a live access token, so not tested here)
    print("Downloading instrument master...")
    instruments = download_instrument_master()
    print(f"Downloaded {len(instruments)} instruments.")
