"""
fno_universe.py

Dynamically builds the list of F&O-eligible NSE equity stocks using the
Upstox instrument master file, instead of a hardcoded stock list.

Source: https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz

Logic:
1. Download + parse the full instrument master (gzip JSON).
2. Find all NSE_FO FUT (stock futures) contracts -> gives us the set of
   underlying stock names that currently have an F&O contract, AND (new)
   the near-month futures instrument_key for each -- needed because
   equity/cash instruments never carry Open Interest; only the futures
   contract does.
3. Cross-reference those names against NSE_EQ / EQ instruments to get the
   equity instrument_key + trading_symbol needed for LTP/VWAP quoting.
4. Cache the result locally (fno_universe.json) so you don't re-download
   the ~30-50MB master file on every run. Refresh daily (F&O list changes
   rarely, but expiries/additions do happen -- especially around contract
   review cycles).

Usage:
    python fno_universe.py            # builds/refreshes fno_universe.json
    from fno_universe import load_fno_universe
    universe = load_fno_universe()
    # {trading_symbol: {"equity_key": ..., "futures_key": ..., "futures_expiry": ...}}
"""

import os
import json
import time

from upstox_downloads import download_instrument_master, INSTRUMENT_MASTER_URL

# Anchor all paths to this file's directory (per your VS Code working-dir fix)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, "fno_universe.json")
CACHE_MAX_AGE_SECONDS = 24 * 60 * 60  # refresh once a day

# Index futures never have a corresponding NSE_EQ instrument -- skip them.
INDEX_NAMES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}


def _parse_expiry(inst):
    """
    Normalize the instrument's expiry field to an epoch-milliseconds int
    so near-month contracts can be sorted/compared regardless of whether
    Upstox returns expiry as an int (ms) or an ISO date string.
    """
    expiry = inst.get("expiry")
    if expiry is None:
        return None
    if isinstance(expiry, (int, float)):
        return int(expiry)
    try:
        import datetime
        return int(datetime.date.fromisoformat(str(expiry)[:10]).strftime("%s")) * 1000
    except Exception:
        return None


def _build_universe(instruments):
    """
    From the full instrument list, derive per-stock:
        equity_key      -> for LTP / VWAP (Full Market Quote on the cash instrument)
        futures_key      -> near-month FUT contract, for Open Interest
        futures_expiry   -> expiry of the selected futures contract (for sanity checks)
    """
    now_ms = time.time() * 1000

    # name -> list of (expiry_ms, instrument_key, trading_symbol) across all live FUT expiries
    fut_candidates = {}
    for inst in instruments:
        if inst.get("segment") == "NSE_FO" and inst.get("instrument_type") == "FUT":
            name = (inst.get("name") or "").upper()
            if not name or name in INDEX_NAMES:
                continue
            expiry_ms = _parse_expiry(inst)
            instrument_key = inst.get("instrument_key")
            trading_symbol = inst.get("trading_symbol")
            if instrument_key:
                fut_candidates.setdefault(name, []).append((expiry_ms, instrument_key, trading_symbol))

    # Pick the near-month (soonest non-expired) contract per underlying
    fut_lookup = {}
    for name, candidates in fut_candidates.items():
        upcoming = [c for c in candidates if c[0] is not None and c[0] >= now_ms]
        chosen = min(upcoming, key=lambda c: c[0]) if upcoming else min(
            candidates, key=lambda c: (c[0] is None, c[0])
        )
        fut_lookup[name] = {"futures_key": chosen[1], "futures_expiry": chosen[0]}

    fo_underlyings = set(fut_lookup.keys())

    universe = {}
    for inst in instruments:
        if inst.get("segment") == "NSE_EQ" and inst.get("instrument_type") == "EQ":
            name = (inst.get("name") or "").upper()
            if name in fo_underlyings:
                trading_symbol = inst.get("trading_symbol")
                equity_key = inst.get("instrument_key")
                if trading_symbol and equity_key:
                    universe[trading_symbol] = {
                        "equity_key": equity_key,
                        "futures_key": fut_lookup[name]["futures_key"],
                        "futures_expiry": fut_lookup[name]["futures_expiry"],
                    }

    return universe


def refresh_fno_universe():
    """Force a fresh download + rebuild of the F&O universe cache."""
    print(f"[fno_universe] Downloading instrument master from {INSTRUMENT_MASTER_URL} ...")
    instruments = download_instrument_master()
    print(f"[fno_universe] Parsed {len(instruments)} total instruments.")

    universe = _build_universe(instruments)
    print(f"[fno_universe] Resolved {len(universe)} F&O-eligible equity stocks "
          f"(each with equity_key + futures_key).")

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"generated_at": time.time(), "universe": universe},
            f,
            indent=2,
        )
    print(f"[fno_universe] Cached to {CACHE_FILE}")
    return universe


def load_fno_universe(force_refresh=False):
    """
    Load the F&O universe, using the local cache if it's fresh enough.
    Returns {trading_symbol: {"equity_key", "futures_key", "futures_expiry"}}.
    """
    if not force_refresh and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        age = time.time() - cached.get("generated_at", 0)
        # Guard against loading a cache built by the OLD schema (plain str values)
        sample = next(iter(cached.get("universe", {}).values()), None)
        if age < CACHE_MAX_AGE_SECONDS and isinstance(sample, dict):
            return cached["universe"]
        print(f"[fno_universe] Cache stale or old schema, refreshing...")

    return refresh_fno_universe()


if __name__ == "__main__":
    universe = refresh_fno_universe()
    print(f"\nSample (first 5): {list(universe.items())[:5]}")
