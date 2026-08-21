"""
paper_trader.py

Live paper trading engine, runs inside the same 15s poll cycle as the main
scanner. No real orders are ever placed -- this only simulates entries/exits
against real live prices, so you can validate the strategy before risking
capital.

ENTRY: each poll cycle, checks the top N candidates by RVOL% (from the
already-computed live table) for the full delta-zone trigger (reuses
delta_zone_scanner.evaluate_stock -- same POC/VWAP/zone/RVOL/ATR logic, and
its Stop Loss/Target are used directly). Long-only, matching the scanner's
bullish-breakout convention. Caps how many candidates get the heavy 2-call
check per cycle, and how many positions can be open at once.

EXIT: uses the LTP/VWAP already fetched for the main table this cycle --
zero extra API calls. Checks stop, target, VWAP invalidation (Stage 4: exit
if price closes back below VWAP), and EOD square-off.

PERSISTENCE: state lives in memory for the running app session only. It
does NOT survive an app reboot or redeploy (Streamlit Cloud's filesystem
resets). Treat this as a same-session simulation log, not a permanent
trade journal -- export/screenshot results before rebooting if you want to
keep them.
"""

import threading
import time
import datetime
from zoneinfo import ZoneInfo

from delta_zone_scanner import evaluate_stock

IST = ZoneInfo("Asia/Kolkata")

MAX_OPEN_POSITIONS = 2          # Stage 7: rank simultaneous signals, take top 1-2
MAX_CANDIDATES_PER_CYCLE = 5    # cap heavy delta-zone checks per entry-check cycle
ENTRY_CHECK_INTERVAL_SECONDS = 60  # only look for new entries this often, NOT every
                                     # 15s poll cycle -- with slots open, this check
                                     # runs continuously, so at 15s intervals it was
                                     # adding ~20 extra API calls every cycle and
                                     # sustainably starving the main quote poll's
                                     # rate-limit budget instead of just a one-time burst.
EOD_SQUAREOFF_TIME = datetime.time(15, 20)  # IST, matches backtest_one_day.py
POSITION_QTY = 1  # shares per paper position -- this measures signal quality,
                   # not realistic position sizing. Change to reflect real
                   # intended quantity if you want P&L in a meaningful rupee scale.

_last_entry_check_time = None  # module-level, tracks last time find_new_entries actually ran


class PaperTradeState:
    """Thread-safe container, same pattern as ScannerState in fno_scanner_app.py."""

    def __init__(self):
        self.lock = threading.Lock()
        self.open_positions = {}   # symbol -> position dict
        self.closed_trades = []    # list of closed trade dicts
        self.enabled = False       # start OFF by default -- explicit opt-in

    def get(self):
        with self.lock:
            return dict(self.open_positions), list(self.closed_trades), self.enabled

    def set_enabled(self, value: bool):
        with self.lock:
            self.enabled = value

    def _open(self, symbol, entry, stop, target, entry_time):
        with self.lock:
            self.open_positions[symbol] = {
                "symbol": symbol, "side": "long", "entry": entry, "stop": stop,
                "target": target, "entry_time": entry_time, "qty": POSITION_QTY,
                "ltp": entry, "unrealized_pnl": 0.0,
            }

    def _update_ltp(self, symbol, ltp):
        with self.lock:
            pos = self.open_positions.get(symbol)
            if pos:
                pos["ltp"] = ltp
                pos["unrealized_pnl"] = round((ltp - pos["entry"]) * pos["qty"], 2)

    def _close(self, symbol, exit_price, reason, exit_time):
        with self.lock:
            pos = self.open_positions.pop(symbol, None)
            if not pos:
                return
            r_multiple = (exit_price - pos["entry"]) / (pos["entry"] - pos["stop"]) if pos["entry"] != pos["stop"] else 0
            pnl = round((exit_price - pos["entry"]) * pos["qty"], 2)
            self.closed_trades.append({
                **pos, "exit": exit_price, "exit_reason": reason, "exit_time": exit_time,
                "r_multiple": round(r_multiple, 2), "pnl": pnl,
            })


def _is_eod(now_ist: datetime.datetime) -> bool:
    return now_ist.time() >= EOD_SQUAREOFF_TIME


def manage_exits(pt_state: PaperTradeState, equity_by_token: dict, universe: dict, now_ist: datetime.datetime):
    """
    Check every open position against LTP/VWAP already fetched this poll
    cycle (equity_by_token) -- zero extra API calls.
    """
    open_positions, _, _ = pt_state.get()
    eod = _is_eod(now_ist)

    for symbol, pos in open_positions.items():
        keys = universe.get(symbol)
        if not keys:
            continue
        equity_quote = equity_by_token.get(keys["equity_key"])
        if not equity_quote:
            continue

        ltp = equity_quote.get("last_price", 0) or 0
        vwap = equity_quote.get("average_price", 0) or 0
        if not ltp:
            continue

        pt_state._update_ltp(symbol, ltp)

        if eod:
            pt_state._close(symbol, ltp, "eod_squareoff", now_ist.strftime("%H:%M:%S"))
        elif ltp <= pos["stop"]:
            pt_state._close(symbol, pos["stop"], "stop", now_ist.strftime("%H:%M:%S"))
        elif ltp >= pos["target"]:
            pt_state._close(symbol, pos["target"], "target", now_ist.strftime("%H:%M:%S"))
        elif vwap and ltp < vwap:
            pt_state._close(symbol, ltp, "vwap_invalidation", now_ist.strftime("%H:%M:%S"))


def find_new_entries(pt_state: PaperTradeState, live_df, universe: dict, access_token: str,
                      now_ist: datetime.datetime):
    """
    Check the top-RVOL candidates (from the already-computed live table) for
    a fresh delta-zone trigger. Only runs if paper trading is enabled, we're
    below the position cap, and we're not past EOD square-off (no new
    entries in the closing minutes).

    Throttled to ENTRY_CHECK_INTERVAL_SECONDS: with open slots available,
    this check would otherwise run every single 15s poll cycle indefinitely
    (not just once like the daily baseline build), which sustainably
    competes with the main quote poll for API rate-limit budget rather than
    causing a one-time, self-recovering burst.
    """
    global _last_entry_check_time

    open_positions, _, enabled = pt_state.get()
    if not enabled or _is_eod(now_ist):
        return
    if len(open_positions) >= MAX_OPEN_POSITIONS:
        return

    if _last_entry_check_time is not None:
        elapsed = (now_ist - _last_entry_check_time).total_seconds()
        if elapsed < ENTRY_CHECK_INTERVAL_SECONDS:
            return
    _last_entry_check_time = now_ist

    if live_df is None or live_df.empty or "RVOL %" not in live_df.columns:
        return

    slots_available = MAX_OPEN_POSITIONS - len(open_positions)
    candidates = (
        live_df[~live_df["Symbol"].isin(open_positions.keys())]
        .dropna(subset=["RVOL %"])
        .sort_values("RVOL %", ascending=False)
        .head(MAX_CANDIDATES_PER_CYCLE)["Symbol"]
        .tolist()
    )

    opened = 0
    for symbol in candidates:
        if opened >= slots_available:
            break
        keys = universe.get(symbol)
        if not keys:
            continue
        try:
            result = evaluate_stock(symbol, keys["equity_key"], access_token)
        except Exception as e:
            print(f"[paper_trader] skip {symbol}: {e}")
            continue
        if not result or not result.get("All Conditions Met"):
            continue
        if result.get("Stop Loss") is None or result.get("Target") is None:
            continue  # ATR% unavailable -- can't size the trade safely

        pt_state._open(symbol, result["LTP"], result["Stop Loss"], result["Target"],
                        now_ist.strftime("%H:%M:%S"))
        opened += 1
        time.sleep(0.15)  # same pacing as delta_zone_scanner's run_scan
