"""
core/models/signal_recalibrator.py
────────────────────────────────────
Live performance tracker that adapts confidence thresholds based on
recent trade outcomes — without touching the underlying ML model.

This is the correct approach to "adapting to current market conditions":
instead of corrupting the trained model with 5-10 live samples, we track
what is working in recent bars and adjust the gate that decides whether
to act on a signal.

How it works
------------
1. Every closed trade is logged (symbol, side, outcome, confidence, regime).
2. A rolling window of the last WINDOW_SIZE trades is maintained per symbol.
3. The recalibrator computes a rolling win rate and average confidence of
   winners vs losers.
4. If win rate drops below PAUSE_THRESHOLD for a direction, that direction
   gets a confidence penalty (higher bar to enter).
5. If win rate is above BOOST_THRESHOLD, we allow slightly lower confidence.
6. All adjustments are bounded: max ±ADJUSTMENT_CAP pp from base threshold.

Example:
    Base threshold: 68%
    Recent SHORT win rate: 25% over last 10 trades
    Penalty applied: +5pp -> effective SHORT threshold = 73%
    Recent LONG win rate: 70% over last 10 trades
    Boost applied: -2pp -> effective LONG threshold = 66%

Usage:
    recalibrator = SignalRecalibrator()
    recalibrator.record_trade("BTCUSDT", "short", won=False, confidence=0.71)
    adj = recalibrator.get_threshold_adjustment("BTCUSDT", "short")
    effective_threshold = base_threshold + adj
"""

from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

from config.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeRecord:
    timestamp:  str
    symbol:     str
    side:       str        # "long" or "short"
    won:        bool
    confidence: float
    pnl_usdt:   float
    pnl_r:      float
    bars_held:  int
    exit_reason: str
    regime:     str


class SignalRecalibrator:
    """
    Tracks live trade outcomes and adjusts per-symbol per-direction
    confidence thresholds without modifying the ML model.
    """

    WINDOW_SIZE      = 20    # rolling window of recent trades per direction
    PAUSE_THRESHOLD  = 0.35  # if win rate drops below 35%, add penalty
    BOOST_THRESHOLD  = 0.60  # if win rate above 60%, reduce threshold slightly
    PENALTY_PP       = 4.0   # was 5.0 — gentler penalty: 4pp instead of 5pp per poor-WR period
    BOOST_PP         = 2.0   # pp removed from threshold when win rate is good
    ADJUSTMENT_CAP   = 6.0   # was 8.0 — max 6pp matches futures_trader RECALIB_CAP
    MIN_TRADES       = 6     # was 5 — require at least 6 trades before applying penalties

    def __init__(self, log_dir: Optional[Path] = None):
        # Rolling windows: {symbol: {direction: deque[TradeRecord]}}
        self._windows: Dict[str, Dict[str, Deque[TradeRecord]]] = {}
        # CSV log path
        self._log_path = (log_dir or Path("logs")) / "trade_journal.csv"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_header()

    # ── Public API ────────────────────────────────────────────────────────

    def record_trade(
        self,
        symbol:      str,
        side:        str,
        won:         bool,
        confidence:  float,
        pnl_usdt:    float   = 0.0,
        pnl_r:       float   = 0.0,
        bars_held:   int     = 0,
        exit_reason: str     = "UNKNOWN",
        regime:      str     = "UNKNOWN",
    ):
        """Call this every time a position closes."""
        record = TradeRecord(
            timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            symbol=symbol, side=side.lower(), won=won,
            confidence=confidence, pnl_usdt=pnl_usdt, pnl_r=pnl_r,
            bars_held=bars_held, exit_reason=exit_reason, regime=regime,
        )

        # Store in rolling window
        if symbol not in self._windows:
            self._windows[symbol] = {"long": deque(maxlen=self.WINDOW_SIZE),
                                     "short": deque(maxlen=self.WINDOW_SIZE)}
        key = side.lower()
        if key not in self._windows[symbol]:
            self._windows[symbol][key] = deque(maxlen=self.WINDOW_SIZE)
        self._windows[symbol][key].append(record)

        # Append to CSV
        self._append_csv(record)

        # Log summary
        adj = self.get_threshold_adjustment(symbol, side)
        wr  = self.get_win_rate(symbol, side)
        logger.info(
            f"[RECALIBRATOR] {symbol} {side.upper()} recorded | "
            f"won={won} pnl={pnl_usdt:+.2f} | "
            f"WR={wr:.0%} over last {self._count(symbol, side)} trades | "
            f"threshold_adj={adj:+.1f}pp"
        )

    def get_threshold_adjustment(self, symbol: str, side: str) -> float:
        """
        Return the pp adjustment to add to the base confidence threshold.
        Positive = harder to enter (penalty), negative = easier (boost).
        Range: [-ADJUSTMENT_CAP, +ADJUSTMENT_CAP]
        """
        n = self._count(symbol, side)
        if n < self.MIN_TRADES:
            return 0.0

        wr = self.get_win_rate(symbol, side)

        if wr < self.PAUSE_THRESHOLD:
            # Poor win rate → raise the bar
            adj = self.PENALTY_PP
        elif wr > self.BOOST_THRESHOLD:
            # Good win rate → lower the bar slightly
            adj = -self.BOOST_PP
        else:
            # Linear interpolation in the middle zone
            # 0 at wr=PAUSE_THRESHOLD, -BOOST_PP at wr=BOOST_THRESHOLD
            span = self.BOOST_THRESHOLD - self.PAUSE_THRESHOLD
            pos  = wr - self.PAUSE_THRESHOLD
            adj  = self.PENALTY_PP - (self.PENALTY_PP + self.BOOST_PP) * (pos / span)

        return max(-self.ADJUSTMENT_CAP, min(self.ADJUSTMENT_CAP, adj))

    def get_win_rate(self, symbol: str, side: str) -> float:
        """Return win rate for recent trades in this direction. 0.5 if unknown."""
        key = side.lower()
        if symbol not in self._windows or key not in self._windows[symbol]:
            return 0.5
        trades = list(self._windows[symbol][key])
        if not trades:
            return 0.5
        return sum(1 for t in trades if t.won) / len(trades)

    def get_summary(self, symbol: str) -> dict:
        """Return a summary dict for dashboard display."""
        result = {}
        for side in ("long", "short"):
            n  = self._count(symbol, side)
            wr = self.get_win_rate(symbol, side)
            adj = self.get_threshold_adjustment(symbol, side)
            result[side] = {"trades": n, "win_rate": wr, "threshold_adj": adj}
        return result

    def _count(self, symbol: str, side: str) -> int:
        key = side.lower()
        if symbol not in self._windows or key not in self._windows[symbol]:
            return 0
        return len(self._windows[symbol][key])

    # ── CSV logging ───────────────────────────────────────────────────────

    def _ensure_csv_header(self):
        if not self._log_path.exists():
            with open(self._log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "symbol", "side", "won",
                    "confidence", "pnl_usdt", "pnl_r",
                    "bars_held", "exit_reason", "regime",
                ])

    def _append_csv(self, record: TradeRecord):
        try:
            with open(self._log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    record.timestamp, record.symbol, record.side,
                    int(record.won), f"{record.confidence:.4f}",
                    f"{record.pnl_usdt:.4f}", f"{record.pnl_r:.4f}",
                    record.bars_held, record.exit_reason, record.regime,
                ])
        except Exception as e:
            logger.warning(f"CSV log error: {e}")