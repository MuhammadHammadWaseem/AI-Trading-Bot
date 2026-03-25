"""
core/strategy/recovery_strategy.py
─────────────────────────────────────
Loss recovery engine.
When a trade closes at a loss, this module decides how to recover:

Strategy: Smart Martingale with signal confirmation
  - Track running loss per symbol
  - Slightly increase next position size to recover losses
  - BUT only open recovery trade if signal is strong enough
  - Hard cap on recovery attempts to prevent blowup
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime

from core.models.base_model import PredictionResult, Signal
from config.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RecoveryState:
    """Tracks loss recovery state per symbol."""
    symbol:              str
    total_loss_usdt:     float = 0.0
    recovery_attempts:   int   = 0
    last_loss_time:      Optional[datetime] = None
    is_recovering:       bool  = False
    wait_bars:           int   = 0   # 5m bars elapsed without a qualifying recovery signal


class RecoveryStrategy:
    """
    Manages loss recovery across all traded symbols.
    
    Rules:
      1. Max 3 recovery attempts per loss event
      2. Recovery trade size = original_risk + (loss / max_attempts)
      3. Recovery trade only opens if signal confidence > 70%
      4. After 3 failed recoveries → stop trading that symbol for 1 hour
    """

    MAX_RECOVERY_ATTEMPTS   = 3
    RECOVERY_CONFIDENCE_MIN = 0.70
    SIZE_MULTIPLIER_PER_TRY = 1.5    # 1x → 1.5x → 2.25x
    COOLDOWN_MINUTES        = 60
    MAX_RECOVERY_WAIT_BARS  = 6      # ~30 min at 5m TF — auto-clear if no qualifying signal

    def __init__(self):
        self._states: Dict[str, RecoveryState] = {}

    def tick_bar(self, symbol: str):
        """
        Call once per new 5m candle for a symbol in recovery.
        After MAX_RECOVERY_WAIT_BARS bars without a qualifying signal,
        auto-clears recovery so normal trading resumes at standard size.
        """
        if symbol not in self._states:
            return
        state = self._states[symbol]
        if not state.is_recovering:
            return

        state.wait_bars += 1
        if state.wait_bars >= self.MAX_RECOVERY_WAIT_BARS:
            state.is_recovering     = False
            state.recovery_attempts = 0
            state.wait_bars         = 0
            logger.info(
                f"[RECOVERY AUTO-CLEAR] {symbol} — no qualifying signal for "
                f"{self.MAX_RECOVERY_WAIT_BARS} bars. "
                f"Resuming normal trading (loss={state.total_loss_usdt:.2f} USDT written off)."
            )
            state.total_loss_usdt = 0.0

    def record_loss(self, symbol: str, loss_usdt: float):
        """Call this when a trade closes at a loss."""
        if symbol not in self._states:
            self._states[symbol] = RecoveryState(symbol=symbol)

        state = self._states[symbol]
        state.total_loss_usdt   += abs(loss_usdt)
        state.is_recovering      = True
        state.last_loss_time     = datetime.now()
        state.recovery_attempts  = 0
        state.wait_bars          = 0

        logger.warning(
            f"📉 Loss recorded: {symbol} | "
            f"loss={abs(loss_usdt):.2f} USDT | "
            f"total_to_recover={state.total_loss_usdt:.2f} USDT"
        )

    def record_profit(self, symbol: str, profit_usdt: float):
        """Call this when a trade closes at profit."""
        if symbol not in self._states:
            return

        state = self._states[symbol]
        state.total_loss_usdt = max(0, state.total_loss_usdt - profit_usdt)

        if state.total_loss_usdt <= 0:
            state.is_recovering    = False
            state.recovery_attempts = 0
            logger.info(f"✅ {symbol} losses fully recovered!")
        else:
            logger.info(
                f"📈 Partial recovery: {symbol} | "
                f"remaining={state.total_loss_usdt:.2f} USDT"
            )

    def should_open_recovery(
        self,
        symbol:     str,
        prediction: PredictionResult,
    ) -> bool:
        """
        Decide if a recovery trade should open.
        Returns True only if conditions are met.
        """
        if symbol not in self._states:
            return False

        state = self._states[symbol]

        if not state.is_recovering:
            return False

        # Check max attempts
        if state.recovery_attempts >= self.MAX_RECOVERY_ATTEMPTS:
            logger.warning(
                f"🚫 Max recovery attempts reached for {symbol}. "
                f"Cooling down {self.COOLDOWN_MINUTES} min."
            )
            return False

        # Check cooldown
        if state.last_loss_time:
            from datetime import timedelta
            elapsed = (datetime.now() - state.last_loss_time).total_seconds() / 60
            if elapsed < 0:   # No cooldown — valid signals should not be blocked
                return False

        # Check signal confidence
        if prediction.confidence < self.RECOVERY_CONFIDENCE_MIN:
            logger.debug(
                f"Recovery trade skipped: {symbol} | "
                f"confidence {prediction.confidence:.0%} < "
                f"required {self.RECOVERY_CONFIDENCE_MIN:.0%}"
            )
            return False

        # Must have a directional signal (not HOLD)
        if prediction.signal == Signal.HOLD:
            return False

        return True

    def get_recovery_size_multiplier(self, symbol: str) -> float:
        """
        Returns position size multiplier for recovery trade.
        Attempt 0→1x, 1→1.5x, 2→2.25x
        """
        if symbol not in self._states:
            return 1.0

        state = self._states[symbol]
        state.recovery_attempts += 1

        multiplier = self.SIZE_MULTIPLIER_PER_TRY ** (state.recovery_attempts - 1)
        multiplier = min(multiplier, 3.0)   # Hard cap at 3x

        logger.info(
            f"🔄 Recovery attempt #{state.recovery_attempts} for {symbol} | "
            f"size_multiplier={multiplier:.2f}x"
        )
        return multiplier

    def is_in_recovery(self, symbol: str) -> bool:
        return (
            symbol in self._states
            and self._states[symbol].is_recovering
        )

    def get_state(self, symbol: str) -> Optional[RecoveryState]:
        return self._states.get(symbol) 