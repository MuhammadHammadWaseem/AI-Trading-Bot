"""
core/strategies/range_strategy.py
──────────────────────────────────
Rule-based fallback strategy for ranging / low-volatility markets.

Activates when the ML model has no statistical edge (confidence below
threshold). Uses classic mean-reversion logic: enter at Bollinger Band
extremes confirmed by RSI and Stochastic, targeting the opposite band.

Architecture
────────────
The RangeStrategy is a SECOND OPINION, not a replacement for the ML model.
Execution flow in futures_trader.py:

  1. ML model evaluates → if conf ≥ threshold  → ML trade (existing path)
  2. ML model evaluates → if conf < threshold  → ask RangeStrategy.evaluate()
     a. Range signal found + quality filter passes → RANGE trade
     b. No range signal → HOLD (do nothing)

Range setup criteria (ALL must be met)
───────────────────────────────────────
LONG at lower band:
  • price ≤ bb_lower × (1 + BAND_TOUCH_BUFFER)     — near or below lower band
  • rsi < RSI_OVERSOLD                               — oversold
  • stoch_k < STOCH_OVERSOLD                        — stochastic oversold
  • previous candle had price > current (falling into band, not already bouncing)
  • ATR_ratio < ATR_RATIO_MAX_FOR_RANGE              — not a breakout, stay in range

SHORT at upper band:
  • price ≥ bb_upper × (1 - BAND_TOUCH_BUFFER)     — near or above upper band
  • rsi > RSI_OVERBOUGHT                             — overbought
  • stoch_k > STOCH_OVERBOUGHT                      — stochastic overbought
  • previous candle had price < current (rising into band)
  • ATR_ratio < ATR_RATIO_MAX_FOR_RANGE

Position parameters
───────────────────
  SL  = SL_ATR_MULT × ATR beyond the band (stop outside the range)
  TP  = TP_ATR_MULT × ATR toward the opposite band
  These are passed back to FuturesTrader which calls risk_manager.calculate_trade()
  with the regime-appropriate multipliers.

Rate limiting
─────────────
  MAX_RANGE_TRADES_PER_SYMBOL_PER_DAY caps overtrading in flat markets.
  MIN_BARS_BETWEEN_RANGE_TRADES enforces a cooldown between entries.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config.logger import get_logger

logger = get_logger(__name__)


# ── Tunable constants ─────────────────────────────────────────────────────────

# Band touch: how close to the band counts as "at the band"
# 0.002 = within 0.2% of the band price
BAND_TOUCH_BUFFER = 0.003

# RSI thresholds for confirming oversold/overbought
RSI_OVERSOLD   = 38   # slightly relaxed from classic 30 for more signals
RSI_OVERBOUGHT = 62   # slightly relaxed from classic 70

# Stochastic thresholds
STOCH_OVERSOLD   = 30
STOCH_OVERBOUGHT = 70

# ATR_ratio ceiling: if market is too volatile, it may be breaking out not reversing
# At ATR_ratio > 1.10, moves are above average — could be a real breakout
ATR_RATIO_MAX_FOR_RANGE = 1.10

# SL just outside the band: 1.5× ATR beyond entry
# TP targeting opposite band: 2.5× ATR (full BB width is ~2.7×ATR)
RANGE_SL_ATR_MULT = 1.5
RANGE_TP_ATR_MULT = 2.5

# Rate limits
MAX_RANGE_TRADES_PER_SYMBOL_PER_DAY = 4   # prevent overtrading in flat market
MIN_BARS_BETWEEN_RANGE_TRADES       = 6   # 30 minutes between range entries

# Minimum quality score to fire a trade (0.0 – 1.0)
# Higher = fewer trades, higher quality
MIN_QUALITY_SCORE = 0.55


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RangeSignal:
    """Output of RangeStrategy.evaluate()."""
    has_signal:   bool
    direction:    str          # "LONG" | "SHORT" | "NONE"
    quality:      float        # 0.0–1.0  (higher = cleaner setup)
    sl_atr_mult:  float        # pass to risk_manager
    tp_atr_mult:  float        # pass to risk_manager
    reason:       str          # human-readable log string
    rsi:          float = 0.0
    stoch_k:      float = 0.0
    atr_ratio:    float = 0.0
    bb_position:  float = 0.0  # 0=at lower band, 1=at upper band

    @property
    def is_long(self) -> bool:
        return self.direction == "LONG"

    @property
    def is_short(self) -> bool:
        return self.direction == "SHORT"


@dataclass
class _RangeState:
    """Per-symbol runtime state."""
    trades_today:     int   = 0
    last_trade_bar:   int   = -999    # bar index of last range trade
    last_reset_date:  str   = ""      # YYYY-MM-DD for daily counter reset
    consecutive_wins: int   = 0
    consecutive_losses: int = 0


# ── Main strategy class ───────────────────────────────────────────────────────

class RangeStrategy:
    """
    Rule-based mean-reversion strategy for ranging markets.

    One instance per symbol, held inside FuturesTrader.
    """

    def __init__(self, symbol: str):
        self.symbol  = symbol
        self._state  = _RangeState()
        self._bar_counter = 0   # incremented each time evaluate() is called

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        df:        pd.DataFrame,
        atr_ratio: float,
        regime:    str,
    ) -> RangeSignal:
        """
        Evaluate whether a range trade setup exists on the current candle.

        Parameters
        ----------
        df        : indicator-enriched OHLCV dataframe (needs bb_*, rsi, stoch_k/d, atr)
        atr_ratio : current ATR / 20-bar ATR mean  (pre-computed by FuturesTrader)
        regime    : confirmed regime string ("TRENDING" | "RANGE" | "HIGH_VOLATILITY")

        Returns
        -------
        RangeSignal — always, caller checks .has_signal
        """
        self._bar_counter += 1
        self._reset_daily_counter()

        no_signal = RangeSignal(
            has_signal=False, direction="NONE", quality=0.0,
            sl_atr_mult=RANGE_SL_ATR_MULT, tp_atr_mult=RANGE_TP_ATR_MULT,
            reason="no range setup", rsi=0, stoch_k=0,
            atr_ratio=atr_ratio, bb_position=0.5,
        )

        # ── Pre-checks ────────────────────────────────────────────────────────

        if len(df) < 20:
            return no_signal

        # Rate limiting
        if self._state.trades_today >= MAX_RANGE_TRADES_PER_SYMBOL_PER_DAY:
            return RangeSignal(
                has_signal=False, direction="NONE", quality=0.0,
                sl_atr_mult=RANGE_SL_ATR_MULT, tp_atr_mult=RANGE_TP_ATR_MULT,
                reason=f"daily limit reached ({self._state.trades_today}/{MAX_RANGE_TRADES_PER_SYMBOL_PER_DAY})",
                atr_ratio=atr_ratio, bb_position=0.5,
            )

        bars_since_last = self._bar_counter - self._state.last_trade_bar
        if bars_since_last < MIN_BARS_BETWEEN_RANGE_TRADES:
            return RangeSignal(
                has_signal=False, direction="NONE", quality=0.0,
                sl_atr_mult=RANGE_SL_ATR_MULT, tp_atr_mult=RANGE_TP_ATR_MULT,
                reason=f"cooldown ({bars_since_last}/{MIN_BARS_BETWEEN_RANGE_TRADES} bars)",
                atr_ratio=atr_ratio, bb_position=0.5,
            )

        # Do not range-trade during breakout conditions
        if atr_ratio > ATR_RATIO_MAX_FOR_RANGE:
            return RangeSignal(
                has_signal=False, direction="NONE", quality=0.0,
                sl_atr_mult=RANGE_SL_ATR_MULT, tp_atr_mult=RANGE_TP_ATR_MULT,
                reason=f"ATR_ratio={atr_ratio:.2f} too high — possible breakout",
                atr_ratio=atr_ratio, bb_position=0.5,
            )

        # ── Extract indicators ────────────────────────────────────────────────

        row  = df.iloc[-1]
        prev = df.iloc[-2]

        close    = float(row["close"])
        prev_close = float(prev["close"])

        bb_upper = float(row.get("bb_upper", 0) or 0)
        bb_lower = float(row.get("bb_lower", 0) or 0)
        bb_mid   = float(row.get("bb_mid", close) or close)

        rsi      = float(row.get("rsi", 50) or 50)
        stoch_k  = float(row.get("stoch_k", 50) or 50)
        stoch_d  = float(row.get("stoch_d", 50) or 50)
        atr_cur  = float(row.get("atr", 0) or 0)

        if bb_upper <= 0 or bb_lower <= 0 or atr_cur <= 0:
            return no_signal

        # BB position: 0 = at lower, 1 = at upper
        bb_range    = bb_upper - bb_lower
        bb_position = (close - bb_lower) / bb_range if bb_range > 0 else 0.5

        # ── LONG setup: price at lower band ───────────────────────────────────

        long_signal = self._evaluate_long(
            close, prev_close, bb_lower, bb_mid, bb_upper,
            rsi, stoch_k, stoch_d, atr_cur, atr_ratio, bb_position,
        )

        if long_signal is not None and long_signal.quality >= MIN_QUALITY_SCORE:
            logger.info(
                f"[RANGE] {self.symbol} LONG setup | "
                f"BB_pos={bb_position:.2f} RSI={rsi:.1f} Stoch={stoch_k:.1f} | "
                f"quality={long_signal.quality:.2f} | {long_signal.reason}"
            )
            return long_signal

        # ── SHORT setup: price at upper band ──────────────────────────────────

        short_signal = self._evaluate_short(
            close, prev_close, bb_lower, bb_mid, bb_upper,
            rsi, stoch_k, stoch_d, atr_cur, atr_ratio, bb_position,
        )

        if short_signal is not None and short_signal.quality >= MIN_QUALITY_SCORE:
            logger.info(
                f"[RANGE] {self.symbol} SHORT setup | "
                f"BB_pos={bb_position:.2f} RSI={rsi:.1f} Stoch={stoch_k:.1f} | "
                f"quality={short_signal.quality:.2f} | {short_signal.reason}"
            )
            return short_signal

        return no_signal

    def record_trade(self) -> None:
        """Call after a range trade is executed."""
        self._state.trades_today  += 1
        self._state.last_trade_bar = self._bar_counter
        logger.info(
            f"[RANGE] {self.symbol} trade recorded "
            f"({self._state.trades_today}/{MAX_RANGE_TRADES_PER_SYMBOL_PER_DAY} today)"
        )

    def record_outcome(self, won: bool) -> None:
        """Call when a range trade closes (win or loss)."""
        if won:
            self._state.consecutive_wins   += 1
            self._state.consecutive_losses  = 0
        else:
            self._state.consecutive_losses += 1
            self._state.consecutive_wins    = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _evaluate_long(
        self,
        close:        float,
        prev_close:   float,
        bb_lower:     float,
        bb_mid:       float,
        bb_upper:     float,
        rsi:          float,
        stoch_k:      float,
        stoch_d:      float,
        atr_cur:      float,
        atr_ratio:    float,
        bb_position:  float,
    ) -> Optional[RangeSignal]:
        """Evaluate LONG (oversold at lower band) setup."""

        # Price must be at or near lower band
        lower_touch_threshold = bb_lower * (1 + BAND_TOUCH_BUFFER)
        if close > lower_touch_threshold:
            return None

        # RSI must be oversold
        if rsi >= RSI_OVERSOLD:
            return None

        # Stochastic must be oversold
        if stoch_k >= STOCH_OVERSOLD:
            return None

        # Price should be falling into the band (not already bouncing strongly)
        # Allow if close ≤ prev_close (still declining or flat at band)
        # — this ensures we're catching the touch, not chasing a bounce
        # Relaxed: also allow if close is within 0.5×ATR of prev (tiny bounce ok)
        if close > prev_close + 0.5 * atr_cur:
            return None   # already bounced hard — entry is chasing

        # ── Quality scoring ───────────────────────────────────────────────────
        quality = 0.5   # base score for meeting all criteria

        # Bonus: RSI deep oversold
        if rsi < 25:
            quality += 0.20
        elif rsi < 30:
            quality += 0.12

        # Bonus: Stochastic crossover (K crossing above D from below)
        if stoch_k > stoch_d and stoch_k < 25:
            quality += 0.15
        elif stoch_k < 20:
            quality += 0.08

        # Bonus: price BELOW lower band (full penetration)
        if close < bb_lower:
            quality += 0.15

        # Bonus: ATR_ratio shows quiet market (cleaner range)
        if atr_ratio < 0.80:
            quality += 0.10
        elif atr_ratio < 0.90:
            quality += 0.05

        # Penalty: consecutive losses → reduce quality to slow down trading
        if self._state.consecutive_losses >= 2:
            quality -= 0.15 * self._state.consecutive_losses

        quality = max(0.0, min(1.0, quality))

        return RangeSignal(
            has_signal=True,
            direction="LONG",
            quality=quality,
            sl_atr_mult=RANGE_SL_ATR_MULT,
            tp_atr_mult=RANGE_TP_ATR_MULT,
            reason=(
                f"BB_lower touch (pos={bb_position:.2f}) | "
                f"RSI={rsi:.1f} | Stoch={stoch_k:.1f}"
            ),
            rsi=rsi,
            stoch_k=stoch_k,
            atr_ratio=atr_ratio,
            bb_position=bb_position,
        )

    def _evaluate_short(
        self,
        close:        float,
        prev_close:   float,
        bb_lower:     float,
        bb_mid:       float,
        bb_upper:     float,
        rsi:          float,
        stoch_k:      float,
        stoch_d:      float,
        atr_cur:      float,
        atr_ratio:    float,
        bb_position:  float,
    ) -> Optional[RangeSignal]:
        """Evaluate SHORT (overbought at upper band) setup."""

        # Price must be at or near upper band
        upper_touch_threshold = bb_upper * (1 - BAND_TOUCH_BUFFER)
        if close < upper_touch_threshold:
            return None

        # RSI must be overbought
        if rsi <= RSI_OVERBOUGHT:
            return None

        # Stochastic must be overbought
        if stoch_k <= STOCH_OVERBOUGHT:
            return None

        # Price should be rising into the band (not already reversing hard)
        if close < prev_close - 0.5 * atr_cur:
            return None

        # ── Quality scoring ───────────────────────────────────────────────────
        quality = 0.5

        # Bonus: RSI deep overbought
        if rsi > 75:
            quality += 0.20
        elif rsi > 70:
            quality += 0.12

        # Bonus: Stochastic crossover (K crossing below D from above)
        if stoch_k < stoch_d and stoch_k > 75:
            quality += 0.15
        elif stoch_k > 80:
            quality += 0.08

        # Bonus: price ABOVE upper band (full penetration)
        if close > bb_upper:
            quality += 0.15

        # Bonus: ATR_ratio shows quiet market
        if atr_ratio < 0.80:
            quality += 0.10
        elif atr_ratio < 0.90:
            quality += 0.05

        # Penalty: consecutive losses
        if self._state.consecutive_losses >= 2:
            quality -= 0.15 * self._state.consecutive_losses

        quality = max(0.0, min(1.0, quality))

        return RangeSignal(
            has_signal=True,
            direction="SHORT",
            quality=quality,
            sl_atr_mult=RANGE_SL_ATR_MULT,
            tp_atr_mult=RANGE_TP_ATR_MULT,
            reason=(
                f"BB_upper touch (pos={bb_position:.2f}) | "
                f"RSI={rsi:.1f} | Stoch={stoch_k:.1f}"
            ),
            rsi=rsi,
            stoch_k=stoch_k,
            atr_ratio=atr_ratio,
            bb_position=bb_position,
        )

    def _reset_daily_counter(self) -> None:
        """Reset daily trade count at UTC midnight."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._state.last_reset_date != today:
            self._state.trades_today    = 0
            self._state.last_reset_date = today