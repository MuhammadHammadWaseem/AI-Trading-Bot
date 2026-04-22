"""
core/market/regime_detector.py
══════════════════════════════════════════════════════════════════════════════
Market Regime Detection and Adaptive Strategy Parameters.

WHAT IT DOES
────────────
Analyses the current bar's indicators (ADX, ATR, EMA) to classify the market
into one of three regimes, then returns a set of adjusted strategy parameters
for that regime. FuturesTrader uses these parameters every cycle instead of
fixed class-level constants.

REGIMES
───────
  TRENDING       — Strong directional movement (ADX > ADX_TREND_THRESHOLD)
                   • Lower confidence threshold  → enter on slightly weaker signals
                   • Wider TP (larger ATR mult)  → let trends run
                   • Slightly wider SL           → avoid noise stopping you out early
                   • Normal position size

  RANGE          — Price oscillating (ADX low, ATR near mean)
                   • Higher confidence threshold → require stronger confirmation
                   • Tighter TP                  → take profit faster before reversal
                   • Normal SL
                   • Normal position size

  HIGH_VOLATILITY — Large price swings (ATR >> historical mean)
                   • Normal confidence threshold
                   • Wider SL to survive spikes
                   • Reduced position size (0.6×) to limit drawdown per trade

INTEGRATION
───────────
  Called once per cycle in FuturesTrader.run_cycle() after indicators are built.
  Returns a RegimeResult that overrides:
    - conf_thr_delta (delta added to BASE_THRESHOLD, replaces CONF_THRESHOLDS[symbol])
    - atr_sl_mult          (replaces ATR_SL_MULT[symbol])
    - atr_tp_mult          (replaces ATR_TP_MULT[symbol])
    - position_size_scale  (multiplies Kelly-sized quantity)
    - early_profit_r       (replaces EARLY_PROFIT_THRESHOLD_R)

DOES NOT TOUCH
──────────────
  - Portfolio risk cap (still enforced after sizing)
  - Cooldown logic
  - Agreement filter
  - Entry gate
  - MAX_HOLD_BARS timeout
  - TP/SL recovery on restart

DETECTION STABILITY
───────────────────
  A single noisy bar cannot flip the regime. The detector uses a short rolling
  confirmation window (CONFIRMATION_BARS): the regime is only confirmed when
  the same regime is detected for N consecutive bars. During the confirmation
  window the previous regime is held. This prevents thrashing between regimes
  on volatile single bars.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from config.logger import get_logger

logger = get_logger(__name__)


# ── Regime enum ───────────────────────────────────────────────────────────────

class Regime(str, Enum):
    TRENDING        = "TRENDING"
    RANGE           = "RANGE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RegimeParams:
    """
    All strategy parameters adjusted for the current market regime.
    FuturesTrader uses these in place of its class-level constants.
    """
    regime:               Regime
    conf_thr_delta:       float   # delta added to BASE_THRESHOLD (regime adjustment)
    sl_mult:              float   # SL = entry ± sl_mult × ATR
    tp_mult:              float   # TP = entry ± tp_mult × ATR
    size_scale:           float   # Kelly quantity × this (1.0 = no change)
    early_profit_r:       float   # close early when R >= this value

    # Diagnostic fields (shown in dashboard, not used for trading)
    adx:                  float = 0.0
    atr:                  float = 0.0
    atr_ratio:            float = 0.0   # current ATR / rolling mean ATR
    confirmation_pct:     float = 0.0   # how consistently same regime detected
    # In TRENDING regime: whether AGREE is required regardless of global setting
    require_agree:        bool  = False


# ── Per-regime parameter tables ───────────────────────────────────────────────
# All values are relative to the symbol's base ATR multipliers.
# base_sl and base_tp are passed in from ATR_SL_MULT / ATR_TP_MULT at call time
# so these are multiplied against the base, not hardcoded absolute values.

# ── Why TRENDING raises the threshold, not lowers it ─────────────────────────
# ADX > 25 means strong directional movement — but it does NOT tell you whether
# the trend is up or down. A SHORT signal in a strong UPTREND is the worst
# combination. Original design lowered threshold in TRENDING regime to "let
# trend signals through easier", but this allowed SPLIT (ML disagrees with
# technical) shorts through in uptrending assets (SOL, Mar 16). The fix:
# raise the threshold slightly and REQUIRE AGREE — in a trending market you
# need more conviction, not less, before trading counter-momentum.
#
# TP expansion (1.5×) is also removed from TRENDING. In a ranging crypto
# market that the ADX briefly calls "trending" (ADX=27-38), a 5.4×ATR TP
# target on SOL means price must move $3.40 from entry — almost never reached
# in 8 bars on a $93 asset. Kept at 1.2× (modest expansion only).
_REGIME_PARAMS = {
    Regime.TRENDING: {
        # Trending: RAISE threshold — need conviction before trading in a trend.
        # AGREE required (enforced in futures_trader.py, see TRENDING_REQUIRE_AGREE).
        "conf_adjustment":    +0.03,   # higher threshold by 3pp (e.g. 0.42 → 0.45)
        "sl_mult_factor":      1.25,   # wider SL: 1.25× base (trend noise buffer)
        "tp_mult_factor":      1.20,   # modest TP expansion: 1.2× base only
        "position_size_scale": 0.80,   # slightly smaller size — trending = riskier
        "early_profit_r":      0.80,   # FIX: was 0.60 — higher floor so exit covers fees (need ~0.75R+ to net)
    },
    Regime.RANGE: {
        # Range: require stronger confirmation, take profit quickly
        "conf_adjustment":    +0.02,   # lowered 5→2pp: in RANGE, model rarely exceeds 60% threshold; 2pp still filters noise
        "sl_mult_factor":      1.00,   # normal SL
        "tp_mult_factor":      1.00,   # normal TP — don't reach for large targets
        "position_size_scale": 1.00,   # normal size
        "early_profit_r":      1.10,   # raised 0.75→1.10: let trades run further before auto-exit; 1.10R ≈ 60% of way to TP
    },
    Regime.HIGH_VOLATILITY: {
        # High vol: widen SL to survive spikes, reduce size to limit loss
        "conf_adjustment":    +0.05,   # also raise threshold in volatile market
        "sl_mult_factor":      1.40,   # much wider SL: 1.4× base
        "tp_mult_factor":      1.10,   # only slightly wider TP
        "position_size_scale": 0.50,   # 50% of normal size (was 60%)
        "early_profit_r":      0.65,   # FIX: was 0.40 — raised; wide SL in HIGH_VOL means 0.40R gross is tiny
    },
}

# In TRENDING regime, REQUIRE_AGREEMENT is enforced unconditionally regardless
# of the trader's global REQUIRE_AGREEMENT setting. This prevents SPLIT signals
# from slipping through just because ADX is high.
# Set False only for research purposes.
TRENDING_REQUIRE_AGREE = True

# ── Detection thresholds ──────────────────────────────────────────────────────

ADX_TREND_THRESHOLD   = 25.0   # ADX above this → trending
ATR_VOLATILITY_MULT   = 1.5    # ATR > mean × this → high volatility
ATR_LOOKBACK_BARS     = 48     # bars to compute rolling ATR mean (48 × 15m = 12h)
CONFIRMATION_BARS     = 2      # consecutive bars of same regime to confirm flip


class RegimeDetector:
    """
    Detects market regime from OHLCV indicator data and returns adjusted
    strategy parameters.

    One instance per FuturesTrader symbol. Maintains a short history of
    recent detections to prevent single-bar regime flips.

    Usage:
        detector = RegimeDetector(symbol="BTCUSDT")

        # In run_cycle(), after add_all_indicators(df):
        result = detector.detect(df, base_sl_mult=1.5, base_tp_mult=3.0,
                                 base_conf_threshold=0.42)

        # Use result.conf_thr_delta instead of CONF_THRESHOLDS[symbol]
        # Use result.atr_sl_mult / result.atr_tp_mult for TP/SL computation
        # Multiply Kelly quantity by result.position_size_scale
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._history: deque[Regime] = deque(maxlen=CONFIRMATION_BARS + 2)
        self._confirmed_regime: Regime = Regime.RANGE   # default until data arrives
        self._raw_regime:       Regime = Regime.RANGE

    def detect(
        self,
        df:                   pd.DataFrame,
        base_sl_mult:         float = 2.0,
        base_tp_mult:         float = 3.0,
        base_conf_threshold:  float = 0.55,
    ) -> RegimeParams:
        """
        Detect current regime and return adjusted strategy parameters.

        Parameters
        ----------
        df                  : indicator-enriched OHLCV dataframe (from add_all_indicators)
        base_sl_mult        : symbol's default ATR SL multiplier (default 2.0)
        base_tp_mult        : symbol's default ATR TP multiplier (default 3.0)
        base_conf_threshold : symbol's default confidence threshold (default 0.55,
                              overridden by FuturesTrader with user's dashboard setting)
        """
        adx, atr, atr_ratio = self._extract_indicators(df)

        # ── Raw regime detection ───────────────────────────────────────────────
        if adx >= ADX_TREND_THRESHOLD:
            raw = Regime.TRENDING
        elif atr_ratio >= ATR_VOLATILITY_MULT:
            raw = Regime.HIGH_VOLATILITY
        else:
            raw = Regime.RANGE

        self._raw_regime = raw
        self._history.append(raw)

        # ── Confirmation: only flip confirmed regime after N consistent bars ──
        # This prevents a single volatile bar from switching all parameters.
        if len(self._history) >= CONFIRMATION_BARS:
            recent = list(self._history)[-CONFIRMATION_BARS:]
            if all(r == raw for r in recent):
                if raw != self._confirmed_regime:
                    logger.info(
                        f"[REGIME CHANGE] {self.symbol}: "
                        f"{self._confirmed_regime.value} → {raw.value} | "
                        f"ADX={adx:.1f}  ATR_ratio={atr_ratio:.2f}"
                    )
                self._confirmed_regime = raw

        # ── Build result from confirmed regime ─────────────────────────────────
        params = _REGIME_PARAMS[self._confirmed_regime]

        conf_threshold = max(
            0.35,   # never go below 35% regardless of regime
            min(0.75, base_conf_threshold + params["conf_adjustment"])
        )

        atr_sl_mult = base_sl_mult * params["sl_mult_factor"]
        atr_tp_mult = base_tp_mult * params["tp_mult_factor"]
        pos_scale   = params["position_size_scale"]
        early_r     = params["early_profit_r"]

        # Consistency count for diagnostic
        recent_all  = list(self._history)
        same_count  = sum(1 for r in recent_all if r == self._confirmed_regime)
        confirm_pct = same_count / max(len(recent_all), 1)

        # TRENDING regime always requires AGREE regardless of global setting
        req_agree = (self._confirmed_regime == Regime.TRENDING and TRENDING_REQUIRE_AGREE)

        result = RegimeParams(
            regime               = self._confirmed_regime,
            conf_thr_delta       = conf_threshold - base_conf_threshold,
            sl_mult              = atr_sl_mult,
            tp_mult              = atr_tp_mult,
            size_scale           = pos_scale,
            early_profit_r       = early_r,
            adx                  = adx,
            atr                  = atr,
            atr_ratio            = atr_ratio,
            confirmation_pct     = confirm_pct,
            require_agree        = req_agree,
        )

        logger.info(
            f"[REGIME] {self.symbol}: {self._confirmed_regime.value} "
            f"(raw={raw.value}) | "
            f"ADX={adx:.1f}  ATR={atr:.4f}  ATR_ratio={atr_ratio:.2f} | "
            f"conf_thr={conf_threshold:.0%}  sl_mult={atr_sl_mult:.2f}x  "
            f"tp_mult={atr_tp_mult:.2f}x  size_scale={pos_scale:.0%}  "
            f"early_r={early_r:.2f}R"
        )

        return result

    # ── Indicator extraction ──────────────────────────────────────────────────

    def _extract_indicators(
        self, df: pd.DataFrame
    ) -> tuple[float, float, float]:
        """
        Extract ADX, ATR, and ATR ratio from the dataframe.
        Returns (adx, current_atr, atr_ratio).

        Falls back gracefully if columns are missing.
        """
        # ADX
        adx = 0.0
        for col in ["adx", "ADX", "adx_14"]:
            if col in df.columns:
                val = float(df[col].iloc[-1])
                if not np.isnan(val):
                    adx = val
                    break

        # ATR
        atr = 0.0
        for col in ["atr", "ATR", "atr_14"]:
            if col in df.columns:
                val = float(df[col].iloc[-1])
                if not np.isnan(val):
                    atr = val
                    break

        # ATR ratio: current ATR vs rolling mean over last N bars
        atr_ratio = 1.0
        if atr > 0:
            for col in ["atr", "ATR", "atr_14"]:
                if col in df.columns:
                    series = df[col].dropna()
                    if len(series) >= ATR_LOOKBACK_BARS:
                        mean_atr = float(series.iloc[-ATR_LOOKBACK_BARS:].mean())
                    elif len(series) > 5:
                        mean_atr = float(series.mean())
                    else:
                        mean_atr = atr   # not enough history — assume normal vol
                    if mean_atr > 0:
                        atr_ratio = atr / mean_atr
                    break

        return adx, atr, atr_ratio

    @property
    def current_regime(self) -> Regime:
        return self._confirmed_regime