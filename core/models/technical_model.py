"""
core/models/technical_model.py
────────────────────────────────
Rule-based technical analysis model.
Scores multiple indicators and produces a consensus signal.
No training required — works immediately on live data.
"""

import pandas as pd
from typing import Tuple

from core.models.base_model import BaseModel, PredictionResult, Signal
from config.logger import get_logger

logger = get_logger(__name__)


class TechnicalModel(BaseModel):
    """
    Multi-indicator scoring model.
    
    Each indicator votes LONG (+1), SHORT (-1), or HOLD (0).
    Final signal = weighted average of all votes.
    """

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        """Analyze latest candle using technical rules."""
        if df.empty or len(df) < 2:
            return PredictionResult(
                signal=Signal.HOLD,
                confidence=0.0,
                long_probability=0.33,
                short_probability=0.33,
                source="technical",
                reasoning="Insufficient data",
            )

        row    = df.iloc[-1]   # Latest candle
        prev   = df.iloc[-2]   # Previous candle

        votes        = []
        reasons      = []

        # ── 1. RSI ──────────────────────────────────────────────────────────
        rsi = row.get("rsi", 50)
        if rsi < 30:
            votes.append(("rsi", 1.5, "RSI oversold → LONG"))
        elif rsi > 70:
            votes.append(("rsi", -1.5, "RSI overbought → SHORT"))
        elif rsi < 45:
            votes.append(("rsi", 0.5, "RSI bearish zone → mild LONG"))
        elif rsi > 55:
            votes.append(("rsi", -0.5, "RSI bullish zone → mild SHORT"))
        else:
            votes.append(("rsi", 0, "RSI neutral"))

        # ── 2. MACD ─────────────────────────────────────────────────────────
        macd      = row.get("macd", 0)
        macd_sig  = row.get("macd_signal", 0)
        macd_hist = row.get("macd_hist", 0)
        prev_hist = prev.get("macd_hist", 0)

        if macd > macd_sig and prev_hist < 0 <= macd_hist:
            votes.append(("macd", 2.0, "MACD bullish crossover → LONG"))
        elif macd < macd_sig and prev_hist > 0 >= macd_hist:
            votes.append(("macd", -2.0, "MACD bearish crossover → SHORT"))
        elif macd > macd_sig:
            votes.append(("macd", 0.8, "MACD above signal → LONG bias"))
        elif macd < macd_sig:
            votes.append(("macd", -0.8, "MACD below signal → SHORT bias"))
        else:
            votes.append(("macd", 0, "MACD neutral"))

        # ── 3. EMA Trend ────────────────────────────────────────────────────
        close  = row["close"]
        ema9   = row.get("ema_9", close)
        ema21  = row.get("ema_21", close)
        ema50  = row.get("ema_50", close)

        ema_score = 0
        ema_msg   = []
        if close > ema9:
            ema_score += 0.5
            ema_msg.append("price>EMA9")
        else:
            ema_score -= 0.5
            ema_msg.append("price<EMA9")

        if ema9 > ema21:
            ema_score += 0.5
            ema_msg.append("EMA9>EMA21")
        else:
            ema_score -= 0.5
            ema_msg.append("EMA9<EMA21")

        if close > ema50:
            ema_score += 0.5
            ema_msg.append("price>EMA50")
        else:
            ema_score -= 0.5
            ema_msg.append("price<EMA50")

        direction = "LONG" if ema_score > 0 else ("SHORT" if ema_score < 0 else "neutral")
        votes.append(("ema", ema_score, f"EMA trend → {direction} ({', '.join(ema_msg)})"))

        # ── 4. Bollinger Bands ──────────────────────────────────────────────
        bb_upper = row.get("bb_upper", 0)
        bb_lower = row.get("bb_lower", 0)
        bb_mid   = row.get("bb_mid", close)

        if close <= bb_lower:
            votes.append(("bb", 1.5, "Price at BB lower → LONG"))
        elif close >= bb_upper:
            votes.append(("bb", -1.5, "Price at BB upper → SHORT"))
        elif close < bb_mid:
            votes.append(("bb", 0.3, "Price below BB mid → mild LONG"))
        else:
            votes.append(("bb", -0.3, "Price above BB mid → mild SHORT"))

        # ── 5. Stochastic ───────────────────────────────────────────────────
        stoch_k = row.get("stoch_k", 50)
        stoch_d = row.get("stoch_d", 50)

        if stoch_k < 20 and stoch_k > stoch_d:
            votes.append(("stoch", 1.5, "Stoch oversold + K>D → LONG"))
        elif stoch_k > 80 and stoch_k < stoch_d:
            votes.append(("stoch", -1.5, "Stoch overbought + K<D → SHORT"))
        elif stoch_k < 20:
            votes.append(("stoch", 0.8, "Stoch oversold → LONG"))
        elif stoch_k > 80:
            votes.append(("stoch", -0.8, "Stoch overbought → SHORT"))
        else:
            votes.append(("stoch", 0, "Stoch neutral"))

        # ── 6. ADX (trend strength filter) ──────────────────────────────────
        adx     = row.get("adx", 0)
        adx_dmp = row.get("adx_dmp", 0)
        adx_dmn = row.get("adx_dmn", 0)

        if adx > 25:
            if adx_dmp > adx_dmn:
                votes.append(("adx", 1.0, f"ADX={adx:.1f} strong uptrend"))
            else:
                votes.append(("adx", -1.0, f"ADX={adx:.1f} strong downtrend"))
        else:
            # Weak trend — halve all other votes (ranging market)
            votes = [(k, v * 0.5, r) for k, v, r in votes]
            votes.append(("adx", 0, f"ADX={adx:.1f} weak trend — signals dampened"))

        # ── 7. Volume confirmation ───────────────────────────────────────────
        vol_ratio = row.get("volume_ratio", 1.0)
        if vol_ratio > 1.5:
            # High volume amplifies signal
            votes = [(k, v * 1.2, r) for k, v, r in votes]
            votes.append(("vol", 0, f"Volume spike {vol_ratio:.1f}x — signals amplified"))
        elif vol_ratio < 0.5:
            # Low volume dampens signal
            votes = [(k, v * 0.7, r) for k, v, r in votes]
            votes.append(("vol", 0, f"Low volume {vol_ratio:.1f}x — signals dampened"))

        # ── Aggregate votes ──────────────────────────────────────────────────
        total_score = sum(v for _, v, _ in votes)
        max_possible = sum(abs(v) for _, v, _ in votes) or 1
        normalized = total_score / max_possible  # -1.0 to +1.0

        long_prob  = max(0.0, min(1.0, 0.5 + normalized * 0.5))
        short_prob = 1.0 - long_prob
        confidence = abs(normalized)

        if normalized > 0.2:
            signal = Signal.LONG
        elif normalized < -0.2:
            signal = Signal.SHORT
        else:
            signal = Signal.HOLD

        reasoning = " | ".join(r for _, _, r in votes if r)

        logger.debug(
            f"Technical: score={total_score:.2f} norm={normalized:.2f} "
            f"→ {signal.value} (conf={confidence:.0%})"
        )

        return PredictionResult(
            signal=signal,
            confidence=confidence,
            long_probability=long_prob,
            short_probability=short_prob,
            source="technical",
            reasoning=reasoning,
        )

    def get_model_name(self) -> str:
        return "TechnicalIndicatorsModel"
