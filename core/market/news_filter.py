"""
core/market/news_filter.py
──────────────────────────
Lightweight news-awareness layer.

Sources:
  1. Fear & Greed Index  — alternative.me/fng (free, no key, daily)
  2. CryptoCompare news  — data.messari.io/api/v1/news  (free, no key, recent headlines)

Returns a SentimentState that futures_trader uses as a gate:
  NEUTRAL  → no threshold adjustment
  CAUTION  → raise effective confidence threshold by +CAUTION_PENALTY pp
  AVOID    → block all new entries for AVOID_BARS bars

Integration point in futures_trader.py:
    from core.market.news_filter import NewsFilter, SentimentState
    self._news = NewsFilter()          # one per symbol-set (shared)

    # In evaluate_signal(), just before opening a trade:
    state = await self._news.get_state()
    if state == SentimentState.AVOID:
        skip_reason = f"news-avoid ({self._news.last_reason})"
    elif state == SentimentState.CAUTION:
        effective_threshold += NewsFilter.CAUTION_PENALTY
"""

import asyncio
import time
import logging
from enum import Enum
from typing import Optional
import aiohttp

logger = logging.getLogger(__name__)


class SentimentState(str, Enum):
    NEUTRAL = "NEUTRAL"
    CAUTION = "CAUTION"
    AVOID   = "AVOID"


class NewsFilter:
    """
    Singleton-ish: create once and share across all FuturesTrader instances.
    Thread/async safe — uses asyncio.Lock internally.
    """

    # ── Tuning knobs ──────────────────────────────────────────────────────
    CAUTION_PENALTY     = 0.03   # +3pp to confidence threshold in CAUTION
    REFRESH_INTERVAL    = 300    # seconds — re-fetch every 5 minutes
    REQUEST_TIMEOUT     = 8      # seconds per HTTP call

    # Fear & Greed thresholds (index 0-100)
    FNG_EXTREME_FEAR    = 20     # ≤ 20 → CAUTION
    FNG_EXTREME_GREED   = 80     # ≥ 80 → CAUTION (euphoria = risky)

    # Keywords that trigger AVOID (case-insensitive substring match in headline)
    AVOID_KEYWORDS = [
        "hack", "exploit", "stolen", "heist",           # security
        "ban", "illegal", "prohibit", "shutdown",        # regulatory hard
        "crash", "collapse", "insolvent", "bankrupt",    # systemic
        "sec charges", "doj", "arrest",                  # enforcement
        "flash crash", "circuit breaker",                # market disruption
    ]

    # Keywords that trigger CAUTION
    CAUTION_KEYWORDS = [
        "sec", "cftc", "regulation", "regulatory",
        "investigation", "lawsuit", "fine",
        "inflation", "fed rate", "federal reserve",
        "liquidation", "delisting",
    ]

    def __init__(self) -> None:
        self._state:        SentimentState = SentimentState.NEUTRAL
        self._last_fetch:   float          = 0.0
        self._lock:         asyncio.Lock   = asyncio.Lock()
        self.last_reason:   str            = ""
        self._fng_value:    Optional[int]  = None

    # ── Public API ────────────────────────────────────────────────────────

    async def get_state(self) -> SentimentState:
        """Return current sentiment state, refreshing cache if stale."""
        now = time.monotonic()
        if now - self._last_fetch >= self.REFRESH_INTERVAL:
            async with self._lock:
                # Double-check after acquiring lock
                if time.monotonic() - self._last_fetch >= self.REFRESH_INTERVAL:
                    await self._refresh()
        return self._state

    @property
    def fng_value(self) -> Optional[int]:
        """Last fetched Fear & Greed index value (0-100), or None if unavailable."""
        return self._fng_value

    # ── Internal ─────────────────────────────────────────────────────────

    async def _refresh(self) -> None:
        """Fetch both sources and compute new state. Never raises — fails silently."""
        try:
            fng_state, fng_reason, fng_value = await self._fetch_fear_greed()
            news_state, news_reason          = await self._fetch_news_headlines()

            # AVOID beats CAUTION beats NEUTRAL
            if fng_state == SentimentState.AVOID or news_state == SentimentState.AVOID:
                self._state       = SentimentState.AVOID
                self.last_reason  = news_reason if news_state == SentimentState.AVOID else fng_reason
            elif fng_state == SentimentState.CAUTION or news_state == SentimentState.CAUTION:
                self._state       = SentimentState.CAUTION
                self.last_reason  = " | ".join(filter(None, [fng_reason, news_reason]))
            else:
                self._state       = SentimentState.NEUTRAL
                self.last_reason  = f"FnG={fng_value}" if fng_value else "OK"

            self._fng_value   = fng_value
            self._last_fetch  = time.monotonic()

            logger.info(
                f"[NEWS] State={self._state.value} | FnG={fng_value} | {self.last_reason}"
            )

        except Exception as exc:
            # Never block trading due to a network failure
            logger.warning(f"[NEWS] Refresh failed ({exc}) — keeping {self._state.value}")
            self._last_fetch = time.monotonic()  # avoid hammering on repeated failures

    async def _fetch_fear_greed(self):
        """
        Fetch Fear & Greed Index from alternative.me.
        Returns (SentimentState, reason_str, fng_value_int).
        """
        url = "https://api.alternative.me/fng/?limit=1&format=json"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)) as resp:
                    if resp.status != 200:
                        return SentimentState.NEUTRAL, "", None
                    data  = await resp.json(content_type=None)
                    value = int(data["data"][0]["value"])
                    label = data["data"][0]["value_classification"]

                    if value <= self.FNG_EXTREME_FEAR:
                        return SentimentState.CAUTION, f"Extreme Fear (FnG={value})", value
                    if value >= self.FNG_EXTREME_GREED:
                        return SentimentState.CAUTION, f"Extreme Greed (FnG={value})", value
                    return SentimentState.NEUTRAL, "", value

        except Exception:
            return SentimentState.NEUTRAL, "", None

    async def _fetch_news_headlines(self):
        """
        Fetch recent crypto news from Messari.
        Returns (SentimentState, reason_str).
        """
        url = "https://data.messari.io/api/v1/news?limit=20"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)) as resp:
                    if resp.status != 200:
                        return SentimentState.NEUTRAL, ""
                    data     = await resp.json(content_type=None)
                    articles = data.get("data", [])

            # Scan last 20 headlines
            combined_text = " ".join(
                (a.get("title", "") + " " + a.get("content", "")[:200]).lower()
                for a in articles
            )

            for kw in self.AVOID_KEYWORDS:
                if kw in combined_text:
                    return SentimentState.AVOID, f"keyword: '{kw}'"

            for kw in self.CAUTION_KEYWORDS:
                if kw in combined_text:
                    return SentimentState.CAUTION, f"keyword: '{kw}'"

            return SentimentState.NEUTRAL, ""

        except Exception:
            return SentimentState.NEUTRAL, ""