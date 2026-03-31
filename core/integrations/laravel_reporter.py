"""
core/integrations/laravel_reporter.py
──────────────────────────────────────
Non-blocking HTTP reporter from Python bot → Laravel dashboard.

FIXED ISSUES:
  1. _post_safe now logs ALL errors at WARNING so we can diagnose failures
  2. startup_test() verifies the connection before the bot runs
  3. Log handler also captures WARNING level (was missing)
  4. Trade reporting correctly awaits and logs failures
  5. Heartbeat checks stop command correctly
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

LOG_BATCH_SIZE     = 15
LOG_FLUSH_INTERVAL = 3    # flush every 3s
HEARTBEAT_INTERVAL = 30


@dataclass
class _PendingLog:
    level:     str
    channel:   str
    message:   str
    context:   Optional[dict]
    logged_at: str


class LaravelReporter:
    def __init__(self, api_url: str, bot_token: str, bot_id: int):
        self.api_url   = api_url.rstrip("/")
        self.bot_token = bot_token
        self.bot_id    = bot_id

        self._session:        Optional[aiohttp.ClientSession] = None
        self._log_buffer:     list[_PendingLog] = []
        self._last_flush:     float = time.monotonic()
        self._stop_requested:     bool  = False
        self._close_trades_on_stop: bool  = False  # set from heartbeat stop command
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._flush_task:     Optional[asyncio.Task] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.bot_token}",
                    "Content-Type":  "application/json",
                    "Accept":        "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=10),
            )
        return self._session

    async def start(self):
        """Start background tasks. Call immediately after creating reporter."""
        await self._get_session()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._flush_task     = asyncio.create_task(self._flush_loop())

    async def startup_test(self) -> bool:
        """
        Test connectivity to Laravel API. Call before starting the bot loop.
        Returns True if connected, False otherwise.
        Logs a clear diagnostic message so failures are visible in bot_N.log.
        """
        logger.info(f"[Reporter] Testing connection to: {self.api_url}/heartbeat")
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.api_url}/heartbeat",
                json={},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    logger.info(f"[Reporter] ✅ Laravel API connected (HTTP 200)")
                    return True
                elif resp.status == 401:
                    text = await resp.text()
                    logger.error(f"[Reporter] ❌ Auth failed (HTTP 401) — bot_token invalid. "
                                 f"Response: {text[:200]}")
                    return False
                else:
                    text = await resp.text()
                    logger.error(f"[Reporter] ❌ Unexpected HTTP {resp.status} from Laravel. "
                                 f"URL: {self.api_url}/heartbeat — Response: {text[:200]}")
                    return False
        except aiohttp.ClientConnectorError as e:
            logger.error(f"[Reporter] ❌ Cannot connect to Laravel at {self.api_url}. "
                         f"Error: {e}. Check APP_URL in .env and that Apache/XAMPP is running.")
            return False
        except asyncio.TimeoutError:
            logger.error(f"[Reporter] ❌ Connection timed out to {self.api_url}. "
                         f"Apache may be down or URL is wrong.")
            return False
        except Exception as e:
            logger.error(f"[Reporter] ❌ Unexpected connection error: {e}")
            return False

    async def close(self):
        for task in [self._heartbeat_task, self._flush_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self._session and not self._session.closed:
            await self._session.close()

    async def flush(self):
        """Force-flush the pending log buffer."""
        if not self._log_buffer:
            return
        batch, self._log_buffer = self._log_buffer[:], []
        await self._post_safe("log/batch", {
            "entries": [
                {
                    "level":     e.level,
                    "channel":   e.channel,
                    "message":   e.message[:2000],
                    "context":   e.context,
                    "logged_at": e.logged_at,
                }
                for e in batch
            ]
        })
        self._last_flush = time.monotonic()

    # ── Public API ─────────────────────────────────────────────────────────

    def queue_log(self, level: str, message: str, channel: str = "bot",
                  context: Optional[dict] = None):
        self._log_buffer.append(_PendingLog(
            level=level, channel=channel, message=message,
            context=context,
            logged_at=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        ))
        if len(self._log_buffer) >= LOG_BATCH_SIZE:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.flush())
            except RuntimeError:
                pass

    def should_stop(self) -> bool:
        return self._stop_requested

    def should_close_trades(self) -> bool:
        """True if the dashboard stop command included 'close all trades'."""
        return self._close_trades_on_stop

    async def report_trade_open(
        self,
        symbol: str, side: str, entry_price: float, quantity: float,
        leverage: int = 10,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
        confidence: Optional[float] = None,
        signal_type: Optional[str] = None,
        regime: Optional[str] = None,
        risk_usdt: Optional[float] = None,
        order_id: Optional[str] = None,
    ) -> Optional[int]:
        payload = {
            "symbol": symbol, "side": side.lower(),
            "entry_price": entry_price, "quantity": quantity,
            "leverage": leverage, "tp_price": tp_price, "sl_price": sl_price,
            "confidence": round(float(confidence), 2) if confidence else None,
            "signal_type": signal_type, "regime": regime,
            "risk_usdt": risk_usdt, "order_id": order_id,
            "opened_at": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        }
        resp = await self._post_safe("trade/open", payload)
        if resp and resp.get("success"):
            trade_id = resp.get("trade_id")
            logger.debug(f"[Reporter] Trade open recorded — Laravel trade_id={trade_id}")
            return trade_id
        else:
            logger.warning(f"[Reporter] ⚠️ Failed to record trade open for {symbol}. "
                          f"Response: {resp}")
            return None

    async def report_trade_close(
        self,
        symbol: str, side: str, exit_price: float, pnl_usdt: float,
        pnl_r: Optional[float] = None,
        exit_reason: Optional[str] = None,
        bars_held: Optional[int] = None,
        trade_id: Optional[int] = None,
    ):
        # Clamp pnl_r to ±9999 — extreme values (e.g. 1,166,999) indicate a
        # near-zero risk denominator and would overflow DECIMAL(12,4) in MySQL.
        # The sign is preserved so wins/losses are still correctly reflected.
        safe_pnl_r = None
        if pnl_r is not None:
            safe_pnl_r = round(max(-9999.0, min(9999.0, float(pnl_r))), 4)

        payload = {
            "trade_id": trade_id, "symbol": symbol, "side": side.lower(),
            "exit_price": exit_price, "pnl_usdt": round(float(pnl_usdt), 4),
            "pnl_r": safe_pnl_r,
            "exit_reason": exit_reason, "bars_held": bars_held,
            "closed_at": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        }
        resp = await self._post_safe("trade/close", payload)
        if not (resp and resp.get("success")):
            logger.warning(f"[Reporter] ⚠️ Failed to record trade close for {symbol}. "
                          f"Response: {resp}")

    async def report_signal(
        self,
        symbol: str, signal: str, confidence: float,
        signal_type: Optional[str] = None, regime: Optional[str] = None,
        adx: Optional[float] = None, atr_ratio: Optional[float] = None,
        ema_long: Optional[float] = None, ema_short: Optional[float] = None,
        action_taken: Optional[str] = None, price_at_signal: Optional[float] = None,
    ):
        await self._post_safe("signal", {
            "symbol": symbol, "signal": signal.lower(),
            "confidence": round(float(confidence), 2),
            "signal_type": signal_type, "regime": regime,
            "adx": round(adx, 4) if adx else None,
            "atr_ratio": round(atr_ratio, 4) if atr_ratio else None,
            "ema_long": round(ema_long, 2) if ema_long else None,
            "ema_short": round(ema_short, 2) if ema_short else None,
            "action_taken": action_taken, "price_at_signal": price_at_signal,
            "signaled_at": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        })

    async def report_status(self, status: str, message: str = ""):
        resp = await self._post_safe("status", {"status": status, "message": message})
        if not resp:
            # Retry once after 2s — status update is critical
            await asyncio.sleep(2)
            await self._post_safe("status", {"status": status, "message": message})

    # ── Background tasks ───────────────────────────────────────────────────

    async def _heartbeat_loop(self):
        while True:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                resp = await self._post_safe("heartbeat", {})
                if resp:
                    for cmd in resp.get("commands", []):
                        if cmd.get("command") == "stop":
                            close_trades = cmd.get("close_open_trades", False)
                            self._close_trades_on_stop = close_trades
                            logger.info(
                                f"[Reporter] 🛑 Stop command received from Laravel. "
                                f"{'Will close open positions before exit.' if close_trades else 'Positions will remain open on exchange.'}"
                            )
                            self._stop_requested = True
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[Reporter] Heartbeat error: {e}")

    async def _flush_loop(self):
        while True:
            try:
                await asyncio.sleep(LOG_FLUSH_INTERVAL)
                if self._log_buffer:
                    await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[Reporter] Flush error: {e}")

    # ── HTTP helper ────────────────────────────────────────────────────────

    async def _post_safe(self, endpoint: str, payload: dict) -> Optional[dict]:
        url = f"{self.api_url}/{endpoint}"
        try:
            session = await self._get_session()
            async with session.post(url, json=payload) as resp:
                if resp.status == 401:
                    logger.warning(f"[Reporter] 401 Unauthorized on /{endpoint} — "
                                   f"bot_token may be invalid or expired")
                    return None
                if resp.status == 404:
                    logger.warning(f"[Reporter] 404 Not Found: {url} — "
                                   f"check APP_URL in .env and routes are registered")
                    return None
                if resp.status == 500:
                    text = await resp.text()
                    logger.warning(f"[Reporter] 500 Server Error on /{endpoint}: "
                                   f"{text[:300]}")
                    return None
                if resp.content_type == "application/json":
                    return await resp.json()
                return {"status": resp.status}
        except aiohttp.ClientConnectorError as e:
            logger.warning(f"[Reporter] Cannot reach Laravel /{endpoint}: {e}")
        except asyncio.TimeoutError:
            logger.warning(f"[Reporter] Timeout on /{endpoint}")
        except Exception as e:
            logger.warning(f"[Reporter] HTTP error on /{endpoint}: {type(e).__name__}: {e}")
        return None

    # ── Logging integration ────────────────────────────────────────────────

    def attach_to_logging(self):
        """
        Hook Python's root logger so every logger.info/warning/error/critical
        call in the bot automatically appears in the dashboard.
        Call ONCE at startup.
        """
        handler = _LaravelLogHandler(self)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(name)s — %(message)s"))
        # Avoid duplicate handlers if called multiple times
        root = logging.getLogger()
        for h in root.handlers:
            if isinstance(h, _LaravelLogHandler):
                return
        root.addHandler(handler)


class _LaravelLogHandler(logging.Handler):
    """Routes Python log records to LaravelReporter's log buffer."""

    CHANNEL_MAP = {
        "core.trader":   "bot",
        "core.exchange": "trade",
        "core.market":   "signal",
        "core.models":   "signal",
        "core.risk":     "bot",
        "core.strategy": "bot",
        "__main__":      "bot",
    }

    # Messages to skip — these are DEBUG-level noise from internal libs
    SKIP_PATTERNS = [
        "[Reporter]",          # avoid feedback loop of reporter logging itself
        "ccxt",                # ccxt internal messages
        "aiohttp",             # aiohttp connection pool noise
        "asyncio",             # event loop noise
    ]

    def __init__(self, reporter: LaravelReporter):
        super().__init__()
        self.reporter = reporter

    def emit(self, record: logging.LogRecord):
        try:
            level = record.levelname.lower()
            if level not in ("info", "warning", "error", "critical"):
                return

            msg = self.format(record)

            # Skip internal reporter/library noise
            for pattern in self.SKIP_PATTERNS:
                if pattern in msg or pattern in record.name:
                    return

            channel = "bot"
            for prefix, ch in self.CHANNEL_MAP.items():
                if record.name.startswith(prefix):
                    channel = ch
                    break

            self.reporter.queue_log(
                level   = level,
                message = msg,
                channel = channel,
            )
        except Exception:
            pass  # NEVER crash the bot due to logging failure