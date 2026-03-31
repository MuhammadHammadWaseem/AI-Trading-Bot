"""
config/logger.py
----------------
Centralized logging — colored terminal + rotating file logs.
Emoji-free for Windows terminal compatibility.

Windows log rotation fix
------------------------
Python's RotatingFileHandler rotates by renaming the current log file
(trading_bot.log → trading_bot.log.1) then opening a new one.
On Windows this raises WinError 32 ("file in use") because the OS holds
an exclusive file lock while the process has it open — even with a
shared handle.

Fix: switch to TimedRotatingFileHandler with when="midnight".
  - Rotation happens once per day at midnight, not mid-session when the
    file reaches a size limit.
  - At midnight Windows briefly has the old file closed before the new
    one opens, so the rename succeeds.
  - backupCount=7 keeps one week of daily logs.

If you still want size-based rotation on Windows, the only reliable
approach is to use when="midnight" and accept daily rotation, or to
write a custom handler that copies then truncates instead of renaming
(see WindowsSafeRotatingFileHandler below — enabled automatically on
Windows, RotatingFileHandler used on Linux/Mac).
"""

import logging
import os
import sys
import shutil
import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
import colorlog

from config.settings import settings


# ── Windows-safe size-based rotating handler ──────────────────────────────────

class _WindowsSafeRotatingFileHandler(RotatingFileHandler):
    """
    Drop-in replacement for RotatingFileHandler that works on Windows.

    Standard RotatingFileHandler rotates by:
        os.rename(current_log, current_log.1)   ← WinError 32 — file locked

    This subclass rotates by:
        1. Close the current file handle.
        2. Copy current_log → current_log.1 (copy, not rename).
        3. Truncate current_log to 0 bytes (keeps the same inode/handle).
        4. Re-open current_log for continued writing.

    This avoids the rename entirely so Windows never needs to move a
    file that another handle has open.
    """

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None

        base = self.baseFilename

        # Shift existing backups: .4 → .5, .3 → .4 … .1 → .2
        for i in range(self.backupCount - 1, 0, -1):
            src = f"{base}.{i}"
            dst = f"{base}.{i + 1}"
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.copy2(src, dst)

        # Copy current log to .1
        dst1 = f"{base}.1"
        if os.path.exists(base):
            if os.path.exists(dst1):
                os.remove(dst1)
            shutil.copy2(base, dst1)

        # Truncate the current log (open in write mode to zero it out)
        with open(base, "w", encoding=self.encoding or "utf-8"):
            pass

        # Re-open for append
        self.mode = "a"
        self.stream = self._open()


# ── Public factory ─────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # ── Terminal handler (colored) ─────────────────────────────────────────
    terminal_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)-8s]%(reset)s %(cyan)s%(name)s%(reset)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG":    "white",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "bold_red",
        },
    )
    terminal_handler = logging.StreamHandler()
    terminal_handler.setFormatter(terminal_formatter)
    terminal_handler.stream.reconfigure(encoding="utf-8", errors="replace")
    logger.addHandler(terminal_handler)

    # ── File handler (rotating) ────────────────────────────────────────────
    if settings.log_to_file:
        logs_dir = settings.logs_dir
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "trading_bot.log"

        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if sys.platform == "win32":
            # Windows: use copy-truncate strategy to avoid WinError 32.
            # Rotates at 10 MB, keeps 5 backups, never renames the active file.
            file_handler = _WindowsSafeRotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
        else:
            # Linux/Mac: standard rename-based rotation works fine.
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )

        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger