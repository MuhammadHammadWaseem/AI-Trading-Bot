"""
config/settings.py
──────────────────
Central configuration — loads from .env and provides typed settings
to every module in the project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

# ─── Load .env ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


# ─── Exchange Credentials ─────────────────────────────────────────────────────
class ExchangeCredentials(BaseModel):
    api_key: str
    secret: str
    testnet: bool = True


# ─── Risk Settings ────────────────────────────────────────────────────────────
class RiskSettings(BaseModel):
    leverage: int = Field(default=10, ge=1, le=125)
    risk_per_trade_pct: float = Field(default=0.5, gt=0, le=10)
    take_profit_pct: float = Field(default=2.0, gt=0)
    stop_loss_pct: float = Field(default=1.0, gt=0)
    max_open_trades: int = Field(default=5, ge=1)
    max_daily_loss_pct: float = Field(default=5.0, gt=0)
    max_daily_loss_usdt: float = Field(default=0.0, ge=0)


# ─── Model Settings ───────────────────────────────────────────────────────────
class ModelSettings(BaseModel):
    retrain_interval_hours: int = 24
    lookback_candles: int = 100
    confidence_threshold: float = 0.65
    saved_models_dir: Path = BASE_DIR / "saved_models"


# ─── Main Settings ────────────────────────────────────────────────────────────
class Settings(BaseModel):
    # Environment
    environment: str = os.getenv("ENVIRONMENT", "testnet")
    base_dir: Path = BASE_DIR
    logs_dir: Path = BASE_DIR / "logs"

    # Exchange
    binance_testnet: ExchangeCredentials = ExchangeCredentials(
        api_key=os.getenv("BINANCE_TESTNET_API_KEY", ""),
        secret=os.getenv("BINANCE_TESTNET_SECRET", ""),
        testnet=True,
    )
    binance_live: ExchangeCredentials = ExchangeCredentials(
        api_key=os.getenv("BINANCE_LIVE_API_KEY", ""),
        secret=os.getenv("BINANCE_LIVE_SECRET", ""),
        testnet=False,
    )

    # Trading pairs
    trading_pairs: List[str] = os.getenv(
        "TRADING_PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT"
    ).split(",")

    # Risk — default risk_per_trade_pct changed to 0.5
    risk: RiskSettings = RiskSettings(
        leverage=int(os.getenv("DEFAULT_LEVERAGE", 10)),
        risk_per_trade_pct=float(os.getenv("DEFAULT_RISK_PER_TRADE", 0.5)),
        take_profit_pct=float(os.getenv("DEFAULT_TAKE_PROFIT_PCT", 2.0)),
        stop_loss_pct=float(os.getenv("DEFAULT_STOP_LOSS_PCT", 1.0)),
        max_open_trades=int(os.getenv("MAX_OPEN_TRADES", 5)),
        max_daily_loss_pct=float(os.getenv("MAX_DAILY_LOSS_PCT", 5.0)),
    )

    # Model
    model: ModelSettings = ModelSettings(
        retrain_interval_hours=int(os.getenv("MODEL_RETRAIN_INTERVAL_HOURS", 24)),
        lookback_candles=int(os.getenv("LOOKBACK_CANDLES", 100)),
        confidence_threshold=float(os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", 0.65)),
    )

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_to_file: bool = os.getenv("LOG_TO_FILE", "true").lower() == "true"

    def get_exchange_credentials(self, exchange: str = "binance") -> ExchangeCredentials:
        """Get credentials based on current environment."""
        if self.environment == "testnet":
            return self.binance_testnet
        return self.binance_live

    class Config:
        arbitrary_types_allowed = True


# ─── Singleton instance ───────────────────────────────────────────────────────
settings = Settings()