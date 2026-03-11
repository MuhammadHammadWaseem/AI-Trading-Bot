"""
core/exchange/exchange_factory.py
──────────────────────────────────
Factory pattern — create exchange instances by name.
Adding a new exchange = add one entry here. Nothing else changes.
"""

from config.settings import settings, ExchangeCredentials
from core.exchange.base_exchange import BaseExchange
from core.exchange.binance_exchange import BinanceExchange


EXCHANGE_MAP = {
    "binance": BinanceExchange,
    # "okx":    OKXExchange,     ← add here when ready
    # "bybit":  BybitExchange,   ← add here when ready
}


def create_exchange(
    exchange_name: str = "binance",
    credentials: ExchangeCredentials = None,
) -> BaseExchange:
    """
    Create and return an exchange instance.

    Args:
        exchange_name: "binance" | "okx" | "bybit"
        credentials:   Optional override; uses settings default if None

    Returns:
        BaseExchange instance (not yet connected — call await .connect())
    """
    exchange_name = exchange_name.lower()

    if exchange_name not in EXCHANGE_MAP:
        raise ValueError(
            f"Exchange '{exchange_name}' not supported. "
            f"Available: {list(EXCHANGE_MAP.keys())}"
        )

    if credentials is None:
        credentials = settings.get_exchange_credentials(exchange_name)

    exchange_class = EXCHANGE_MAP[exchange_name]
    return exchange_class(credentials)
