from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass
class QuoteData:
    """Current price quote data."""

    symbol: str
    close: Decimal
    timestamp: datetime


@dataclass
class PriceBar:
    """OHLCV data for a single bar."""

    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int | None = None


class PriceProvider(ABC):
    """Abstract base class for price data providers."""

    @abstractmethod
    async def get_daily_history(self, symbol: str, days: int = 30) -> list[PriceBar]:
        """Fetch daily price history for a symbol."""
        pass

    @abstractmethod
    async def get_quote(self, symbol: str) -> QuoteData:
        """Fetch current quote for a symbol."""
        pass
