from sqlalchemy import func, and_
from sqlalchemy.orm import Session

from app.models import Instrument, Price, Watchlist
from app.providers.base import PriceProvider


class PricingService:
    """Service for updating prices with upsert logic."""

    def __init__(self, db: Session, provider: PriceProvider):
        """
        Initialize pricing service.

        Args:
            db: SQLAlchemy session
            provider: Price data provider (implements PriceProvider)
        """
        self.db = db
        self.provider = provider

    async def refresh_watchlist_prices(self) -> dict:
        """
        Refresh prices for all instruments in watchlist.

        Returns:
            Dictionary with refresh stats (updated_count, error_count, symbols)
        """
        watchlist_items = self.db.query(Watchlist.instrument_id).distinct().all()
        symbols = [
            self.db.query(Instrument.symbol).filter(Instrument.id == item[0]).scalar()
            for item in watchlist_items
        ]

        updated_count = 0
        error_count = 0
        updated_symbols = []

        for symbol in symbols:
            try:
                count = await self.update_instrument_prices(symbol)
                if count > 0:
                    updated_count += count
                    updated_symbols.append(symbol)
            except Exception as e:
                print(f"Error updating prices for {symbol}: {e}")
                error_count += 1

        self.db.commit()
        return {
            "updated_count": updated_count,
            "error_count": error_count,
            "symbols": updated_symbols,
        }

    async def update_instrument_prices(self, symbol: str, days: int = 30) -> int:
        """
        Update prices for a single instrument (upsert logic).

        Args:
            symbol: Symbol to update
            days: Number of days of history to fetch

        Returns:
            Number of price records updated/inserted
        """
        # Get instrument
        instrument = (
            self.db.query(Instrument)
            .filter(Instrument.symbol == symbol.upper())
            .first()
        )

        if not instrument:
            print(f"Instrument {symbol} not found in database")
            return 0

        # Fetch data from provider
        try:
            bars = await self.provider.get_daily_history(symbol, days)
        except Exception as e:
            print(f"Provider error for {symbol}: {e}")
            return 0

        if not bars:
            print(f"No data returned from provider for {symbol}")
            return 0

        updated_count = 0

        # Upsert logic: insert new or update existing
        for bar in bars:
            existing = (
                self.db.query(Price)
                .filter(
                    and_(
                        Price.instrument_id == instrument.id,
                        func.date(Price.timestamp) == bar.timestamp.date(),
                    )
                )
                .first()
            )

            if existing:
                # Update existing record
                existing.open = bar.open
                existing.high = bar.high
                existing.low = bar.low
                existing.close = bar.close
                existing.volume = bar.volume
                existing.timestamp = bar.timestamp
            else:
                # Insert new record
                new_price = Price(
                    instrument_id=instrument.id,
                    timestamp=bar.timestamp,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                )
                self.db.add(new_price)

            updated_count += 1

        return updated_count
