import csv
from datetime import datetime
from decimal import Decimal
from io import StringIO
from typing import Optional

try:
    import aiohttp
except ImportError:
    aiohttp = None

from app.providers.base import PriceProvider, PriceBar, QuoteData


class StooqProvider(PriceProvider):
    """Price provider using Stooq API (free historical and current data)."""

    BASE_URL = "https://stooq.com"

    async def get_daily_history(self, symbol: str, days: int = 30) -> list[PriceBar]:
        """
        Fetch daily OHLCV data from Stooq.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            days: Number of days to fetch (approximate)

        Returns:
            List of PriceBar objects
        """
        if aiohttp is None:
            raise RuntimeError("aiohttp is required for StooqProvider")

        url = f"{self.BASE_URL}/q/stooq/download/?s={symbol.upper()}&d1=&d2=&i=d"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status != 200:
                        return []

                    content = await resp.text()
                    return self._parse_csv_data(symbol, content, days)
        except Exception as e:
            print(f"Error fetching data from Stooq for {symbol}: {e}")
            return []

    async def get_quote(self, symbol: str) -> Optional[QuoteData]:
        """
        Fetch current quote from Stooq.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            QuoteData with latest close price, or None if unavailable
        """
        if aiohttp is None:
            raise RuntimeError("aiohttp is required for StooqProvider")

        try:
            # Stooq quote endpoint
            url = f"{self.BASE_URL}/q/stooq/download/?s={symbol.upper()}&d1=&d2=&i=d"

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status != 200:
                        return None

                    content = await resp.text()
                    bars = self._parse_csv_data(symbol, content, days=1)

                    if bars:
                        latest = bars[0]  # Most recent bar
                        return QuoteData(
                            symbol=symbol.upper(),
                            close=latest.close,
                            timestamp=latest.timestamp,
                        )
                    return None
        except Exception as e:
            print(f"Error fetching quote from Stooq for {symbol}: {e}")
            return None

    @staticmethod
    def _parse_csv_data(symbol: str, csv_content: str, days: int) -> list[PriceBar]:
        """
        Parse CSV response from Stooq.

        Expected format:
        Date,Open,High,Low,Close,Volume
        """
        bars = []
        try:
            reader = csv.DictReader(StringIO(csv_content))
            if reader is None or reader.fieldnames is None:
                return []

            count = 0
            for row in reader:
                if count >= days:
                    break

                try:
                    bar = PriceBar(
                        symbol=symbol.upper(),
                        timestamp=datetime.strptime(row["Date"], "%Y-%m-%d"),
                        open=Decimal(row["Open"]),
                        high=Decimal(row["High"]),
                        low=Decimal(row["Low"]),
                        close=Decimal(row["Close"]),
                        volume=int(float(row["Volume"])) if row.get("Volume") else None,
                    )
                    bars.append(bar)
                    count += 1
                except (ValueError, KeyError) as e:
                    print(f"Error parsing row for {symbol}: {e}")
                    continue

            return bars
        except Exception as e:
            print(f"Error parsing CSV for {symbol}: {e}")
            return []
