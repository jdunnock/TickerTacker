import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

try:
    import yfinance as yf
except ImportError:
    yf = None

from app.providers.base import PriceProvider, PriceBar, QuoteData


class YFinanceProvider(PriceProvider):
    """Price provider using yfinance (Yahoo Finance data)."""

    def __init__(self):
        """Initialize provider."""
        if yf is None:
            raise RuntimeError("yfinance is required for YFinanceProvider")

    async def get_daily_history(self, symbol: str, days: int = 30) -> list[PriceBar]:
        """
        Fetch daily OHLCV data from Yahoo Finance.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            days: Number of days of history to fetch

        Returns:
            List of PriceBar objects
        """
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            data = await loop.run_in_executor(
                None,
                lambda: yf.download(symbol.upper(), start=start_date, progress=False),
            )

            if data is None or data.empty:
                return []

            # yfinance returns MultiIndex columns for single ticker
            # Use xs (cross-section) to select ticker and flatten columns
            if hasattr(data.columns, "names") and "Ticker" in data.columns.names:
                try:
                    data = data.xs(symbol.upper(), level="Ticker", axis=1)
                except KeyError:
                    # Ticker not found in MultiIndex, try direct column selection
                    pass

            bars = []
            # Process each row
            for idx in range(len(data)):
                try:
                    row = data.iloc[idx]
                    bar = PriceBar(
                        symbol=symbol.upper(),
                        timestamp=data.index[idx].to_pydatetime(),
                        open=Decimal(str(float(row["Open"]))),
                        high=Decimal(str(float(row["High"]))),
                        low=Decimal(str(float(row["Low"]))),
                        close=Decimal(str(float(row["Close"]))),
                        volume=int(float(row["Volume"]))
                        if float(row["Volume"]) > 0
                        else None,
                    )
                    bars.append(bar)
                except (ValueError, TypeError, KeyError):
                    continue

            return bars
        except Exception as e:
            print(f"Error fetching data from yfinance for {symbol}: {e}")
            return []

    async def get_quote(self, symbol: str) -> Optional[QuoteData]:
        """
        Fetch current quote from Yahoo Finance.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            QuoteData with latest close price, or None if unavailable
        """
        try:
            loop = asyncio.get_event_loop()

            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol.upper()))

            history = await loop.run_in_executor(
                None, lambda: ticker.history(period="1d")
            )

            if history.empty:
                return None

            latest = history.iloc[-1]
            return QuoteData(
                symbol=symbol.upper(),
                close=Decimal(str(latest["Close"])),
                timestamp=history.index[-1].to_pydatetime(),
            )
        except Exception as e:
            print(f"Error fetching quote from yfinance for {symbol}: {e}")
            return None

    async def get_intraday_history(
        self, symbol: str, interval: str = "5m"
    ) -> list[PriceBar]:
        """Fetch intraday OHLCV data from Yahoo Finance."""
        try:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol.upper()))
            data = await loop.run_in_executor(
                None, lambda: ticker.history(period="5d", interval=interval)
            )

            if data is None or data.empty:
                return []

            bars = []
            for idx in range(len(data)):
                try:
                    row = data.iloc[idx]
                    bar = PriceBar(
                        symbol=symbol.upper(),
                        timestamp=data.index[idx].to_pydatetime(),
                        open=Decimal(str(float(row["Open"]))),
                        high=Decimal(str(float(row["High"]))),
                        low=Decimal(str(float(row["Low"]))),
                        close=Decimal(str(float(row["Close"]))),
                        volume=int(float(row["Volume"]))
                        if float(row.get("Volume", 0)) > 0
                        else None,
                    )
                    bars.append(bar)
                except (ValueError, TypeError, KeyError):
                    continue

            return bars
        except Exception as e:
            print(f"Error fetching intraday data from yfinance for {symbol}: {e}")
            return []

    async def get_instrument_details(
        self, symbol: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Fetch instrument name and exchange from Yahoo Finance."""
        try:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol.upper()))
            info = await loop.run_in_executor(None, lambda: ticker.info or {})

            name = info.get("shortName") or info.get("longName")
            exchange = info.get("exchange")
            return name, exchange
        except Exception as e:
            print(f"Error fetching instrument details for {symbol}: {e}")
            return None, None

    async def get_fundamentals(self, symbol: str) -> dict:
        """Fetch basic fundamentals from Yahoo Finance."""
        try:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol.upper()))
            info = await loop.run_in_executor(None, lambda: ticker.info or {})

            def to_float(value):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            return {
                "pe": to_float(info.get("trailingPE")),
                "ps": to_float(info.get("priceToSalesTrailing12Months")),
                "evToEbitda": to_float(info.get("enterpriseToEbitda")),
                "ebitda": to_float(info.get("ebitda")),
                "marketCap": to_float(info.get("marketCap")),
                "enterpriseValue": to_float(info.get("enterpriseValue")),
            }
        except Exception as e:
            print(f"Error fetching fundamentals for {symbol}: {e}")
            return {}
