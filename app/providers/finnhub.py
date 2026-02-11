import os
from datetime import datetime, timedelta
from typing import Optional

import httpx


class FinnhubProvider:
    """Fetch fundamentals from Finnhub."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY")
        if not self.api_key:
            raise RuntimeError("FINNHUB_API_KEY is not set")
        self.base_url = "https://finnhub.io/api/v1"

    def normalize_symbol(self, symbol: str) -> str:
        symbol = symbol.strip().upper()
        if symbol.endswith(".HE"):
            return "HEL:" + symbol[:-3].replace("-", " ")
        if symbol.endswith(".ST"):
            return "STO:" + symbol[:-3].replace("-", " ")
        if symbol.endswith(".NY") or symbol.endswith(".N") or symbol.endswith(".OQ"):
            return symbol[:-3]
        return symbol

    async def get_fundamentals(self, symbol: str) -> dict:
        normalized = self.normalize_symbol(symbol)

        async def fetch(symbol_value: str) -> tuple[dict, dict]:
            params = {"symbol": symbol_value, "token": self.api_key}
            async with httpx.AsyncClient(timeout=10) as client:
                metric_response = await client.get(
                    f"{self.base_url}/stock/metric",
                    params={**params, "metric": "all"},
                )
                profile_response = await client.get(
                    f"{self.base_url}/stock/profile2", params=params
                )

            metric_data = (
                metric_response.json() if metric_response.status_code == 200 else {}
            )
            profile_data = (
                profile_response.json() if profile_response.status_code == 200 else {}
            )
            metrics = (
                metric_data.get("metric", {}) if isinstance(metric_data, dict) else {}
            )
            return metrics, profile_data

        metrics, profile_data = await fetch(normalized)
        if not metrics and ":" not in normalized:
            metrics, profile_data = await fetch(f"US:{normalized}")

        def to_float(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        return {
            "pe": to_float(metrics.get("peTTM") or metrics.get("peNormalizedAnnual")),
            "ps": to_float(metrics.get("psTTM") or metrics.get("psAnnual")),
            "evToEbitda": to_float(
                metrics.get("evToEbitdaTTM") or metrics.get("evToEbitdaAnnual")
            ),
            "ebitda": to_float(metrics.get("ebitdaTTM") or metrics.get("ebitdaAnnual")),
            "marketCap": to_float(profile_data.get("marketCapitalization")),
            "enterpriseValue": to_float(metrics.get("enterpriseValue")),
            "industry": profile_data.get("finnhubIndustry"),
            "sector": profile_data.get("sector"),
        }

    async def get_company_news(self, symbol: str, days: int = 7) -> list[dict]:
        normalized = self.normalize_symbol(symbol)
        to_date = datetime.utcnow().date()
        from_date = to_date - timedelta(days=days)
        params = {
            "symbol": normalized,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "token": self.api_key,
        }
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{self.base_url}/company-news", params=params)
        if response.status_code != 200:
            return []
        data = response.json()
        return data if isinstance(data, list) else []
