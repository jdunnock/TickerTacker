from __future__ import annotations

import httpx


class StocktwitsProvider:
    """Fetch public symbol streams from Stocktwits."""

    base_url = "https://api.stocktwits.com/api/2"

    async def get_symbol_stream(self, symbol: str, limit: int = 20) -> list[dict]:
        symbol = symbol.strip().upper()
        if not symbol:
            return []
        params = {"limit": str(limit)}
        url = f"{self.base_url}/streams/symbol/{symbol}.json"
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url, params=params)
        if response.status_code != 200:
            return []
        data = response.json()
        messages = data.get("messages") if isinstance(data, dict) else None
        return messages if isinstance(messages, list) else []
