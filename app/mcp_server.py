from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal

from mcp.server.fastmcp import FastMCP
from sqlalchemy import desc

from app.db import get_session
from app.models import Instrument, Price, Watchlist
from app.providers.yfinance import YFinanceProvider

mcp = FastMCP("TickerTacker MCP")


@contextmanager
def session_scope():
    db = get_session()
    try:
        yield db
    finally:
        db.close()


def _to_float(value: Decimal | float | int | None) -> float | None:
    if value is None:
        return None
    return float(value)


@mcp.tool()
def list_watchlist(owner: str) -> list[dict]:
    with session_scope() as db:
        rows = (
            db.query(Watchlist)
            .filter(Watchlist.owner == owner)
            .order_by(desc(Watchlist.added_at))
            .all()
        )
        return [
            {
                "watchlist_id": row.id,
                "symbol": row.instrument.symbol if row.instrument else None,
                "exchange": row.instrument.exchange if row.instrument else None,
                "alert_upper": _to_float(row.alert_upper),
                "alert_lower": _to_float(row.alert_lower),
                "alert_triggered": bool(row.alert_triggered),
            }
            for row in rows
        ]


@mcp.tool()
async def get_quote(symbol: str) -> dict:
    provider = YFinanceProvider()
    quote = await provider.get_quote(symbol)
    if quote is None:
        return {"symbol": symbol.upper(), "found": False}

    return {
        "symbol": quote.symbol,
        "found": True,
        "close": float(quote.close),
        "timestamp": quote.timestamp.isoformat(),
    }


@mcp.tool()
def latest_close_from_db(symbol: str) -> dict:
    with session_scope() as db:
        instrument = (
            db.query(Instrument)
            .filter(Instrument.symbol == symbol.upper())
            .first()
        )
        if not instrument:
            return {"symbol": symbol.upper(), "found": False}

        latest = (
            db.query(Price)
            .filter(Price.instrument_id == instrument.id)
            .order_by(desc(Price.timestamp))
            .first()
        )
        if not latest:
            return {"symbol": symbol.upper(), "found": False}

        return {
            "symbol": symbol.upper(),
            "found": True,
            "close": _to_float(latest.close),
            "timestamp": latest.timestamp.isoformat(),
        }


@mcp.tool()
def set_alert(owner: str, symbol: str, upper: float | None = None, lower: float | None = None) -> dict:
    if upper is None and lower is None:
        return {"success": False, "error": "Provide at least one threshold (upper or lower)."}

    if upper is not None and lower is not None and lower >= upper:
        return {"success": False, "error": "Lower threshold must be less than upper threshold."}

    with session_scope() as db:
        instrument = (
            db.query(Instrument)
            .filter(Instrument.symbol == symbol.upper())
            .first()
        )
        if not instrument:
            return {"success": False, "error": "Instrument not found."}

        row = (
            db.query(Watchlist)
            .filter(Watchlist.owner == owner, Watchlist.instrument_id == instrument.id)
            .first()
        )
        if not row:
            return {"success": False, "error": "Watchlist item not found for owner."}

        row.alert_upper = upper
        row.alert_lower = lower
        row.alert_triggered = False
        db.commit()

        return {
            "success": True,
            "watchlist_id": row.id,
            "symbol": symbol.upper(),
            "alert_upper": _to_float(row.alert_upper),
            "alert_lower": _to_float(row.alert_lower),
        }


@mcp.tool()
def clear_alert(owner: str, symbol: str) -> dict:
    with session_scope() as db:
        instrument = (
            db.query(Instrument)
            .filter(Instrument.symbol == symbol.upper())
            .first()
        )
        if not instrument:
            return {"success": False, "error": "Instrument not found."}

        row = (
            db.query(Watchlist)
            .filter(Watchlist.owner == owner, Watchlist.instrument_id == instrument.id)
            .first()
        )
        if not row:
            return {"success": False, "error": "Watchlist item not found for owner."}

        row.alert_upper = None
        row.alert_lower = None
        row.alert_triggered = False
        db.commit()

        return {"success": True, "watchlist_id": row.id, "symbol": symbol.upper()}


if __name__ == "__main__":
    mcp.run(transport="stdio")
