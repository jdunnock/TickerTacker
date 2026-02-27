from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any

from sqlalchemy import desc

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = str(REPO_ROOT / ".venv" / "bin" / "python")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.db import get_session
from app.models import Watchlist


def _load_database_url() -> str | None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "DATABASE_URL":
            return value.strip()
    return None


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    return value


def _resolve_owner_from_db() -> str | None:
    db = get_session()
    try:
        row = (
            db.query(Watchlist.owner)
            .filter(Watchlist.owner.isnot(None), Watchlist.owner != "")
            .order_by(desc(Watchlist.added_at))
            .first()
        )
        if not row:
            return None
        return row[0]
    finally:
        db.close()


async def _run(tool_name: str, args: dict[str, Any]) -> None:
    database_url = _load_database_url()
    env = {"DATABASE_URL": database_url} if database_url else None

    params = StdioServerParameters(
        command=PYTHON_BIN,
        args=["-m", "app.mcp_server"],
        cwd=str(REPO_ROOT),
        env=env,
    )

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            if tool_name == "list-tools":
                result = await session.list_tools()
            else:
                result = await session.call_tool(tool_name, args)

            print(json.dumps(_to_jsonable(result), indent=2, default=str, ensure_ascii=False))


async def _run_scenario(owner: str, symbol: str, upper: float | None, lower: float | None) -> None:
    database_url = _load_database_url()
    env = {"DATABASE_URL": database_url} if database_url else None

    params = StdioServerParameters(
        command=PYTHON_BIN,
        args=["-m", "app.mcp_server"],
        cwd=str(REPO_ROOT),
        env=env,
    )

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            print("\n=== 1) Quote ===")
            quote_result = await session.call_tool("get_quote", {"symbol": symbol})
            print(json.dumps(_to_jsonable(quote_result), indent=2, default=str, ensure_ascii=False))

            print("\n=== 2) Set alert ===")
            alert_payload: dict[str, Any] = {"owner": owner, "symbol": symbol}
            if upper is not None:
                alert_payload["upper"] = upper
            if lower is not None:
                alert_payload["lower"] = lower
            set_alert_result = await session.call_tool("set_alert", alert_payload)
            print(json.dumps(_to_jsonable(set_alert_result), indent=2, default=str, ensure_ascii=False))

            print("\n=== 3) Watchlist ===")
            watchlist_result = await session.call_tool("list_watchlist", {"owner": owner})
            print(json.dumps(_to_jsonable(watchlist_result), indent=2, default=str, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TickerTacker MCP tools from VS Code terminal")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-tools", help="List available MCP tools")

    sub.add_parser("owner", help="Print first valid owner from DB")

    watchlist = sub.add_parser("watchlist", help="Call list_watchlist")
    watchlist.add_argument("--owner", required=True)

    quote = sub.add_parser("quote", help="Call get_quote")
    quote.add_argument("--symbol", required=True)

    latest = sub.add_parser("latest", help="Call latest_close_from_db")
    latest.add_argument("--symbol", required=True)

    set_alert = sub.add_parser("set-alert", help="Call set_alert")
    set_alert.add_argument("--owner", required=True)
    set_alert.add_argument("--symbol", required=True)
    set_alert.add_argument("--upper", type=float)
    set_alert.add_argument("--lower", type=float)

    clear_alert = sub.add_parser("clear-alert", help="Call clear_alert")
    clear_alert.add_argument("--owner", required=True)
    clear_alert.add_argument("--symbol", required=True)

    scenario = sub.add_parser("scenario", help="Run quote -> set-alert -> watchlist in one command")
    scenario.add_argument("--owner", required=False)
    scenario.add_argument("--symbol", required=True)
    scenario.add_argument("--upper", type=float)
    scenario.add_argument("--lower", type=float)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list-tools":
        asyncio.run(_run("list-tools", {}))
        return

    if args.command == "owner":
        owner = _resolve_owner_from_db()
        if not owner:
            print("No valid owner found in watchlist table.")
            return
        print(owner)
        return

    if args.command == "watchlist":
        asyncio.run(_run("list_watchlist", {"owner": args.owner}))
        return

    if args.command == "quote":
        asyncio.run(_run("get_quote", {"symbol": args.symbol}))
        return

    if args.command == "latest":
        asyncio.run(_run("latest_close_from_db", {"symbol": args.symbol}))
        return

    if args.command == "set-alert":
        payload: dict[str, Any] = {"owner": args.owner, "symbol": args.symbol}
        if args.upper is not None:
            payload["upper"] = args.upper
        if args.lower is not None:
            payload["lower"] = args.lower
        asyncio.run(_run("set_alert", payload))
        return

    if args.command == "clear-alert":
        asyncio.run(_run("clear_alert", {"owner": args.owner, "symbol": args.symbol}))
        return

    if args.command == "scenario":
        owner = args.owner or _resolve_owner_from_db()
        if not owner:
            print("No valid owner found. Pass --owner explicitly.")
            return
        asyncio.run(_run_scenario(owner, args.symbol, args.upper, args.lower))
        return


if __name__ == "__main__":
    main()
