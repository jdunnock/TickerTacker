from fastapi import FastAPI, Depends, Form, Response
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import desc
from starlette.requests import Request
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
import os

from app.db import engine, SessionLocal
from app.models import Base, Watchlist, Price, Instrument
from app.providers.yfinance import YFinanceProvider
from app.providers.finnhub import FinnhubProvider
from app.services.pricing import PricingService

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

QUOTE_CACHE_TTL = timedelta(seconds=45)
QUOTE_CACHE: dict[str, tuple[datetime, float]] = {}
LAST_DISPLAY_CACHE: dict[str, float] = {}
ANALYST_NOTE_CACHE: dict[str, tuple[datetime.date, dict]] = {}

# Mount static files
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
    name="static",
)

# Setup Jinja2 templates
templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)


def get_db() -> Session:
    """Dependency to get DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_owner(request: Request, response: Response | None = None) -> str:
    owner = request.cookies.get("tt_owner") or "user1"
    if response is not None and request.cookies.get("tt_owner") is None:
        response.set_cookie(
            "tt_owner",
            owner,
            max_age=60 * 60 * 24 * 365,
            httponly=True,
            samesite="lax",
        )
    return owner


def compute_sma(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    window = values[-period:]
    return sum(window) / period


def compute_ema_series(values: list[float], period: int) -> list[float | None]:
    if len(values) < period:
        return [None for _ in values]
    k = 2 / (period + 1)
    ema_values: list[float | None] = [None for _ in values]
    initial = sum(values[:period]) / period
    ema_values[period - 1] = initial
    prev = initial
    for idx in range(period, len(values)):
        prev = values[idx] * k + prev * (1 - k)
        ema_values[idx] = prev
    return ema_values


def compute_rsi(values: list[float], period: int = 14) -> float | None:
    if len(values) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, period + 1):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    for i in range(period + 1, len(values)):
        change = values[i] - values[i - 1]
        gain = max(change, 0)
        loss = max(-change, 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def clean_float(value: float | None) -> float | None:
    if value is None:
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def normalize_sector(value: str | None) -> str:
    if not value:
        return "other"
    text = value.lower()
    if "bank" in text or "financial" in text:
        return "financial"
    if "utility" in text:
        return "utilities"
    if "energy" in text or "oil" in text or "gas" in text:
        return "energy"
    if "real estate" in text:
        return "realestate"
    if "health" in text or "pharma" in text or "biotech" in text:
        return "healthcare"
    if "tech" in text or "software" in text or "semiconductor" in text:
        return "tech"
    if "consumer" in text or "retail" in text:
        return "consumer"
    if "industrial" in text or "manufact" in text:
        return "industrial"
    return "other"


def score_ratio(value: float | None, metric: str, sector_key: str) -> str:
    if value is None:
        return "neutral"
    thresholds = {
        "tech": {"pe": (25, 40), "ps": (4, 8), "ev": (15, 25)},
        "financial": {"pe": (12, 18), "ps": (2, 4), "ev": (10, 15)},
        "utilities": {"pe": (18, 25), "ps": (2.5, 4), "ev": (12, 18)},
        "energy": {"pe": (12, 20), "ps": (1.5, 3), "ev": (6, 10)},
        "realestate": {"pe": (20, 30), "ps": (3, 5), "ev": (12, 18)},
        "healthcare": {"pe": (20, 35), "ps": (4, 8), "ev": (15, 25)},
        "consumer": {"pe": (18, 30), "ps": (2.5, 5), "ev": (12, 20)},
        "industrial": {"pe": (18, 30), "ps": (2.5, 5), "ev": (12, 20)},
        "other": {"pe": (18, 30), "ps": (2.5, 5), "ev": (12, 20)},
    }
    bucket = thresholds.get(sector_key, thresholds["other"])
    low, high = bucket[metric]
    if value <= low:
        return "good"
    if value <= high:
        return "neutral"
    return "bad"


def build_analyst_note(
    symbol: str,
    fundamentals: dict,
    technicals: dict,
    headlines: list[str],
    twits_summary: dict,
    twits_headlines: list[str],
) -> dict:
    sector_key = normalize_sector(
        fundamentals.get("industry") or fundamentals.get("sector")
    )
    pe_score = score_ratio(fundamentals.get("pe"), "pe", sector_key)
    ps_score = score_ratio(fundamentals.get("ps"), "ps", sector_key)
    ev_score = score_ratio(fundamentals.get("evToEbitda"), "ev", sector_key)
    score_map = {"good": "attractive", "neutral": "fair", "bad": "stretched"}

    valuation_sentence = (
        f"Valuation looks {score_map[pe_score]} on P/E, "
        f"{score_map[ps_score]} on P/S, and {score_map[ev_score]} on EV/EBITDA "
        f"relative to {sector_key} peers."
    )

    trend = "neutral"
    ma50 = technicals.get("ma50")
    ma200 = technicals.get("ma200")
    rsi = technicals.get("rsi")
    if ma50 is not None and ma200 is not None:
        if ma50 >= ma200 and (rsi or 0) >= 55:
            trend = "bullish"
        elif ma50 < ma200 and (rsi or 0) <= 45:
            trend = "bearish"
    trend_sentence = "Trend read is %s based on MA50/MA200 and RSI." % trend

    if headlines:
        headline_text = "; ".join(headlines[:2])
        news_sentence = f"Recent headlines: {headline_text}."
    else:
        news_sentence = "No notable headlines in the last 7 days."

    sentiment_sentence = ""
    if twits_summary:
        bullish = twits_summary.get("bullish", 0)
        bearish = twits_summary.get("bearish", 0)
        total = twits_summary.get("total", 0)
        if total:
            sentiment_sentence = (
                "Stocktwits sentiment: %s bullish, %s bearish (last %s posts)."
                % (bullish, bearish, total)
            )

    twits_sentence = ""
    if twits_headlines:
        twits_sentence = "Stocktwits highlights: %s." % "; ".join(
            twits_headlines[:2]
        )

    parts = [valuation_sentence, trend_sentence, news_sentence]
    if sentiment_sentence:
        parts.append(sentiment_sentence)
    if twits_sentence:
        parts.append(twits_sentence)
    note = " ".join(parts)
    return {
        "text": note,
        "updatedAt": datetime.utcnow().isoformat() + "Z",
    }


async def build_watchlist_with_prices(
    db: Session, provider: YFinanceProvider, owner: str
) -> list[dict]:
    """Build watchlist rows using latest quote when available."""
    watchlist_items = db.query(Watchlist).filter(Watchlist.owner == owner).all()
    watchlist_with_prices = []

    for item in watchlist_items:
        prices = (
            db.query(Price)
            .filter(Price.instrument_id == item.instrument_id)
            .order_by(desc(Price.timestamp))
            .limit(10)
            .all()
        )

        latest_close = float(prices[0].close) if prices else 0.0
        prev_close = latest_close
        if prices:
            latest_date = prices[0].timestamp.date()
            for price in prices[1:]:
                if price.timestamp.date() < latest_date:
                    prev_close = float(price.close)
                    break

        last = latest_close

        symbol = item.instrument.symbol
        now = datetime.now()
        cached = QUOTE_CACHE.get(symbol)
        if cached and now - cached[0] <= QUOTE_CACHE_TTL:
            last = cached[1]
        else:
            try:
                quote = await provider.get_quote(symbol)
                if quote is not None:
                    last = float(quote.close)
                    QUOTE_CACHE[symbol] = (now, last)
            except Exception as e:
                print(f"Quote fetch failed for {symbol}: {e}")

        last_close = prev_close if prev_close else last

        change_pct = (
            ((last - last_close) / last_close * 100) if last_close != 0 else 0.0
        )

        changed = False
        previous_value = LAST_DISPLAY_CACHE.get(symbol)
        if previous_value is not None and previous_value != last:
            changed = True
        LAST_DISPLAY_CACHE[symbol] = last

        watchlist_with_prices.append(
            {
                "instrument": item.instrument,
                "last": last,
                "last_close": last_close,
                "change_pct": change_pct,
                "changed": changed,
            }
        )

    return watchlist_with_prices


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def home():
    return RedirectResponse(url="/dashboard", status_code=302)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    """Render dashboard with watchlist - refreshes prices on page load."""
    response = None
    owner = get_owner(request)
    # Refresh prices from provider on page load
    try:
        provider = YFinanceProvider()
        service = PricingService(db, provider)
        stats = await service.refresh_watchlist_prices()
        print(
            f"Dashboard load: Updated {stats['updated_count']} prices for {len(stats['symbols'])} symbols"
        )
    except Exception as e:
        print(f"Error refreshing prices on dashboard load: {e}")

    watchlist_with_prices = await build_watchlist_with_prices(db, provider, owner)

    response = templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "watchlist": watchlist_with_prices,
            "watchlist_cleared": request.query_params.get("cleared") == "1",
            "current_page": "dashboard",
        },
    )
    get_owner(request, response)
    return response


@app.get("/detail", response_class=HTMLResponse)
async def ticker_detail(request: Request, db: Session = Depends(get_db)):
    """Render ticker detail view for charts and analysis."""
    response = None
    owner = get_owner(request)
    provider = YFinanceProvider()
    watchlist_with_prices = await build_watchlist_with_prices(db, provider, owner)

    response = templates.TemplateResponse(
        "detail.html",
        {
            "request": request,
            "watchlist": watchlist_with_prices,
            "current_page": "detail",
        },
    )
    get_owner(request, response)
    return response


@app.get("/add-instruments", response_class=HTMLResponse)
def add_instruments_page(request: Request):
    """Show page to add instruments to watchlist."""
    return templates.TemplateResponse(
        "add_instruments.html",
        {
            "request": request,
        },
    )


@app.post("/api/instruments/add")
async def add_instruments(
    request: Request,
    symbols: list[str] = Form(default=[]),
    db: Session = Depends(get_db),
):
    """Add selected instruments to watchlist."""
    # Get form data
    form_data = await request.form()
    form_symbols = form_data.getlist("symbols")
    text_symbols = (form_data.get("symbols_text") or "").strip()

    symbols = []
    if form_symbols:
        symbols = form_symbols
    elif text_symbols:
        raw_parts = text_symbols.replace("\n", ",").split(",")
        symbols = [part.strip().upper() for part in raw_parts if part.strip()]

    if symbols:
        symbols = list(dict.fromkeys(symbols))

    if not symbols:
        return RedirectResponse(url="/add-instruments", status_code=303)

    added_count = 0
    response: RedirectResponse | None = None
    owner = get_owner(request)
    provider = None
    try:
        provider = YFinanceProvider()
    except Exception as e:
        print(f"Metadata provider unavailable: {e}")
    for symbol in symbols:
        # Check if already exists
        existing = db.query(Instrument).filter(Instrument.symbol == symbol).first()
        if existing:
            # Add to watchlist if not already there
            watchlist_item = (
                db.query(Watchlist)
                .filter(Watchlist.instrument_id == existing.id)
                .first()
            )
            if not watchlist_item:
                db.add(Watchlist(instrument_id=existing.id, owner=owner))
                added_count += 1
        else:
            # Create new instrument with the symbol; fill metadata if available
            instr = Instrument(symbol=symbol)
            if provider:
                name, exchange = await provider.get_instrument_details(symbol)
                if name:
                    instr.name = name
                if exchange:
                    instr.exchange = exchange
            db.add(instr)
            db.flush()
            db.add(Watchlist(instrument_id=instr.id, owner=owner))
            added_count += 1

    db.commit()

    # Redirect to dashboard
    response = RedirectResponse(url="/dashboard", status_code=303)
    get_owner(request, response)
    return response


@app.post("/api/prices/refresh", response_class=HTMLResponse)
async def refresh_prices(request: Request, db: Session = Depends(get_db)):
    """Refresh prices and return updated watchlist table (HTMX endpoint)."""
    response = None
    owner = get_owner(request)
    # Use PricingService to update prices from provider
    provider = YFinanceProvider()
    service = PricingService(db, provider)

    try:
        stats = await service.refresh_watchlist_prices()
        print(
            f"Price update: {stats['updated_count']} records updated, {stats['error_count']} errors"
        )
    except Exception as e:
        print(f"Error refreshing prices: {e}")

    # Fetch latest prices for all instruments in watchlist
    watchlist_with_prices = await build_watchlist_with_prices(db, provider, owner)

    # Return only the table partial for HTMX swap
    response = templates.TemplateResponse(
        "_watchlist_table.html",
        {
            "request": request,
            "watchlist": watchlist_with_prices,
        },
    )
    get_owner(request, response)
    return response


@app.post("/api/watchlist/clear")
def clear_watchlist(request: Request, db: Session = Depends(get_db)):
    """Clear all items from the watchlist."""
    owner = get_owner(request)
    db.query(Watchlist).filter(Watchlist.owner == owner).delete()
    db.commit()
    response = RedirectResponse(url="/dashboard?cleared=1", status_code=303)
    get_owner(request, response)
    return response


@app.get("/api/prices/history")
async def price_history(symbol: str, range: str = "6m", db: Session = Depends(get_db)):
    """Return recent close prices for a symbol."""
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return JSONResponse({"symbol": "", "prices": []})

    range_key = (range or "").strip().lower()
    now = datetime.now()
    if range_key == "1d":
        try:
            provider = YFinanceProvider()
            bars = []
            used_interval = None
            for interval in ("5m", "15m", "30m", "60m"):
                bars = await provider.get_intraday_history(symbol, interval=interval)
                if bars:
                    used_interval = interval
                    break
            if used_interval:
                first_ts = bars[0].timestamp
                last_ts = bars[-1].timestamp
                print(
                    f"Intraday {symbol} interval={used_interval} points={len(bars)} range={first_ts}..{last_ts}"
                )
            
            market_tz = None
            open_time = None
            close_time = None
            if symbol.endswith(".HE"):
                market_tz = ZoneInfo("Europe/Helsinki")
                open_time = time(10, 0)
                close_time = time(18, 30)
            elif symbol.endswith(".ST"):
                market_tz = ZoneInfo("Europe/Stockholm")
                open_time = time(10, 0)
                close_time = time(17, 30)
            elif symbol.endswith((".NY", ".N", ".OQ")):
                market_tz = ZoneInfo("America/New_York")
                open_time = time(9, 30)
                close_time = time(16, 0)
            else:
                market_tz = ZoneInfo("America/New_York")
                open_time = time(9, 30)
                close_time = time(16, 0)

            localized_bars = None
            if market_tz and bars:
                utc_tz = ZoneInfo("UTC")
                local_times_utc = []
                local_times_local = []
                for bar in bars:
                    ts = bar.timestamp
                    if ts.tzinfo is None:
                        local_times_utc.append(
                            ts.replace(tzinfo=utc_tz).astimezone(market_tz)
                        )
                        local_times_local.append(ts.replace(tzinfo=market_tz))
                    else:
                        local_ts = ts.astimezone(market_tz)
                        local_times_utc.append(local_ts)
                        local_times_local.append(local_ts)

                def filter_market_hours(candidate_times: list[datetime]) -> list[tuple]:
                    reference_date = max(candidate_times).date()
                    filtered_bars = []
                    for bar, local_ts in zip(bars, candidate_times):
                        if local_ts.date() != reference_date:
                            continue
                        if open_time and local_ts.time() < open_time:
                            continue
                        if close_time and local_ts.time() > close_time:
                            continue
                        filtered_bars.append((bar, local_ts))
                    return filtered_bars

                filtered_utc = filter_market_hours(local_times_utc)
                filtered_local = filter_market_hours(local_times_local)
                filtered = (
                    filtered_utc
                    if len(filtered_utc) >= len(filtered_local)
                    else filtered_local
                )

                if not filtered:
                    reference_utc = max(local_times_utc).date()
                    reference_local = max(local_times_local).date()
                    date_filtered_utc = [
                        (bar, local_ts)
                        for bar, local_ts in zip(bars, local_times_utc)
                        if local_ts.date() == reference_utc
                    ]
                    date_filtered_local = [
                        (bar, local_ts)
                        for bar, local_ts in zip(bars, local_times_local)
                        if local_ts.date() == reference_local
                    ]
                    filtered = (
                        date_filtered_utc
                        if len(date_filtered_utc) >= len(date_filtered_local)
                        else date_filtered_local
                    )

                if filtered:
                    localized_bars = filtered
                    bars = [bar for bar, _local_ts in filtered]
                else:
                    bars = []
                    localized_bars = []
                    print(
                        f"Intraday {symbol} filtered to 0 points for market hours {open_time}-{close_time}"
                    )

            if localized_bars:
                midnight_only = all(
                    local_ts.time() == time(0, 0) for _bar, local_ts in localized_bars
                )
                if midnight_only:
                    print(f"Intraday {symbol} returned daily bars; skipping")
                    bars = []
                    localized_bars = []
            data = [
                {
                    "t": (
                        local_ts.isoformat()
                        if localized_bars is not None
                        else bar.timestamp.isoformat()
                    ),
                    "o": float(bar.open),
                    "h": float(bar.high),
                    "l": float(bar.low),
                    "c": float(bar.close),
                    "v": float(bar.volume) if bar.volume is not None else None,
                }
                for bar, local_ts in (localized_bars or [(bar, bar.timestamp) for bar in bars])
            ]
            return JSONResponse({"symbol": symbol, "prices": data})
        except Exception as e:
            print(f"Error updating intraday history for {symbol}: {e}")
            return JSONResponse({"symbol": symbol, "prices": []})

    if range_key == "ytd":
        start_date = datetime(now.year, 1, 1)
        days = max((now - start_date).days, 1)
    elif range_key in {"1m", "1mo"}:
        start_date = now - timedelta(days=30)
        days = 30
    elif range_key in {"6m", "6mo"}:
        start_date = now - timedelta(days=182)
        days = 182
    elif range_key in {"1y", "1yr"}:
        start_date = now - timedelta(days=365)
        days = 365
    else:
        start_date = now - timedelta(days=182)
        days = 182

    instrument = db.query(Instrument).filter(Instrument.symbol == symbol).first()
    if not instrument:
        return JSONResponse({"symbol": symbol, "prices": []})

    try:
        provider = YFinanceProvider()
        service = PricingService(db, provider)
        await service.update_instrument_prices(symbol, days=days)
        db.commit()
    except Exception as e:
        print(f"Error updating history for {symbol}: {e}")

    prices = (
        db.query(Price)
        .filter(
            Price.instrument_id == instrument.id,
            Price.timestamp >= start_date,
        )
        .order_by(Price.timestamp)
        .all()
    )

    data = [
        {
            "t": price.timestamp.isoformat(),
            "o": float(price.open),
            "h": float(price.high),
            "l": float(price.low),
            "c": float(price.close),
            "v": float(price.volume) if price.volume is not None else None,
        }
        for price in prices
    ]
    return JSONResponse({"symbol": symbol, "prices": data})


@app.get("/api/analysis")
async def price_analysis(symbol: str, db: Session = Depends(get_db)):
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return JSONResponse({"symbol": "", "fundamentals": {}, "technicals": {}})

    provider = YFinanceProvider()
    instrument = db.query(Instrument).filter(Instrument.symbol == symbol).first()
    closes: list[float] = []

    if instrument:
        prices = (
            db.query(Price)
            .filter(Price.instrument_id == instrument.id)
            .order_by(desc(Price.timestamp))
            .limit(260)
            .all()
        )
        prices = list(reversed(prices))
        closes = [float(price.close) for price in prices if price.close is not None]

    if len(closes) < 220:
        try:
            service = PricingService(db, provider)
            await service.update_instrument_prices(symbol, days=800)
            db.commit()
        except Exception as e:
            print(f"Error refreshing history for {symbol}: {e}")

        if instrument:
            prices = (
                db.query(Price)
                .filter(Price.instrument_id == instrument.id)
                .order_by(desc(Price.timestamp))
                .limit(260)
                .all()
            )
            prices = list(reversed(prices))
            closes = [float(price.close) for price in prices if price.close is not None]

    if not closes:
        try:
            bars = await provider.get_daily_history(symbol, days=800)
            bars = sorted(bars, key=lambda bar: bar.timestamp)
            closes = [float(bar.close) for bar in bars if bar.close is not None]
        except Exception as e:
            print(f"Error fetching history fallback for {symbol}: {e}")

    rsi = compute_rsi(closes, 14) if closes else None
    ema12 = compute_ema_series(closes, 12)
    ema26 = compute_ema_series(closes, 26)
    macd_series = [
        (e12 - e26) if e12 is not None and e26 is not None else None
        for e12, e26 in zip(ema12, ema26)
    ]
    macd_values = [value for value in macd_series if value is not None]
    signal_series = compute_ema_series(macd_values, 9) if macd_values else []
    macd = macd_values[-1] if macd_values else None
    signal = signal_series[-1] if signal_series else None

    ma50 = compute_sma(closes, 50)
    ma200 = compute_sma(closes, 200)

    rsi = clean_float(rsi)
    macd = clean_float(macd)
    signal = clean_float(signal)
    ma50 = clean_float(ma50)
    ma200 = clean_float(ma200)

    fundamentals = {}
    try:
        finnhub = FinnhubProvider()
        fundamentals = await finnhub.get_fundamentals(symbol)
    except Exception:
        try:
            fundamentals = await provider.get_fundamentals(symbol)
        except Exception as e:
            print(f"Error fetching fundamentals for {symbol}: {e}")

    analyst_note = None
    today = datetime.utcnow().date()
    cached = ANALYST_NOTE_CACHE.get(symbol)
    if cached and cached[0] == today:
        analyst_note = cached[1]
    else:
        headlines: list[str] = []
        twits_summary: dict = {}
        twits_headlines: list[str] = []
        try:
            finnhub = FinnhubProvider()
            news_items = await finnhub.get_company_news(symbol, days=7)
            headlines = [
                item.get("headline")
                for item in news_items
                if isinstance(item, dict) and item.get("headline")
            ]
        except Exception:
            headlines = []
        try:
            stocktwits = StocktwitsProvider()
            messages = await stocktwits.get_symbol_stream(symbol, limit=20)
            bullish = 0
            bearish = 0
            total = 0
            for message in messages:
                if not isinstance(message, dict):
                    continue
                total += 1
                body = message.get("body")
                if isinstance(body, str) and body.strip():
                    twits_headlines.append(body.strip())
                sentiment = message.get("entities", {}).get("sentiment", {})
                if isinstance(sentiment, dict):
                    label = sentiment.get("basic")
                    if label == "Bullish":
                        bullish += 1
                    elif label == "Bearish":
                        bearish += 1
            if total:
                twits_summary = {
                    "bullish": bullish,
                    "bearish": bearish,
                    "total": total,
                }
        except Exception:
            twits_summary = {}
            twits_headlines = []

        analyst_note = build_analyst_note(
            symbol,
            fundamentals,
            {
                "rsi": rsi,
                "ma50": ma50,
                "ma200": ma200,
            },
            headlines,
            twits_summary,
            twits_headlines,
        )
        ANALYST_NOTE_CACHE[symbol] = (today, analyst_note)

    return JSONResponse(
        {
            "symbol": symbol,
            "fundamentals": fundamentals,
            "technicals": {
                "rsi": rsi,
                "macd": macd,
                "macdSignal": signal,
                "ma50": ma50,
                "ma200": ma200,
            },
            "analystNote": analyst_note,
        }
    )
