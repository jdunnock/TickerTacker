# TickerTacker Specification

Reverse-engineered specification of the TickerTacker application based on codebase analysis.

## Overview

**TickerTacker** is a web-based stock/ticker watchlist application that aggregates real-time price data, fundamental analysis, and technical metrics from multiple sources.

**Stack:** FastAPI (Python) + PostgreSQL + HTML/CSS + JavaScript + HTMX  
**Deployment:** Docker + Railway

---

## Core Features

### 1. User Management

- **Register & Login:** Email-based authentication with bcrypt password hashing.
- **Sessions:** 30-day cookie-based sessions stored in `user_sessions` table.
- **Auth Dependency:** `get_current_user()` extracts user from session cookie.

### 2. Watchlist Management

- Users can add/remove instruments (tickers) to/from a personal watchlist.
- Each watchlist entry tracks which user owns it.
- **Clear Watchlist:** Bulk delete all user's watchlist items.

### 3. Price Data & History

- Store historical OHLCV data (Open, High, Low, Close, Volume) per instrument.
- Multiple data sources:
  - **YFinance**: Primary source for historical prices.
  - **Finnhub**: Company fundamentals (P/E, P/S, EV/EBITDA, Market Cap, etc.).
  - **Stocktwits**: Public sentiment (bullish/bearish).
  - **Stooq**: Alternative data source (referenced, may be optional).

### 4. Real-time Quotes

- Cache quotes for 45 seconds before re-fetching.
- Display last price, daily change (%), High/Low for selected range.
- Auto-refresh UI every 15 seconds via HTMX post.

### 5. Technical Analysis

- **Indicators:**
  - RSI (14-period).
  - MACD.
  - Moving Averages (50/200).
- Trend classification based on MA crossover & RSI.
- Chart rendering (SVG, 640x280px) supporting ranges: 1D, YTD, 1M, 6M, 1Y.

### 6. Fundamental Analysis

- Display P/E, P/S, EV/EBITDA, EBITDA, Market Cap from Finnhub.
- Score ratios (good/neutral/bad) relative to sector benchmarks.

### 7. Analyst Notes

- Auto-generated summary combining:
  - Valuation sentiment (P/E, P/S, EV ratios).
  - Technicals (RSI, trend, MA signals).
  - Sentiment (Stocktwits bullish/bearish %).
  - Sector-relative context.
- Cached daily per instrument.

---

## Database Schema

### Users
- `id` (UUID)
- `email` (unique)
- `password_hash`
- `created_at`

### UserSessions
- `id` (random string, 64 chars)
- `user_id` (FK → Users)
- `expires_at`
- `created_at`

### Instruments
- `id` (int, PK)
- `symbol` (unique ticker, e.g., "AAPL")
- `name` (company name)
- `exchange` (e.g., "NASDAQ")
- `created_at`

### Prices
- `id` (int, PK)
- `instrument_id` (FK → Instruments)
- `timestamp`
- `open`, `high`, `low`, `close` (Numeric)
- `volume` (BigInt, nullable)

### Watchlist
- `id` (int, PK)
- `instrument_id` (FK → Instruments)
- `owner` (user email or ID)
- `added_at`

---

## API Routes (FastAPI)

### Auth
- `GET /` → Redirect to login (if not authenticated) or dashboard.
- `GET /register` → Registration form (HTML).
- `POST /register` → Create user, start session.
- `GET /login` → Login form (HTML).
- `POST /login` → Authenticate, set session cookie.
- `POST /logout` → Invalidate session.

### Watchlist Pages
- `GET /dashboard` → Show user's watchlist with live quote updates.
- `GET /add-instruments` → Form to search & add new instruments.
- `GET /detail` → Detailed view of single ticker (chart, technicals, fundamentals).

### Data APIs
- `POST /api/prices/refresh` → HTMX endpoint to refresh watchlist prices.
- `POST /api/watchlist/clear` → Clear all items for current user.
- `GET /api/instruments/search?q=...` → Search instruments by symbol/name.
- `POST /api/watchlist/add` → Add instrument to watchlist.
- `POST /api/watchlist/remove/:id` → Remove watchlist entry.
- `GET /api/prices/history?symbol=...&range=...` → Fetch historical prices + technicals.
- `GET /api/fundamentals?symbol=...` → Fetch fundamental data (Finnhub).
- `GET /api/sentiment?symbol=...` → Fetch sentiment (Stocktwits).

---

## UI & Templates

### base.html
- Header with branding, nav links, auth actions.
- Fixed sticky header with backdrop blur.
- Main content area (responsive layout).
- Custom CSS with gradients, glass-morphism effects.

### dashboard.html
- Watchlist table (symbol, exchange, last, close, 1D change).
- Live price refresh (HTMX, every 15s).
- Add/Clear buttons.
- Deploy check banner (Build timestamp).
- Responsive table (collapses 1-column on mobile).

### detail.html
- Ticker selector (dropdown from user's watchlist).
- Chart panel (SVG, range buttons: 1D/YTD/1M/6M/1Y).
- Metrics row (Last, Change, High, Low, Volume).
- Analysis grid (Fundamentals: P/E, P/S, EV/EBITDA, etc.; Technicals: RSI, MACD, MA, Trend).
- Analyst note (auto-generated summary).
- Interactive chart with hover tooltip.

### login.html, register.html
- Basic forms for authentication.

---

## Caching Strategy

- **Quote Cache:** 45 seconds (symbol → (timestamp, last_price)).
- **Last Display:** Tracks last price shown per symbol (for UI diffing).
- **Analyst Note Cache:** Daily cache per instrument.

---

## Configuration & Env

- `DATABASE_URL`: Postgres connection string.
  - Local: `postgresql://app:app@localhost:5432/tickers`
  - Docker: `postgresql://app:app@db:5432/tickers`
- `FINNHUB_API_KEY`: API key for Finnhub (optional, fundamentals won't load without it).

---

## Tech Debt / Notes

- Points metric on detail page: removed (no value).
- Docker daemon does not run in dev container (kernel privilege limitation).
- Use Docker Desktop on host machine for full stack testing.
- Railway deployment supports auto-reload on push.

---

## Entry Point

- `app/main.py`: FastAPI app creation, routes, caching logic.
- `app/models.py`: SQLAlchemy ORM models.
- `app/db.py`: Database engine and session.
- `app/providers/`: Data provider classes (yfinance, finnhub, stocktwits).
- `app/services/pricing.py`: Business logic for pricing calculations.
- `app/utils.py`: Password hashing/verification utilities.

---

## Future Enhancements

- Alerts on price thresholds.
- Portfolio grouping beyond watchlist.
- Export to CSV.
- Custom sector mappings.
- Mobile app (React Native).
