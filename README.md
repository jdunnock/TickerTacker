# TickerTacker

Quick start (local, Postgres):

1. Create `.env` with:
   ```
   DATABASE_URL=postgresql+psycopg://app:app@localhost:5432/tickers
   ```
2. Install deps:
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run app:
   ```
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

Railway deploy notes:

- Set `DATABASE_URL` in Railway to your Postgres connection string.
- Ensure your start command is `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
- If using Finnhub, set `FINNHUB_API_KEY` as an environment variable.
