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
