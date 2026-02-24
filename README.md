# TickerTacker

Quick start (local, Postgres):

1. Copy env file:
   ```
   cp .env.local .env
   ```
   (`.env.local` is for local Postgres. `.env.docker` is for Docker.)
2. Create `.env` manually if you prefer:
   ```
   DATABASE_URL=postgresql+psycopg://app:app@localhost:5432/tickers
   ```
3. Install deps:
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. Run app:
   ```
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

Railway deploy notes:

- Set `DATABASE_URL` in Railway to your Postgres connection string.
- Ensure your start command is `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
- If using Finnhub, set `FINNHUB_API_KEY` as an environment variable.

Docker note (dev container):

- Docker daemon does not run inside this dev container due to missing kernel privileges.
- Use Docker Desktop on your host machine, or run locally without Docker.
