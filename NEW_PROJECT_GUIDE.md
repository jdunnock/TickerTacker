# New Project Guide (FastAPI + Postgres + CI/CD + Railway)

Tämän oppaan tavoite: saat puhtaan projektirungon nopeasti tuotantokuntoon ilman TickerTackerin liiketoimintalogiikkaa.

## 1) Tavoitearkkitehtuuri

- Backend: FastAPI
- DB: PostgreSQL
- Migrations: Alembic
- Local dev: `.venv` + optional Docker Postgres
- CI: GitHub Actions (lint + tests)
- CD: Railway (auto deploy `main` branchista)

## 2) Luo uusi repository

Mac terminal:

```bash
mkdir my-new-project && cd my-new-project
git init
```

## 3) Perusrakenne

Luo kansiot ja tiedostot:

```text
my-new-project/
  app/
    __init__.py
    main.py
    db.py
    models.py
  alembic/
  alembic.ini
  tests/
    test_health.py
  .gitignore
  .env.example
  requirements.txt
  Procfile
  docker-compose.yml
  README.md
  .github/workflows/ci.yml
```

## 4) Dependencies

`requirements.txt`:

```txt
fastapi
uvicorn[standard]
sqlalchemy
alembic
psycopg[binary]
python-dotenv
pytest
httpx
ruff
```

## 5) Minimi backend

`app/db.py`:

```python
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dev.db")
engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
```

`app/main.py`:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}
```

`tests/test_health.py`:

```python
from fastapi.testclient import TestClient
from app.main import app


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
```

## 6) Ympäristö ja local run

` .env.example`:

```env
DATABASE_URL=postgresql+psycopg://app:app@localhost:5432/app
```

Mac terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 7) Optional local Postgres Dockerilla

`docker-compose.yml`:

```yaml
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_DB: app
      POSTGRES_USER: app
      POSTGRES_PASSWORD: app
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

Aja:

```bash
docker compose up -d db
```

## 8) Alembic bootstrap

Mac terminal:

```bash
alembic init alembic
```

Aseta `alembic/env.py` käyttämään `DATABASE_URL` ympäristömuuttujaa.
Luo ensimmäinen migration:

```bash
alembic revision -m "initial"
alembic upgrade head
```

## 9) GitHub CI

`.github/workflows/ci.yml`:

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: python -m pip install --upgrade pip
      - run: pip install -r requirements.txt
      - run: ruff check .
      - run: pytest -q
```

## 10) Railway CD

1. Luo Railway project
2. Connect GitHub repo
3. Lisää ympäristömuuttujat Railwayhin:
   - `DATABASE_URL` (Railway Postgres antaa tämän)
4. Aseta Start Command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

5. (Suositus) migration deploy-vaiheessa:
   - joko erillinen migration job
   - tai startup-hook joka ajaa `alembic upgrade head`

## 11) Procfile (fallback)

`Procfile`:

```Procfile
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## 12) Ensimmäinen julkaisu-checklist

- [ ] `pytest` menee läpi lokaalisti
- [ ] `ruff check .` puhdas
- [ ] `/health` toimii lokaalisti
- [ ] Railway deploy onnistuu
- [ ] Railway URL `/health` palauttaa 200

## 13) Yksinkertainen työjärjestys jatkoon

- Feature branch
- PR -> CI green
- Merge `main`
- Railway auto deploy
- Smoke test `/health`

---

Jos haluat, seuraava askel on tehdä tästä "template-repo" (GitHub template), jolloin uusi projekti syntyy yhdellä klikillä näillä asetuksilla.

## 14) Template-repo valmiina tässä workspace:ssa

Lisäsin valmiit template-aputiedostot:

- `TEMPLATE_REPO_CHECKLIST.md`
- `scripts/bootstrap_new_project.sh`
- `.github/workflows/ci-template.yml`

Nopea käyttö uuden projektin alussa:

```bash
bash scripts/bootstrap_new_project.sh my_new_project
```
