# Template Repo Checklist (FastAPI + Postgres + CI/CD + Railway)

Käytä tätä, kun haluat tehdä tästä repossa olevan rungon oikeaksi GitHub templateksi.

## 1) Pakolliset tiedostot templateen

- `app/main.py` (minimi `/health`)
- `app/db.py`
- `requirements.txt`
- `README.md`
- `.env.example`
- `.gitignore`
- `Procfile`
- `.github/workflows/ci.yml`
- `NEW_PROJECT_GUIDE.md`

## 2) Suositellut lisät

- `docker-compose.yml` (vain local Postgres)
- `alembic/` + `alembic.ini`
- `tests/test_health.py`
- `.devcontainer/devcontainer.json`

## 3) Mitä EI kuulu templateen

- tuotantodatan dumpit
- `.env` oikeilla salaisuuksilla
- projektikohtainen business-logiikka
- vanhat migrationit joita ei tarvita rungossa

## 4) GitHub Template -asetukset

1. Pushaa repo GitHubiin
2. GitHub repo -> Settings -> General
3. Ruksaa `Template repository`

## 5) Uuden projektin luonti template-reposta

1. GitHubissa: `Use this template`
2. Anna uusi nimi
3. Clone lokaalisti
4. Aja bootstrap:
   - `bash scripts/bootstrap_new_project.sh your_project_name`

## 6) Railway-ready checklist

- [ ] `ci.yml` toimii PR:ssä
- [ ] `DATABASE_URL` env var käytössä
- [ ] Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- [ ] `/health` palauttaa 200 deployn jälkeen
