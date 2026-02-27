#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/bootstrap_new_project.sh <project_name>"
  exit 1
fi

PROJECT_NAME="$1"

if [[ -f README.md ]]; then
  sed -i "s/^# .*/# ${PROJECT_NAME}/" README.md || true
fi

if [[ -f .env.example ]]; then
  cp .env.example .env
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

if [[ -f .gitignore ]]; then
  grep -q '^\.env$' .gitignore || echo '.env' >> .gitignore
  grep -q '^\.venv/$' .gitignore || echo '.venv/' >> .gitignore
fi

echo "Bootstrap complete for ${PROJECT_NAME}."
echo "Next:"
echo "  1) source .venv/bin/activate"
echo "  2) uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
