#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/srv/algotrade}"
VENV_DIR="${VENV_DIR:-$APP_DIR/.venv}"
SERVICE_NAME="${SERVICE_NAME:-algotrade-api}"
BRANCH="${BRANCH:-production}"

cd "$APP_DIR"

git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r requirements.txt

if [ -f "$VENV_DIR/bin/alembic" ]; then
  "$VENV_DIR/bin/alembic" upgrade head
fi

if systemctl is-active --quiet "$SERVICE_NAME"; then
  sudo systemctl reload "$SERVICE_NAME"
else
  sudo systemctl start "$SERVICE_NAME"
fi
