#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/srv/algotrade}"
VENV_DIR="${VENV_DIR:-$APP_DIR/.venv}"
SERVICE_NAME="${SERVICE_NAME:-algotrade-api}"
BRANCH="${BRANCH:-production}"
REPO_URL="${REPO_URL:-}"

if [ ! -d "$APP_DIR/.git" ]; then
  if [ -z "$REPO_URL" ]; then
    echo "REPO_URL is required for first-time clone." >&2
    exit 1
  fi
  if [ -d "$APP_DIR" ] && [ -n "$(ls -A "$APP_DIR" 2>/dev/null)" ]; then
    echo "APP_DIR exists but is not a git repo: $APP_DIR" >&2
    exit 1
  fi
  mkdir -p "$APP_DIR"
  git clone -b "$BRANCH" "$REPO_URL" "$APP_DIR"
fi

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
