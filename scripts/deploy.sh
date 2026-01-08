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

git config --global --add safe.directory "$APP_DIR"

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

SUDO_CMD=""
if command -v sudo >/dev/null 2>&1; then
  if sudo -n true >/dev/null 2>&1; then
    SUDO_CMD="sudo -n"
  fi
fi

reload_service() {
  if [ -n "$SUDO_CMD" ]; then
    $SUDO_CMD systemctl reload "$SERVICE_NAME"
    return
  fi
  if systemctl --user list-unit-files 2>/dev/null | grep -q "^${SERVICE_NAME}\\.service"; then
    systemctl --user reload "$SERVICE_NAME"
    return
  fi
  echo "No sudo permission to reload ${SERVICE_NAME}. Configure passwordless sudo or a user service." >&2
  exit 1
}

start_service() {
  if [ -n "$SUDO_CMD" ]; then
    $SUDO_CMD systemctl start "$SERVICE_NAME"
    return
  fi
  if systemctl --user list-unit-files 2>/dev/null | grep -q "^${SERVICE_NAME}\\.service"; then
    systemctl --user start "$SERVICE_NAME"
    return
  fi
  echo "No sudo permission to start ${SERVICE_NAME}. Configure passwordless sudo or a user service." >&2
  exit 1
}

if systemctl is-active --quiet "$SERVICE_NAME"; then
  reload_service
else
  start_service
fi
