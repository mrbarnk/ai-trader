#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/srv/algotrade}"
VENV_DIR="${VENV_DIR:-$APP_DIR/.venv}"
SERVICE_NAME="${SERVICE_NAME:-algotrade-api}"
WORKER_SERVICE_NAME="${WORKER_SERVICE_NAME:-${SERVICE_NAME}-worker}"
BACKTEST_WORKER_SERVICE_NAME="${BACKTEST_WORKER_SERVICE_NAME:-${SERVICE_NAME}-backtest-worker}"
SERVICE_USER="${SERVICE_USER:-$(whoami)}"
SERVICE_GROUP="${SERVICE_GROUP:-$(id -gn)}"
APP_PORT="${APP_PORT:-5100}"
ENV_FILE="${ENV_FILE:-$APP_DIR/.env}"
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
"$VENV_DIR/bin/pip" install --upgrade --force-reinstall -r requirements.txt

if [ -f "$VENV_DIR/bin/alembic" ]; then
  "$VENV_DIR/bin/alembic" upgrade head
fi

SUDO_CMD=""
SERVICE_SCOPE="system"
if command -v sudo >/dev/null 2>&1; then
  if sudo -n true >/dev/null 2>&1; then
    SUDO_CMD="sudo -n"
  fi
fi
if [ -z "$SUDO_CMD" ]; then
  SERVICE_SCOPE="user"
fi

systemctl_cmd() {
  if [ "$SERVICE_SCOPE" = "system" ]; then
    $SUDO_CMD systemctl "$@"
  else
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
    systemctl --user "$@"
  fi
}

SERVICE_DIR="/etc/systemd/system"
WANTED_BY="multi-user.target"
if [ "$SERVICE_SCOPE" = "user" ]; then
  SERVICE_DIR="$HOME/.config/systemd/user"
  WANTED_BY="default.target"
  mkdir -p "$SERVICE_DIR"
fi

write_service() {
  local name="$1"
  local exec_start="$2"
  local unit_path="$SERVICE_DIR/$name.service"
  if [ "$SERVICE_SCOPE" = "system" ]; then
    cat <<EOF | $SUDO_CMD tee "$unit_path" >/dev/null
[Unit]
Description=$name
After=network.target

[Service]
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$APP_DIR
EnvironmentFile=-$ENV_FILE
Environment="REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt"
Environment="SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt"
ExecStart=$exec_start
Restart=always
RestartSec=3
TimeoutStopSec=30

[Install]
WantedBy=$WANTED_BY
EOF
  else
    cat <<EOF > "$unit_path"
[Unit]
Description=$name
After=network.target

[Service]
WorkingDirectory=$APP_DIR
EnvironmentFile=-$ENV_FILE
Environment="REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt"
Environment="SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt"
ExecStart=$exec_start
Restart=always
RestartSec=3
TimeoutStopSec=30

[Install]
WantedBy=$WANTED_BY
EOF
  fi
}

API_EXEC="$VENV_DIR/bin/gunicorn --worker-class eventlet --workers 1 --bind 0.0.0.0:$APP_PORT app:app"
WORKER_EXEC="$VENV_DIR/bin/python -m server.metaapi_streaming_worker"
BACKTEST_WORKER_EXEC="$VENV_DIR/bin/python -m server.backtest_worker"

write_service "$SERVICE_NAME" "$API_EXEC"
write_service "$WORKER_SERVICE_NAME" "$WORKER_EXEC"
write_service "$BACKTEST_WORKER_SERVICE_NAME" "$BACKTEST_WORKER_EXEC"

systemctl_cmd daemon-reload
systemctl_cmd enable "$SERVICE_NAME"
systemctl_cmd enable "$WORKER_SERVICE_NAME"
systemctl_cmd enable "$BACKTEST_WORKER_SERVICE_NAME"
systemctl_cmd restart "$SERVICE_NAME"
systemctl_cmd restart "$WORKER_SERVICE_NAME"
systemctl_cmd restart "$BACKTEST_WORKER_SERVICE_NAME"
