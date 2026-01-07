from __future__ import annotations

import hmac
import json
import os
import secrets
import threading
import sys
import tempfile
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

from flask import Flask, Response, Request, request
from werkzeug.security import check_password_hash, generate_password_hash
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from werkzeug.exceptions import HTTPException

BASE_STYLE = """
    :root {
      --bg-1: #0f172a;
      --bg-2: #111827;
      --bg-3: #0b1220;
      --panel: rgba(255, 255, 255, 0.06);
      --panel-strong: rgba(255, 255, 255, 0.12);
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-2: #f97316;
      --border: rgba(148, 163, 184, 0.25);
      --shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: "Georgia", "Times New Roman", serif;
      color: var(--text);
      min-height: 100vh;
      background:
        radial-gradient(1200px 700px at 10% 10%, rgba(56, 189, 248, 0.15), transparent),
        radial-gradient(900px 600px at 90% 20%, rgba(249, 115, 22, 0.12), transparent),
        linear-gradient(160deg, var(--bg-1), var(--bg-2) 60%, var(--bg-3));
    }

    .wrap {
      max-width: 720px;
      margin: 0 auto;
      padding: 36px 20px 48px;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(6px);
    }

    h1, h2 {
      margin: 0 0 8px;
      letter-spacing: 0.4px;
    }

    p {
      color: var(--muted);
      margin: 6px 0 0;
    }

    .meta {
      display: grid;
      gap: 6px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }

    .upload {
      margin-top: 16px;
      display: grid;
      gap: 12px;
    }

    input[type="file"] {
      color: var(--text);
      background: rgba(15, 23, 42, 0.35);
      border: 1px solid var(--border);
      padding: 10px;
      border-radius: 10px;
    }

    button {
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      border: none;
      color: #0b1220;
      font-weight: 700;
      padding: 10px 14px;
      border-radius: 10px;
      cursor: pointer;
    }

    a {
      color: var(--accent);
      text-decoration: none;
    }
"""


BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

from trader.backtest import load_csv_candles, run_backtest  # noqa: E402
from trader import config  # noqa: E402


APP_USERNAME = os.getenv("APP_USERNAME")
APP_PASSWORD = os.getenv("APP_PASSWORD")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///app.db")

MAX_CSV_BYTES = 10 * 1024 * 1024
MAX_CANDLES = 100_000
MAX_RANGE_DAYS = 93

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CSV_BYTES

VIEWER_HTML = (BASE_DIR / "backtest_viewer.html").read_text(encoding="utf-8")

DEFAULT_USER_CONFIG: dict[str, Any] = {
    "model_mode": "aggressive",
    "tp_leg_source": "4H",
    "tp1_leg_percent": 0.5,
    "tp2_leg_percent": 0.9,
    "tp3_enabled": False,
    "tp3_leg_source": "4H",
    "tp3_leg_percent": 1.0,
    "sl_extra_pips": 3.0,
    "enable_break_even": True,
    "use_real_balance": False,
    "starting_balance": 10000.0,
    "risk_per_trade_pct": 1.0,
    "use_1m_entry": False,
    "metaapi_account_id": "",
    "metaapi_token": "",
}

CONFIG_FLOAT_KEYS = {
    "tp1_leg_percent",
    "tp2_leg_percent",
    "tp3_leg_percent",
    "sl_extra_pips",
    "starting_balance",
    "risk_per_trade_pct",
}
CONFIG_BOOL_KEYS = {"tp3_enabled", "enable_break_even", "use_real_balance", "use_1m_entry"}
CONFIG_STRING_KEYS = {
    "model_mode",
    "tp_leg_source",
    "tp3_leg_source",
    "metaapi_account_id",
    "metaapi_token",
}
ALLOWED_CONFIG_KEYS = CONFIG_FLOAT_KEYS | CONFIG_BOOL_KEYS | CONFIG_STRING_KEYS
CONFIG_LOCK = threading.Lock()


def _normalize_database_url() -> tuple[str, dict[str, Any]]:
    url = DATABASE_URL
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    connect_args: dict[str, Any] = {}
    if url.startswith("sqlite:///"):
        connect_args = {"check_same_thread": False}
    return url, connect_args


_db_url, _db_connect_args = _normalize_database_url()
engine = create_engine(_db_url, connect_args=_db_connect_args, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    tokens = relationship("ApiToken", cascade="all, delete-orphan", back_populates="user")
    configs = relationship("UserConfig", cascade="all, delete-orphan", back_populates="user")


class ApiToken(Base):
    __tablename__ = "api_tokens"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    user = relationship("User", back_populates="tokens")


class UserConfig(Base):
    __tablename__ = "user_configs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    config_json = Column(Text, nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    user = relationship("User", back_populates="configs")


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


init_db()


@contextmanager
def db_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _check_auth(req: Request) -> bool:
    if not APP_USERNAME or not APP_PASSWORD:
        return False
    auth = req.authorization
    if auth is None:
        return False
    return hmac.compare_digest(auth.username, APP_USERNAME) and hmac.compare_digest(
        auth.password, APP_PASSWORD
    )


def _require_auth() -> Response | None:
    if _check_auth(request):
        return None
    response = _render_error("Authentication required.", 401)
    response.headers["WWW-Authenticate"] = 'Basic realm="Backtest Login"'
    return response


@app.before_request
def enforce_basic_auth() -> Response | None:
    if request.path.startswith("/api/") or request.path == "/health":
        return None
    return _require_auth()


def _render_error(message: str, status_code: int = 400) -> Response:
    html = f"""
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>Backtest Error</title>
        <style>{BASE_STYLE}</style>
      </head>
      <body>
        <div class="wrap">
          <div class="panel">
            <h2>Upload Error</h2>
            <p>{message}</p>
            <p><a href="/">Back to upload</a></p>
          </div>
        </div>
      </body>
    </html>
    """
    return Response(html, status=status_code, mimetype="text/html")


@app.errorhandler(Exception)
def handle_exception(_err: Exception) -> Response:
    if isinstance(_err, HTTPException):
        message = _err.description or "Request failed."
        if request.path.startswith("/api/"):
            return _json_error(message, _err.code or 400)
        return _render_error(message, _err.code or 400)
    if request.path.startswith("/api/"):
        return _json_error("Something went wrong. Please try again.", 500)
    return _render_error("Something went wrong. Please try again.", 500)


def _json_response(payload: dict[str, Any], status_code: int = 200) -> Response:
    return Response(json.dumps(payload), status=status_code, mimetype="application/json")


def _json_error(message: str, status_code: int = 400) -> Response:
    return _json_response({"error": message}, status_code)


def _bearer_token() -> str | None:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    return auth.split(" ", 1)[1].strip() or None


def _require_token(session) -> User | None:
    token = _bearer_token()
    if not token:
        return None
    token_row = session.query(ApiToken).filter_by(token=token).first()
    if not token_row:
        return None
    return token_row.user


def _sanitize_config(data: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in data.items():
        if key not in ALLOWED_CONFIG_KEYS:
            continue
        if key in CONFIG_FLOAT_KEYS:
            try:
                sanitized[key] = float(value)
            except (TypeError, ValueError):
                continue
        elif key in CONFIG_BOOL_KEYS:
            sanitized[key] = bool(value)
        else:
            sanitized[key] = str(value).strip()
    return sanitized


def _load_user_config(session, user: User) -> dict[str, Any]:
    existing = session.query(UserConfig).filter_by(user_id=user.id).first()
    if not existing:
        return DEFAULT_USER_CONFIG.copy()
    try:
        stored = json.loads(existing.config_json)
    except json.JSONDecodeError:
        stored = {}
    merged = DEFAULT_USER_CONFIG.copy()
    merged.update({k: v for k, v in stored.items() if k in ALLOWED_CONFIG_KEYS})
    return merged


def _save_user_config(session, user: User, config_data: dict[str, Any]) -> dict[str, Any]:
    merged = _load_user_config(session, user)
    merged.update(config_data)
    existing = session.query(UserConfig).filter_by(user_id=user.id).first()
    payload = json.dumps(merged)
    if existing:
        existing.config_json = payload
    else:
        session.add(UserConfig(user_id=user.id, config_json=payload))
    return merged


def _build_config_overrides(user_config: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "MODEL_MODE": user_config.get("model_mode", config.MODEL_MODE),
        "TP_LEG_SOURCE": user_config.get("tp_leg_source", config.TP_LEG_SOURCE),
        "TP1_LEG_PERCENT": user_config.get("tp1_leg_percent", config.TP1_LEG_PERCENT),
        "TP2_LEG_PERCENT": user_config.get("tp2_leg_percent", config.TP2_LEG_PERCENT),
        "TP3_ENABLED": user_config.get("tp3_enabled", config.TP3_ENABLED),
        "TP3_LEG_SOURCE": user_config.get("tp3_leg_source", config.TP3_LEG_SOURCE),
        "TP3_LEG_PERCENT": user_config.get("tp3_leg_percent", config.TP3_LEG_PERCENT),
        "SL_EXTRA_PIPS": user_config.get("sl_extra_pips", config.SL_EXTRA_PIPS),
        "ENABLE_BREAK_EVEN": user_config.get("enable_break_even", config.ENABLE_BREAK_EVEN),
        "RISK_PER_TRADE_PCT": user_config.get("risk_per_trade_pct", config.RISK_PER_TRADE_PCT),
        "USE_1M_ENTRY": user_config.get("use_1m_entry", config.USE_1M_ENTRY),
        "ACCOUNT_BALANCE_OVERRIDE": user_config.get(
            "starting_balance", config.ACCOUNT_BALANCE_OVERRIDE
        ),
    }
    return overrides


@contextmanager
def _apply_config_overrides(overrides: dict[str, Any]):
    with CONFIG_LOCK:
        original = {key: getattr(config, key) for key in overrides.keys()}
        for key, value in overrides.items():
            setattr(config, key, value)
        try:
            yield
        finally:
            for key, value in original.items():
                setattr(config, key, value)


def _load_rows(jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _render_viewer(rows: list[dict[str, Any]]) -> Response:
    payload = json.dumps(rows)
    injection = (
        f"<script>window.PRELOADED_ROWS = {payload};"
        "window.__applyPreloadedRows && window.__applyPreloadedRows(window.PRELOADED_ROWS);"
        "</script>"
    )
    html = VIEWER_HTML.replace("</body>", f"{injection}\n</body>")
    return Response(html, mimetype="text/html")


def _validate_candles(csv_path: Path) -> str | None:
    candles = load_csv_candles(csv_path)
    if not candles:
        return "CSV is empty after parsing."
    if len(candles) > MAX_CANDLES:
        return f"CSV has {len(candles)} candles; max allowed is {MAX_CANDLES}."
    span = candles[-1].time_utc - candles[0].time_utc
    if span > timedelta(days=MAX_RANGE_DAYS):
        return f"CSV spans {span.days} days; max allowed is {MAX_RANGE_DAYS}."
    return None


@app.route("/", methods=["GET"])
def index() -> Response:
    html = f"""
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <title>GU Backtest Upload</title>
        <style>{BASE_STYLE}</style>
      </head>
      <body>
        <div class="wrap">
          <div class="panel">
            <h1>GU Backtest Upload</h1>
            <p>Upload 1M candles and get the summary + viewer in seconds.</p>
            <div class="upload">
              <form action="/run" method="post" enctype="multipart/form-data">
                <input type="file" name="csv" accept=".csv,.txt" required />
                <button type="submit">Run Backtest</button>
              </form>
            </div>
            <div class="meta">
              <div>Max file size: 10 MB</div>
              <div>Max range: 1 month</div>
              <div>Files deleted immediately after processing</div>
            </div>
          </div>
        </div>
      </body>
    </html>
    """
    return Response(html, mimetype="text/html")


@app.route("/health", methods=["GET"])
def health() -> Response:
    return Response("ok", mimetype="text/plain")


@app.route("/run", methods=["POST"])
def run() -> Response:
    upload = request.files.get("csv")
    if not upload or upload.filename == "":
        return _render_error("No CSV file uploaded.")

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "upload.csv"
        upload.save(csv_path)

        error = _validate_candles(csv_path)
        if error:
            return _render_error(error)

        output_path = Path(tmpdir) / "signals.jsonl"

        original_log_setting = config.LOG_DAILY_LIMITS
        config.LOG_DAILY_LIMITS = False
        try:
            run_backtest(csv_path, source_minutes=1, output_path=output_path, start=None, end=None)
        finally:
            config.LOG_DAILY_LIMITS = original_log_setting

        outcome_path = output_path.with_name(output_path.stem + "_outcomes.jsonl")
        rows = _load_rows(outcome_path)

    return _render_viewer(rows)


@app.route("/api/backtest", methods=["POST"])
def api_backtest() -> Response:
    with db_session() as session:
        user = _require_token(session)
        if not user:
            return _json_error("Unauthorized.", 401)

        upload = request.files.get("csv")
        if not upload or upload.filename == "":
            return _json_error("No CSV file uploaded.", 400)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "upload.csv"
            upload.save(csv_path)

            error = _validate_candles(csv_path)
            if error:
                return _json_error(error, 400)

            output_path = Path(tmpdir) / "signals.jsonl"
            outcome_path = output_path.with_name(output_path.stem + "_outcomes.jsonl")
            user_config = _load_user_config(session, user)
            overrides = _build_config_overrides(user_config)

            original_log_setting = config.LOG_DAILY_LIMITS
            config.LOG_DAILY_LIMITS = False
            try:
                with _apply_config_overrides(overrides):
                    run_backtest(
                        csv_path,
                        source_minutes=1,
                        output_path=output_path,
                        start=None,
                        end=None,
                        model_mode=overrides.get("MODEL_MODE"),
                    )
            finally:
                config.LOG_DAILY_LIMITS = original_log_setting

            rows = _load_rows(outcome_path)

        return _json_response(
            {
                "user": {"id": user.id, "email": user.email},
                "model_mode": overrides.get("MODEL_MODE"),
                "rows": rows,
            }
        )


@app.route("/api/signup", methods=["POST"])
def api_signup() -> Response:
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", "")).strip()
    if not email or not password:
        return _json_error("Email and password are required.", 400)
    with db_session() as session:
        if session.query(User).filter_by(email=email).first():
            return _json_error("Email already registered.", 409)
        user = User(email=email, password_hash=generate_password_hash(password))
        session.add(user)
        session.flush()
        token_value = secrets.token_urlsafe(32)
        token_row = ApiToken(user_id=user.id, token=token_value)
        session.add(token_row)
        _save_user_config(session, user, {})
        return _json_response(
            {"token": token_value, "user": {"id": user.id, "email": user.email}}, 201
        )


@app.route("/api/login", methods=["POST"])
def api_login() -> Response:
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", "")).strip()
    if not email or not password:
        return _json_error("Email and password are required.", 400)
    with db_session() as session:
        user = session.query(User).filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            return _json_error("Invalid credentials.", 401)
        token_value = secrets.token_urlsafe(32)
        session.add(ApiToken(user_id=user.id, token=token_value))
        return _json_response(
            {"token": token_value, "user": {"id": user.id, "email": user.email}}
        )


@app.route("/api/logout", methods=["POST"])
def api_logout() -> Response:
    with db_session() as session:
        token = _bearer_token()
        if not token:
            return _json_error("Missing bearer token.", 401)
        token_row = session.query(ApiToken).filter_by(token=token).first()
        if not token_row:
            return _json_error("Invalid token.", 401)
        session.delete(token_row)
        return _json_response({"ok": True})


@app.route("/api/me", methods=["GET"])
def api_me() -> Response:
    with db_session() as session:
        user = _require_token(session)
        if not user:
            return _json_error("Unauthorized.", 401)
        return _json_response({"id": user.id, "email": user.email})


@app.route("/api/config", methods=["GET", "PUT"])
def api_config() -> Response:
    with db_session() as session:
        user = _require_token(session)
        if not user:
            return _json_error("Unauthorized.", 401)
        if request.method == "GET":
            return _json_response({"config": _load_user_config(session, user)})
        payload = request.get_json(silent=True) or {}
        sanitized = _sanitize_config(payload)
        if not sanitized:
            return _json_error("No valid config fields provided.", 400)
        updated = _save_user_config(session, user, sanitized)
        return _json_response({"config": updated})


@app.route("/api/docs", methods=["GET"])
def api_docs() -> Response:
    html = f"""
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <title>API Docs</title>
        <style>{BASE_STYLE}</style>
      </head>
      <body>
        <div class="wrap">
          <div class="panel">
            <h1>API Docs</h1>
            <p>All endpoints return JSON. Use the Bearer token from /api/login or /api/signup.</p>
            <div class="meta">
              <div>POST /api/signup</div>
              <div>POST /api/login</div>
              <div>POST /api/logout</div>
              <div>GET /api/me</div>
              <div>GET /api/config</div>
              <div>PUT /api/config</div>
              <div>POST /api/backtest</div>
            </div>
          </div>
          <div class="panel" style="margin-top: 16px;">
            <h2>Signup</h2>
            <pre><code>{"{ \"email\": \"you@example.com\", \"password\": \"secret\" }"}</code></pre>
            <h2>Login</h2>
            <pre><code>{"{ \"email\": \"you@example.com\", \"password\": \"secret\" }"}</code></pre>
            <h2>Config Update</h2>
            <pre><code>{"{ \"tp1_leg_percent\": 0.5, \"tp2_leg_percent\": 0.8, \"sl_extra_pips\": 3 }"}</code></pre>
            <h2>Backtest</h2>
            <p>Multipart form with a CSV file (field name: <code>csv</code>).</p>
            <h2>Config Keys</h2>
            <p>Allowed keys:</p>
            <pre><code>tp_leg_source, tp1_leg_percent, tp2_leg_percent,
tp3_enabled, tp3_leg_source, tp3_leg_percent,
sl_extra_pips, enable_break_even, use_real_balance,
starting_balance, risk_per_trade_pct, use_1m_entry,
model_mode,
metaapi_account_id, metaapi_token</code></pre>
          </div>
        </div>
      </body>
    </html>
    """
    return Response(html, mimetype="text/html")


if __name__ == "__main__":
    if not APP_USERNAME or not APP_PASSWORD:
        raise SystemExit("Set APP_USERNAME and APP_PASSWORD before running the server.")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5100")), debug=False)
