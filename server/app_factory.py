from __future__ import annotations

import hmac
import logging
import os

from flask import Flask, Request, request
from flask_cors import CORS
from sqlalchemy import text
from werkzeug.exceptions import HTTPException

from alembic import command
from alembic.config import Config

from .http_utils import json_error
from .models import engine
from .routes_api import api
from .routes_web import render_error_html, web
from .settings import (
    APP_PASSWORD,
    APP_USERNAME,
    AUTO_RUN_MIGRATIONS,
    AUTO_RUN_MIGRATIONS_LOCK,
    BASE_DIR,
    CORS_ALLOWED_ORIGINS,
    DATABASE_URL,
    DB_STARTUP_CHECK,
    DB_STARTUP_CHECK_TABLES,
    MAX_CSV_BYTES,
)
from .socketio_manager import init_socketio


def _check_auth(req: Request) -> bool:
    if not APP_USERNAME or not APP_PASSWORD:
        return False
    auth = req.authorization
    if auth is None:
        return False
    return hmac.compare_digest(auth.username, APP_USERNAME) and hmac.compare_digest(
        auth.password, APP_PASSWORD
    )


def _require_auth():
    response = render_error_html("Authentication required.", 401)
    response.headers["WWW-Authenticate"] = 'Basic realm="Backtest Login"'
    return response


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CSV_BYTES
    if not app.logger.handlers:
        logging.basicConfig(level=logging.INFO)
    # supports_credentials = CORS_ALLOWED_ORIGINS != "*"
    CORS(
        app,
        resources={
            r"/api/*": {"origins": CORS_ALLOWED_ORIGINS},
            r"/auth/*": {"origins": CORS_ALLOWED_ORIGINS},
            r"/socket.io/*": {"origins": CORS_ALLOWED_ORIGINS},
        },
        # supports_credentials=supports_credentials,
        # allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
        # methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
    )

    @app.before_request
    def enforce_basic_auth():
        if (
            request.path.startswith("/api/")
            or request.path.startswith("/auth/")
            or request.path.startswith("/socket.io/")
            or request.path == "/health"
        ):
            return None
        if _check_auth(request):
            return None
        return _require_auth()

    @app.errorhandler(Exception)
    def handle_error(error: Exception):
        app.logger.exception("Unhandled error on %s %s", request.method, request.path)
        if isinstance(error, HTTPException):
            if request.path.startswith("/api/") or request.path.startswith("/auth/"):
                return json_error("Something went wrong. Please try again.", error.code or 500)
            return render_error_html("Something went wrong. Please try again.", error.code or 500)
        if request.path.startswith("/api/") or request.path.startswith("/auth/"):
            return json_error("Something went wrong. Please try again.", 500)
        return render_error_html("Something went wrong. Please try again.", 500)

    app.register_blueprint(api)
    app.register_blueprint(web)
    init_socketio(app)
    _run_migrations(app)
    _log_db_startup(app)
    _start_live_trade_worker(app)

    return app


def _run_migrations(app: Flask) -> None:
    if not AUTO_RUN_MIGRATIONS:
        return
    lock_path = AUTO_RUN_MIGRATIONS_LOCK
    lock_handle = None
    try:
        lock_handle = open(lock_path, "w", encoding="utf-8")
        try:
            import fcntl  # type: ignore

            fcntl.flock(lock_handle, fcntl.LOCK_EX)
        except Exception:
            pass
        alembic_cfg = Config(str(BASE_DIR / "alembic.ini"))
        alembic_cfg.set_main_option("script_location", str(BASE_DIR / "migrations"))
        alembic_cfg.set_main_option("sqlalchemy.url", DATABASE_URL)
        command.upgrade(alembic_cfg, "head")
        app.logger.info("Database migrations applied.")
    except Exception as exc:  # pragma: no cover - startup guard
        app.logger.error("Database migration failed: %s", exc)
        raise
    finally:
        if lock_handle:
            try:
                lock_handle.close()
            except Exception:
                pass


def _log_db_startup(app: Flask) -> None:
    if not DB_STARTUP_CHECK:
        return
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            if DB_STARTUP_CHECK_TABLES:
                conn.execute(text("SELECT 1 FROM users LIMIT 1"))
        app.logger.info("Database connection check: OK")
    except Exception as exc:  # pragma: no cover - defensive log
        app.logger.error("Database connection check failed: %s", exc)


def _start_live_trade_worker(app: Flask) -> None:
    try:
        from .live_trade_worker import start_live_trade_worker
    except Exception as exc:
        app.logger.warning("Live trade worker unavailable: %s", exc)
        return
    try:
        start_live_trade_worker()
        app.logger.info("Live trade worker started.")
    except Exception as exc:
        app.logger.warning("Live trade worker failed to start: %s", exc)
