from __future__ import annotations

import hmac

from flask import Flask, Request, request
from werkzeug.exceptions import HTTPException

from .http_utils import json_error
from .routes_api import api
from .routes_web import render_error_html, web
from .settings import APP_PASSWORD, APP_USERNAME, MAX_CSV_BYTES
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

    return app
