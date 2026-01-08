from __future__ import annotations

from flask import request

try:
    import werkzeug.serving as _wz_serving
    from werkzeug._reloader import run_with_reloader as _run_with_reloader
except Exception:  # pragma: no cover - defensive import shim
    _wz_serving = None
    _run_with_reloader = None

if _wz_serving is not None and _run_with_reloader is not None:
    if not hasattr(_wz_serving, "run_with_reloader"):
        _wz_serving.run_with_reloader = _run_with_reloader

from flask_socketio import SocketIO, join_room

from .jwt_utils import decode_access_token
from .settings import CORS_ALLOWED_ORIGINS, SOCKETIO_ASYNC_MODE, SOCKETIO_CORS_ORIGINS

cors_origins = SOCKETIO_CORS_ORIGINS or CORS_ALLOWED_ORIGINS or "*"

socketio = SocketIO(cors_allowed_origins=cors_origins, async_mode=SOCKETIO_ASYNC_MODE)


def init_socketio(app) -> SocketIO:
    socketio.init_app(app)

    @socketio.on("connect")
    def handle_connect(auth):
        token = None
        if isinstance(auth, dict):
            token = auth.get("token")
        if not token:
            token = request.args.get("token")
        if not token:
            return False
        payload = decode_access_token(token)
        if not payload or not payload.get("sub"):
            return False
        join_room(f"user_{payload['sub']}")
        return True

    return socketio
