from __future__ import annotations

from flask import request
from flask_socketio import SocketIO, join_room

from .jwt_utils import decode_access_token
from .settings import SOCKETIO_ASYNC_MODE, SOCKETIO_CORS_ORIGINS

socketio = SocketIO(
    cors_allowed_origins=SOCKETIO_CORS_ORIGINS or "*",
    async_mode=SOCKETIO_ASYNC_MODE,
)


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
