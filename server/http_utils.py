from __future__ import annotations

import json
from typing import Any

from flask import Response, request

from .jwt_utils import decode_access_token
from .models import User


def json_response(payload: dict[str, Any], status_code: int = 200) -> Response:
    return Response(json.dumps(payload), status=status_code, mimetype="application/json")


def json_error(message: str, status_code: int = 400) -> Response:
    return json_response({"error": message}, status_code)


def bearer_token() -> str | None:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    return auth.split(" ", 1)[1].strip() or None


def require_token(session) -> User | None:
    token = bearer_token()
    if not token:
        return None
    payload = decode_access_token(token)
    if payload and payload.get("sub"):
        try:
            user_id = int(payload["sub"])
        except (TypeError, ValueError):
            user_id = None
        if user_id:
            return session.query(User).filter_by(id=user_id).first()
    return None
