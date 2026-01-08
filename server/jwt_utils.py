from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from .settings import (
    JWT_ACCESS_TTL_MINUTES,
    JWT_AUDIENCE,
    JWT_ISSUER,
    JWT_REFRESH_TTL_DAYS,
    JWT_SECRET,
)


def create_access_token(user_id: int) -> str:
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": str(user_id),
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=JWT_ACCESS_TTL_MINUTES)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def create_refresh_token(user_id: int) -> dict[str, Any]:
    now = datetime.now(tz=timezone.utc)
    expires_at = now + timedelta(days=JWT_REFRESH_TTL_DAYS)
    return {
        "user_id": user_id,
        "expires_at": expires_at,
    }


def decode_access_token(token: str) -> dict[str, Any] | None:
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=["HS256"],
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER,
        )
        return payload
    except jwt.PyJWTError:
        return None
