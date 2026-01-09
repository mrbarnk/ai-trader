from __future__ import annotations

import json
from typing import Any

from .models import Notification
from .socketio_manager import socketio


def create_notification(
    session,
    user_id: int,
    n_type: str,
    title: str,
    message: str,
    data: dict[str, Any] | None = None,
) -> Notification:
    payload = json.dumps(data) if data else None
    note = Notification(
        user_id=user_id,
        type=n_type,
        title=title,
        message=message,
        data_json=payload,
        read=False,
    )
    session.add(note)
    session.flush()
    return note


def serialize_notification(note: Notification) -> dict[str, Any]:
    data = None
    if note.data_json:
        try:
            data = json.loads(note.data_json)
        except json.JSONDecodeError:
            data = None
    return {
        "id": note.id,
        "type": note.type,
        "title": note.title,
        "message": note.message,
        "read": bool(note.read),
        "data": data,
        "created_at": note.created_at.isoformat() + "Z" if note.created_at else None,
    }


def emit_notification(user_id: int, payload: dict[str, Any]) -> None:
    try:
        emitter = getattr(socketio, "emit", None)
        if not callable(emitter):
            return
        emitter("notification", payload, room=f"user_{user_id}")
    except Exception as e:
        # Log error but don't crash
        print(f"Failed to emit notification: {e}")
