from __future__ import annotations

import os

from server.app_factory import create_app
from server.settings import APP_PASSWORD, APP_USERNAME
from server.socketio_manager import socketio

app = create_app()


if __name__ == "__main__":
    if not APP_USERNAME or not APP_PASSWORD:
        raise SystemExit("Set APP_USERNAME and APP_PASSWORD before running the server.")
    socketio.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "5100")), debug=False)
