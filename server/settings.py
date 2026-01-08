from __future__ import annotations

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src"))

APP_USERNAME = os.getenv("APP_USERNAME")
APP_PASSWORD = os.getenv("APP_PASSWORD")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///app.db")
METAAPI_TOKEN = os.getenv("METAAPI_TOKEN")
METAAPI_PROVISIONING_URL = os.getenv(
    "METAAPI_PROVISIONING_URL", "https://mt-provisioning-api-v1.metaapi.cloud"
)
METAAPI_CLIENT_URL = os.getenv(
    "METAAPI_CLIENT_URL", "https://mt-client-api-v1.metaapi.cloud"
)
METAAPI_DEALS_PATH = os.getenv(
    "METAAPI_DEALS_PATH", "/users/current/accounts/{account_id}/deals"
)
METAAPI_TRADE_PATH = os.getenv(
    "METAAPI_TRADE_PATH", "/users/current/accounts/{account_id}/trade"
)
METAAPI_TIMEOUT_SECONDS = int(os.getenv("METAAPI_TIMEOUT_SECONDS", "15"))
METAAPI_SYNC_LOOKBACK_DAYS = int(os.getenv("METAAPI_SYNC_LOOKBACK_DAYS", "14"))
PASSWORD_MIN_LENGTH = int(os.getenv("PASSWORD_MIN_LENGTH", "8"))
PASSWORD_RESET_TTL_HOURS = int(os.getenv("PASSWORD_RESET_TTL_HOURS", "2"))
EMAIL_VERIFY_TTL_HOURS = int(os.getenv("EMAIL_VERIFY_TTL_HOURS", "48"))
MAIL_FROM_ADDRESS = os.getenv("MAIL_FROM_ADDRESS", "trader@mrbarnk.com")
MAIL_LOG_ENABLED = os.getenv("MAIL_LOG_ENABLED", "true").lower() == "true"
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")
JWT_ISSUER = os.getenv("JWT_ISSUER", "algotrade-ai")
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "algotrade-ai-users")
JWT_ACCESS_TTL_MINUTES = int(os.getenv("JWT_ACCESS_TTL_MINUTES", "30"))
JWT_REFRESH_TTL_DAYS = int(os.getenv("JWT_REFRESH_TTL_DAYS", "14"))
SMTP_HOST = os.getenv("SMTP_HOST", "imap.hostinger.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
SMTP_USE_SSL = os.getenv("SMTP_USE_SSL", "false").lower() == "true"
SOCKETIO_ASYNC_MODE = os.getenv("SOCKETIO_ASYNC_MODE", "threading")
SOCKETIO_CORS_ORIGINS = os.getenv("SOCKETIO_CORS_ORIGINS","*")
_cors_raw = os.getenv("CORS_ALLOWED_ORIGINS", "*")
if _cors_raw.strip() == "*":
    CORS_ALLOWED_ORIGINS = "*"
else:
    CORS_ALLOWED_ORIGINS = [item.strip() for item in _cors_raw.split(",") if item.strip()]
AUTO_CREATE_SCHEMA = os.getenv("AUTO_CREATE_SCHEMA", "true").lower() == "true"

MAX_CSV_BYTES = 10 * 1024 * 1024
MAX_CANDLES = 100_000
MAX_RANGE_DAYS = 93

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

VIEWER_HTML = (BASE_DIR / "backtest_viewer.html").read_text(encoding="utf-8")
