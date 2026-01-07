from __future__ import annotations

import hmac
import json
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any

from flask import Flask, Response, Request, request
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

MAX_CSV_BYTES = 10 * 1024 * 1024
MAX_CANDLES = 50_000
MAX_RANGE_DAYS = 31

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CSV_BYTES

VIEWER_HTML = (BASE_DIR / "backtest_viewer.html").read_text(encoding="utf-8")


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
        return _render_error(message, _err.code or 400)
    return _render_error("Something went wrong. Please try again.", 500)


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


if __name__ == "__main__":
    if not APP_USERNAME or not APP_PASSWORD:
        raise SystemExit("Set APP_USERNAME and APP_PASSWORD before running the server.")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
