from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, request

from trader import config
from trader.backtest import run_backtest

from .backtest_jobs import validate_candles
from .settings import BASE_STYLE, VIEWER_HTML


web = Blueprint("web", __name__)


def render_error_html(message: str, status_code: int = 400) -> Response:
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
            <h1>Backtest Error</h1>
            <p>{message}</p>
            <div style="margin-top: 16px;">
              <a href="/">Go back</a>
            </div>
          </div>
        </div>
      </body>
    </html>
    """
    return Response(html, status=status_code, mimetype="text/html")


def _render_viewer(rows: list[dict[str, Any]]) -> Response:
    payload = json.dumps(rows)
    injection = (
        f"<script>window.PRELOADED_ROWS = {payload};"
        "window.__applyPreloadedRows && window.__applyPreloadedRows(window.PRELOADED_ROWS);"
        "</script>"
    )
    html = VIEWER_HTML.replace("</body>", f"{injection}\n</body>")
    return Response(html, mimetype="text/html")


def _load_rows(jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@web.route("/", methods=["GET"])
def index() -> Response:
    html = f"""
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>Backtest Upload</title>
        <style>{BASE_STYLE}</style>
      </head>
      <body>
        <div class="wrap">
          <div class="panel">
            <h1>Backtest Upload</h1>
            <p>Upload a 1-minute GU CSV to run a backtest and view results.</p>
            <div class="upload">
              <form action="/run" method="post" enctype="multipart/form-data">
                <input type="file" name="csv" accept=".csv" required />
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


@web.route("/health", methods=["GET"])
def health() -> Response:
    return Response("ok", mimetype="text/plain")


@web.route("/run", methods=["POST"])
def run() -> Response:
    upload = request.files.get("csv")
    if not upload or upload.filename == "":
        return render_error_html("No CSV file uploaded.")

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "upload.csv"
        upload.save(csv_path)

        error = validate_candles(csv_path)
        if error:
            return render_error_html(error)

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
