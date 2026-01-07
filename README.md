# AI-Trader

## CSV backtest format

Required columns (UTC):

- timestamp_utc (or timestamp/time/datetime)
- open
- high
- low
- close

Notes:
- Timestamps must be UTC and ISO 8601 (example: 2023-01-03T07:00:00Z)
- Source candles are resampled into 5m/15m/4h for the engine
- If `USE_1M_ENTRY = True` in `src/trader/config.py`, the CSV must be 1-minute data.

## Backtest examples

Run a full-year replay:

```bash
PYTHONPATH=src python -m trader.backtest data/gu_1m.csv \
  --source-minutes 1 \
  --start 2023-01-01T00:00:00Z \
  --end 2023-12-31T23:59:59Z \
  --output logs/backtest_signals.jsonl
```

Run a shorter window:

```bash
PYTHONPATH=src python -m trader.backtest data/gu_1m.csv \
  --source-minutes 1 \
  --start 2024-01-01T00:00:00Z \
  --end 2024-03-31T23:59:59Z
```

## Local viewers

- `backtest_viewer.html`: load `logs/backtest_signals_outcomes.jsonl` to inspect outcomes, P/L, and win rates.
- `backtest_chart.html`: load your CSV + outcomes JSONL to see a 5M chart with trade markers.

## Hosted upload (Render)

The upload app is in `app.py`. It enforces basic auth, 1-month max CSVs, and deletes uploads immediately.

Local run:

```bash
APP_USERNAME=demo APP_PASSWORD=demo PYTHONPATH=src python app.py
```

Render:
- Use `render.yaml`
- Set `APP_USERNAME` and `APP_PASSWORD` in Render environment variables
- Attach a Render Postgres database and set `DATABASE_URL`

## API

API docs are available at `/api/docs` once the service is running.

Model selection:
- Set `MODEL_MODE = "aggressive"` or `"passive"` in `src/trader/config.py`
- Or update `model_mode` via `PUT /api/config`
