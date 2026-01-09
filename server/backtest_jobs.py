from __future__ import annotations

import json
import secrets
import shutil
import tempfile
from datetime import datetime, timedelta, date, time
from pathlib import Path
from typing import Any

from trader.backtest import load_csv_candles, run_backtest
from trader import config

from .config_overrides import apply_config_overrides, build_config_overrides
from .models import Backtest, db_session
from .notifications import create_notification, emit_notification, serialize_notification
from .settings import BACKTEST_KEEP_CSV, MAX_CANDLES, MAX_RANGE_DAYS


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def validate_candles(csv_path: Path) -> str | None:
    candles = load_csv_candles(csv_path)
    if not candles:
        return "CSV is empty after parsing."
    if len(candles) > MAX_CANDLES:
        return f"CSV has {len(candles)} candles; max allowed is {MAX_CANDLES}."
    span = candles[-1].time_utc - candles[0].time_utc
    if span > timedelta(days=MAX_RANGE_DAYS):
        return f"CSV spans {span.days} days; max allowed is {MAX_RANGE_DAYS}."
    return None


def _date_bounds(value: date | None, is_end: bool) -> datetime | None:
    if not value:
        return None
    if is_end:
        return datetime.combine(value, time.max).replace(microsecond=0)
    return datetime.combine(value, time.min)


def summarize_backtest_rows(rows: list[dict[str, Any]], starting_balance: float | None) -> dict[str, Any]:
    total_trades = len(rows)
    wins = 0
    losses = 0
    break_even = 0
    total_r = 0.0
    best_r = None
    worst_r = None
    total_buys = 0
    total_sells = 0
    buy_wins = 0
    sell_wins = 0
    total_pnl = 0.0

    for row in rows:
        direction = row.get("direction")
        pnl = float(row.get("pnl") or 0)
        r_multiple = float(row.get("r_multiple") or 0)
        total_pnl += pnl
        total_r += r_multiple
        if best_r is None or r_multiple > best_r:
            best_r = r_multiple
        if worst_r is None or r_multiple < worst_r:
            worst_r = r_multiple
        if pnl > 0:
            wins += 1
            if direction == "BUY":
                buy_wins += 1
            elif direction == "SELL":
                sell_wins += 1
        elif pnl < 0:
            losses += 1
        else:
            break_even += 1
        if direction == "BUY":
            total_buys += 1
        elif direction == "SELL":
            total_sells += 1

    win_rate = (wins / total_trades * 100) if total_trades else 0
    buy_win_rate = (buy_wins / total_buys * 100) if total_buys else 0
    sell_win_rate = (sell_wins / total_sells * 100) if total_sells else 0
    avg_r = (total_r / total_trades) if total_trades else 0
    start_balance = float(starting_balance or 0)
    ending_balance = start_balance + total_pnl if start_balance else None
    net_pnl_percent = (total_pnl / start_balance * 100) if start_balance else 0

    return {
        "ending_balance": ending_balance,
        "net_pnl": total_pnl,
        "net_pnl_percent": net_pnl_percent,
        "total_trades": total_trades,
        "winning_trades": wins,
        "losing_trades": losses,
        "break_even_trades": break_even,
        "win_rate": win_rate,
        "total_buys": total_buys,
        "total_sells": total_sells,
        "buy_win_rate": buy_win_rate,
        "sell_win_rate": sell_win_rate,
        "total_r": total_r,
        "avg_r": avg_r,
        "best_r": best_r,
        "worst_r": worst_r,
    }


def create_backtest_job(
    session,
    user_id: int,
    model_mode: str | None,
    name: str | None = None,
    date_start: date | None = None,
    date_end: date | None = None,
    starting_balance: float | None = None,
    symbol: str | None = None,
    settings_json: str | None = None,
    csv_path: Path | None = None,
) -> str:
    job_id = secrets.token_urlsafe(12)
    job = Backtest(
        id=job_id,
        user_id=user_id,
        name=name,
        model=model_mode,
        date_start=date_start,
        date_end=date_end,
        starting_balance=starting_balance,
        symbol=symbol or "GBPUSD",
        settings_json=settings_json,
        csv_path=str(csv_path) if csv_path else None,
        status="pending",
        progress=0,
    )
    session.add(job)
    session.commit()
    return job_id


def update_backtest_job(job_id: str, **fields: Any) -> None:
    with db_session() as session:
        job = session.query(Backtest).filter_by(id=job_id).first()
        if not job:
            return
        for key, value in fields.items():
            setattr(job, key, value)


def run_backtest_job(
    job_id: str,
    csv_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> None:
    output_dir = Path(tempfile.mkdtemp(prefix="backtest_"))
    output_path = output_dir / "signals.jsonl"
    outcome_path = output_path.with_name(output_path.stem + "_outcomes.jsonl")
    cleanup_path: Path | None = None
    start_dt: datetime | None = None
    end_dt: datetime | None = None
    job_settings_json: str | None = None
    job_model: str | None = None
    job_starting_balance: float | None = None

    with db_session() as session:
        job = session.query(Backtest).filter_by(id=job_id).first()
        if not job:
            return
        job_settings_json = job.settings_json
        job_model = job.model
        job_starting_balance = (
            float(job.starting_balance) if job.starting_balance is not None else None
        )
        if csv_path is None and job.csv_path:
            csv_path = Path(job.csv_path)
        cleanup_path = csv_path
        start_dt = _date_bounds(job.date_start, is_end=False)
        end_dt = _date_bounds(job.date_end, is_end=True)

    if csv_path is None:
        update_backtest_job(job_id, status="failed", error_message="CSV file is missing.")
        return
    if not csv_path.exists():
        update_backtest_job(
            job_id, status="failed", error_message="CSV file was removed."
        )
        return

    if overrides is None:
        user_config: dict[str, Any] = {}
        if job_settings_json:
            try:
                user_config = json.loads(job_settings_json)
            except json.JSONDecodeError:
                user_config = {}
        if job_model:
            user_config.setdefault("model_mode", job_model)
        overrides = build_config_overrides(user_config)

    if job_starting_balance is not None:
        overrides["ACCOUNT_BALANCE_OVERRIDE"] = job_starting_balance

    update_backtest_job(
        job_id,
        status="processing",
        progress=0,
        starting_balance=overrides.get("ACCOUNT_BALANCE_OVERRIDE"),
        csv_path=str(csv_path),
    )

    def progress_callback(step_index: int, total_steps: int) -> None:
        if total_steps <= 0:
            percent = 0
        else:
            percent = int((step_index / total_steps) * 100)
        update_backtest_job(job_id, progress=percent)

    original_log_setting = config.LOG_DAILY_LIMITS
    config.LOG_DAILY_LIMITS = False
    try:
        if start_dt is None and end_dt is None:
            candles = load_csv_candles(csv_path)
            if candles:
                start_dt = candles[0].time_utc
                end_dt = candles[-1].time_utc
                update_backtest_job(
                    job_id,
                    date_start=start_dt.date(),
                    date_end=end_dt.date(),
                )
        if start_dt and end_dt and end_dt < start_dt:
            raise ValueError("date_end is before date_start")
        with apply_config_overrides(overrides):
            run_backtest(
                csv_path,
                source_minutes=1,
                output_path=output_path,
                start=start_dt,
                end=end_dt,
                model_mode=overrides.get("MODEL_MODE"),
                progress_callback=progress_callback,
            )
        rows = _load_rows(outcome_path)
        diagnostics = _summarize_signal_diagnostics(output_path)
        summary = summarize_backtest_rows(rows, overrides.get("ACCOUNT_BALANCE_OVERRIDE"))
        update_backtest_job(
            job_id,
            status="completed",
            progress=100,
            rows_json=json.dumps(rows),
            diagnostics_json=json.dumps(diagnostics) if diagnostics else None,
            model=overrides.get("MODEL_MODE"),
            completed_at=datetime.utcnow(),
            ending_balance=summary.get("ending_balance"),
            net_pnl=summary.get("net_pnl"),
            net_pnl_percent=summary.get("net_pnl_percent"),
            total_trades=summary.get("total_trades"),
            winning_trades=summary.get("winning_trades"),
            losing_trades=summary.get("losing_trades"),
            break_even_trades=summary.get("break_even_trades"),
            win_rate=summary.get("win_rate"),
            total_buys=summary.get("total_buys"),
            total_sells=summary.get("total_sells"),
            buy_win_rate=summary.get("buy_win_rate"),
            sell_win_rate=summary.get("sell_win_rate"),
            total_r=summary.get("total_r"),
            avg_r=summary.get("avg_r"),
            best_r=summary.get("best_r"),
            worst_r=summary.get("worst_r"),
        )
        _notify_backtest(job_id, "completed", "Backtest completed.")
    except Exception as exc:
        update_backtest_job(job_id, status="failed", error_message=str(exc))
        _notify_backtest(job_id, "failed", f"Backtest failed: {exc}")
    finally:
        config.LOG_DAILY_LIMITS = original_log_setting
        try:
            if not BACKTEST_KEEP_CSV and cleanup_path and cleanup_path.exists():
                cleanup_path.unlink()
                update_backtest_job(job_id, csv_path=None)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


def _notify_backtest(job_id: str, status: str, message: str) -> None:
    with db_session() as session:
        job = session.query(Backtest).filter_by(id=job_id).first()
        if not job:
            return
        note = create_notification(
            session,
            job.user_id,
            "backtest",
            f"Backtest {status}",
            message,
            {"backtest_id": job_id, "status": status},
        )
        emit_notification(job.user_id, serialize_notification(note))


def _load_rows(jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _summarize_signal_diagnostics(jsonl_path: Path) -> dict[str, Any] | None:
    if not jsonl_path.exists():
        return None
    total_signals = 0
    no_trade_signals = 0
    failed_rule_counts: dict[str, int] = {}
    last_no_trade: dict[str, Any] | None = None
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total_signals += 1
            record = json.loads(line)
            if record.get("decision") != "NO_TRADE":
                continue
            no_trade_signals += 1
            rules_failed = record.get("rules_failed") or []
            if not rules_failed:
                failed_rule_counts["NO_RULES_FAILED"] = (
                    failed_rule_counts.get("NO_RULES_FAILED", 0) + 1
                )
            for rule in rules_failed:
                failed_rule_counts[rule] = failed_rule_counts.get(rule, 0) + 1
            last_no_trade = {
                "timestamp_utc": record.get("timestamp_utc"),
                "rules_failed": rules_failed,
            }
    if total_signals == 0:
        return None
    top_failed = sorted(
        failed_rule_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    return {
        "total_signals": total_signals,
        "no_trade_signals": no_trade_signals,
        "top_failed_rules": [
            {"rule": rule, "count": count} for rule, count in top_failed[:5]
        ],
        "last_no_trade": last_no_trade,
    }
