from __future__ import annotations

import json
from typing import Any

from .models import AccountSettings, Backtest, Mt5Account, Trade, User


def serialize_user(user: User) -> dict[str, Any]:
    return {
        "id": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "avatar_url": user.avatar_url,
        "timezone": user.timezone,
        "created_at": user.created_at.isoformat() + "Z" if user.created_at else None,
        "updated_at": user.updated_at.isoformat() + "Z" if user.updated_at else None,
        "email_verified": bool(user.email_verified),
        "two_factor_enabled": bool(user.two_factor_enabled),
        "subscription_tier": user.subscription_tier,
        "subscription_expires_at": (
            user.subscription_expires_at.isoformat() + "Z"
            if user.subscription_expires_at
            else None
        ),
    }


def serialize_account_settings(settings: AccountSettings | None) -> dict[str, Any]:
    if settings is None:
        return {}
    return {
        "ai_enabled": bool(settings.ai_enabled),
        "selected_model": settings.selected_model,
        "risk_per_trade": float(settings.risk_per_trade or 0),
        "use_real_balance": bool(settings.use_real_balance),
        "balance_override": float(settings.balance_override or 0),
        "min_lot": float(settings.min_lot or 0),
        "max_lot": float(settings.max_lot or 0),
        "enable_drawdown_limit": bool(settings.enable_drawdown_limit),
        "max_drawdown_percent": float(settings.max_drawdown_percent or 0),
        "enable_consecutive_loss": bool(settings.enable_consecutive_loss),
        "max_consecutive_losses": settings.max_consecutive_losses,
        "enable_max_daily_losses": bool(settings.enable_max_daily_losses),
        "max_daily_losses": settings.max_daily_losses,
        "enable_profit_target": bool(settings.enable_profit_target),
        "profit_target_percent": float(settings.profit_target_percent or 0),
        "sl_lookback": settings.sl_lookback,
        "sl_buffer": settings.sl_buffer,
        "enable_break_even": bool(settings.enable_break_even),
        "be_buffer": settings.be_buffer,
        "tp1_percent": settings.tp1_percent,
        "tp2_percent": settings.tp2_percent,
        "enable_tp3": bool(settings.enable_tp3),
        "tp3_percent": settings.tp3_percent,
        "london_enabled": bool(settings.london_enabled),
        "ny_enabled": bool(settings.ny_enabled),
        "notify_trade_open": bool(settings.notify_trade_open),
        "notify_tp_hit": bool(settings.notify_tp_hit),
        "notify_sl_hit": bool(settings.notify_sl_hit),
        "notify_limit_hit": bool(settings.notify_limit_hit),
    }


def serialize_account(account: Mt5Account) -> dict[str, Any]:
    settings = account.settings
    return {
        "id": account.id,
        "nickname": account.nickname,
        "account_number": account.account_number,
        "platform": account.platform,
        "type": account.account_type,
        "broker": account.broker,
        "server": account.server,
        "metaapi_account_id": account.metaapi_account_id,
        "metaapi_region": account.metaapi_region,
        "balance": float(account.balance or 0),
        "equity": float(account.equity or 0),
        "margin": float(account.margin or 0),
        "free_margin": float(account.free_margin or 0),
        "status": account.status,
        "ai_enabled": bool(settings.ai_enabled) if settings else False,
        "active_model": settings.selected_model if settings else None,
        "trade_tag": account.trade_tag,
        "magic_number": account.magic_number,
        "last_sync": account.last_sync_at.isoformat() + "Z" if account.last_sync_at else None,
        "settings": serialize_account_settings(settings),
    }


def serialize_trade(trade: Trade) -> dict[str, Any]:
    entry_time = trade.entry_time
    date_value = entry_time.date().isoformat() if entry_time else None
    time_value = entry_time.time().strftime("%H:%M") if entry_time else None
    duration_seconds = trade.duration_seconds or 0
    duration = None
    if duration_seconds:
        hours = duration_seconds // 3600
        minutes = (duration_seconds % 3600) // 60
        duration = f"{hours}h {minutes}m"
    rules_passed = trade.rules_passed
    if rules_passed:
        try:
            rules_passed = json.loads(rules_passed)
        except json.JSONDecodeError:
            pass
    return {
        "id": trade.id,
        "date": date_value,
        "time": time_value,
        "direction": trade.direction,
        "entry_price": float(trade.entry_price),
        "exit_price": float(trade.exit_price) if trade.exit_price is not None else None,
        "stop_loss": float(trade.stop_loss),
        "tp1": float(trade.take_profit_1) if trade.take_profit_1 is not None else None,
        "tp2": float(trade.take_profit_2) if trade.take_profit_2 is not None else None,
        "tp3": float(trade.take_profit_3) if trade.take_profit_3 is not None else None,
        "pips": float(trade.pips) if trade.pips is not None else None,
        "pnl": float(trade.pnl) if trade.pnl is not None else None,
        "r_multiple": float(trade.r_multiple) if trade.r_multiple is not None else None,
        "duration": duration,
        "duration_seconds": duration_seconds,
        "outcome": trade.outcome,
        "session": trade.session,
        "position_size": float(trade.position_size),
        "risk_amount": float(trade.risk_amount) if trade.risk_amount is not None else None,
        "balance_after": float(trade.balance_after) if trade.balance_after is not None else None,
        "entry_time": trade.entry_time.isoformat() + "Z" if trade.entry_time else None,
        "exit_time": trade.exit_time.isoformat() + "Z" if trade.exit_time else None,
        "rules_passed": rules_passed,
    }


def serialize_backtest(backtest: Backtest, include_rows: bool = True) -> dict[str, Any]:
    payload = {
        "id": backtest.id,
        "name": backtest.name,
        "status": backtest.status,
        "progress": backtest.progress,
        "model": backtest.model,
        "model_mode": backtest.model,
        "date_start": backtest.date_start.isoformat() if backtest.date_start else None,
        "date_end": backtest.date_end.isoformat() if backtest.date_end else None,
        "error": backtest.error_message,
        "starting_balance": float(backtest.starting_balance) if backtest.starting_balance else None,
        "ending_balance": float(backtest.ending_balance) if backtest.ending_balance is not None else None,
        "net_pnl": float(backtest.net_pnl) if backtest.net_pnl is not None else None,
        "net_pnl_percent": float(backtest.net_pnl_percent) if backtest.net_pnl_percent is not None else None,
        "total_trades": backtest.total_trades,
        "winning_trades": backtest.winning_trades,
        "losing_trades": backtest.losing_trades,
        "break_even_trades": backtest.break_even_trades,
        "win_rate": float(backtest.win_rate) if backtest.win_rate is not None else None,
        "diagnostics": None,
        "rows": None,
        "created_at": backtest.created_at.isoformat() + "Z" if backtest.created_at else None,
        "updated_at": backtest.updated_at.isoformat() + "Z" if backtest.updated_at else None,
    }
    if include_rows and backtest.rows_json:
        try:
            payload["rows"] = json.loads(backtest.rows_json)
        except json.JSONDecodeError:
            payload["rows"] = None
    if backtest.diagnostics_json:
        try:
            payload["diagnostics"] = json.loads(backtest.diagnostics_json)
        except json.JSONDecodeError:
            payload["diagnostics"] = None
    return payload
