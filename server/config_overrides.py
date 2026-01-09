from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from typing import Any

from trader import config

from .models import User, UserConfig

DEFAULT_MODEL_MODE = "aggressive"

MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "aggressive": {
        "model_mode": "aggressive",
        "tp_leg_source": "15M",
        "tp1_leg_percent": 0.5,
        "tp2_leg_percent": 0.9,
        "tp3_enabled": False,
        "tp3_leg_source": "4H",
        "tp3_leg_percent": 1.0,
        "sl_extra_pips": 3.0,
        "enable_break_even": True,
        "use_real_balance": False,
        "starting_balance": 10000.0,
        "risk_per_trade_pct": 1.0,
        "use_1m_entry": True,
        "enable_1m_entry_sell": True,
        "require_1m_choch_premium": True,
        "require_5m_choch_premium": True,
        "require_5m_choch_premium_sell": False,
        "premium_cross_level": 0.5,
        "discount_cross_level": 0.5,
        "enable_spread_filter": False,
        "max_spread_pips": 1.0,
        "assume_zero_spread": False,
        "enable_choch_range_filter": False,
        "min_choch_range_pips": 6.0,
        "require_no_liquidity_sweep": False,
        "require_liquidity_sweep_sell": False,
    },
    "passive": {
        "model_mode": "passive",
        "tp_leg_source": "4H",
        "tp1_leg_percent": 0.5,
        "tp2_leg_percent": 0.9,
        "tp3_enabled": False,
        "tp3_leg_source": "4H",
        "tp3_leg_percent": 1.0,
        "sl_extra_pips": 3.0,
        "enable_break_even": True,
        "use_real_balance": False,
        "starting_balance": 10000.0,
        "risk_per_trade_pct": 1.0,
        "use_1m_entry": False,
        "enable_1m_entry_sell": False,
        "require_1m_choch_premium": True,
        "require_5m_choch_premium": True,
        "require_5m_choch_premium_sell": False,
        "premium_cross_level": 0.75,
        "discount_cross_level": 0.25,
        "enable_spread_filter": False,
        "max_spread_pips": 2.0,
        "assume_zero_spread": False,
        "enable_choch_range_filter": False,
        "min_choch_range_pips": 6.0,
        "require_no_liquidity_sweep": False,
        "require_liquidity_sweep_sell": False,
    },
    "sniper": {
        "model_mode": "sniper",
        "tp_leg_source": "4H",
        "tp1_leg_percent": 0.5,
        "tp2_leg_percent": 0.9,
        "tp3_enabled": False,
        "tp3_leg_source": "4H",
        "tp3_leg_percent": 1.0,
        "sl_extra_pips": 3.0,
        "enable_break_even": True,
        "use_real_balance": False,
        "starting_balance": 10000.0,
        "risk_per_trade_pct": 1.0,
        "use_1m_entry": True,
        "enable_1m_entry_sell": True,
        "require_1m_choch_premium": False,
        "require_5m_choch_premium": False,
        "require_5m_choch_premium_sell": False,
        "premium_cross_level": 0.5,
        "discount_cross_level": 0.5,
        "enable_spread_filter": False,
        "max_spread_pips": 1.0,
        "assume_zero_spread": False,
        "enable_choch_range_filter": False,
        "min_choch_range_pips": 6.0,
        "require_no_liquidity_sweep": False,
        "require_liquidity_sweep_sell": False,
    },
}

BASE_DEFAULTS: dict[str, Any] = {
    "metaapi_account_id": "",
    "metaapi_token": "",
}


def _model_defaults(model_mode: str | None) -> dict[str, Any]:
    selected = (model_mode or DEFAULT_MODEL_MODE).lower()
    defaults = MODEL_DEFAULTS.get(selected, MODEL_DEFAULTS[DEFAULT_MODEL_MODE]).copy()
    defaults.update(BASE_DEFAULTS)
    return defaults


DEFAULT_USER_CONFIG: dict[str, Any] = _model_defaults(DEFAULT_MODEL_MODE)

CONFIG_FLOAT_KEYS = {
    "tp1_leg_percent",
    "tp2_leg_percent",
    "tp3_leg_percent",
    "sl_extra_pips",
    "starting_balance",
    "risk_per_trade_pct",
    "premium_cross_level",
    "discount_cross_level",
    "max_spread_pips",
    "min_choch_range_pips",
}
CONFIG_BOOL_KEYS = {
    "tp3_enabled",
    "enable_break_even",
    "use_real_balance",
    "use_1m_entry",
    "enable_1m_entry_sell",
    "require_1m_choch_premium",
    "require_5m_choch_premium",
    "require_5m_choch_premium_sell",
    "enable_spread_filter",
    "assume_zero_spread",
    "enable_choch_range_filter",
    "require_no_liquidity_sweep",
    "require_liquidity_sweep_sell",
}
CONFIG_STRING_KEYS = {
    "model_mode",
    "tp_leg_source",
    "tp3_leg_source",
    "metaapi_account_id",
    "metaapi_token",
}
ALLOWED_CONFIG_KEYS = CONFIG_FLOAT_KEYS | CONFIG_BOOL_KEYS | CONFIG_STRING_KEYS
CONFIG_LOCK = threading.Lock()


def sanitize_config(data: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in data.items():
        if key not in ALLOWED_CONFIG_KEYS:
            continue
        if key in CONFIG_FLOAT_KEYS:
            try:
                sanitized[key] = float(value)
            except (TypeError, ValueError):
                continue
        elif key in CONFIG_BOOL_KEYS:
            sanitized[key] = bool(value)
        else:
            sanitized[key] = str(value).strip()
    return sanitized


def load_user_config(
    session, user: User, model_mode: str | None = None
) -> dict[str, Any]:
    existing = session.query(UserConfig).filter_by(user_id=user.id).first()
    if not existing:
        return _model_defaults(model_mode)
    try:
        stored = json.loads(existing.config_json)
    except json.JSONDecodeError:
        stored = {}
    stored_mode = stored.get("model_mode")
    merged = _model_defaults(model_mode or stored_mode)
    merged.update({k: v for k, v in stored.items() if k in ALLOWED_CONFIG_KEYS})
    return merged


def save_user_config(session, user: User, config_data: dict[str, Any]) -> dict[str, Any]:
    merged = load_user_config(session, user, config_data.get("model_mode"))
    merged.update(config_data)
    existing = session.query(UserConfig).filter_by(user_id=user.id).first()
    payload = json.dumps(merged)
    if existing:
        existing.config_json = payload
    else:
        session.add(UserConfig(user_id=user.id, config_json=payload))
    return merged


def build_config_overrides(user_config: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "MODEL_MODE": user_config.get("model_mode", config.MODEL_MODE),
        "TP_LEG_SOURCE": user_config.get("tp_leg_source", config.TP_LEG_SOURCE),
        "TP1_LEG_PERCENT": user_config.get("tp1_leg_percent", config.TP1_LEG_PERCENT),
        "TP2_LEG_PERCENT": user_config.get("tp2_leg_percent", config.TP2_LEG_PERCENT),
        "TP3_ENABLED": user_config.get("tp3_enabled", config.TP3_ENABLED),
        "TP3_LEG_SOURCE": user_config.get("tp3_leg_source", config.TP3_LEG_SOURCE),
        "TP3_LEG_PERCENT": user_config.get("tp3_leg_percent", config.TP3_LEG_PERCENT),
        "SL_EXTRA_PIPS": user_config.get("sl_extra_pips", config.SL_EXTRA_PIPS),
        "ENABLE_BREAK_EVEN": user_config.get(
            "enable_break_even", config.ENABLE_BREAK_EVEN
        ),
        "RISK_PER_TRADE_PCT": user_config.get(
            "risk_per_trade_pct", config.RISK_PER_TRADE_PCT
        ),
        "SNIPER_RISK_PER_TRADE_PCT": user_config.get(
            "risk_per_trade_pct", config.SNIPER_RISK_PER_TRADE_PCT
        ),
        "USE_1M_ENTRY": user_config.get("use_1m_entry", config.USE_1M_ENTRY),
        "ENABLE_1M_ENTRY_SELL": user_config.get(
            "enable_1m_entry_sell", config.ENABLE_1M_ENTRY_SELL
        ),
        "REQUIRE_1M_CHOCH_PREMIUM": user_config.get(
            "require_1m_choch_premium", config.REQUIRE_1M_CHOCH_PREMIUM
        ),
        "REQUIRE_5M_CHOCH_PREMIUM": user_config.get(
            "require_5m_choch_premium", config.REQUIRE_5M_CHOCH_PREMIUM
        ),
        "REQUIRE_5M_CHOCH_PREMIUM_SELL": user_config.get(
            "require_5m_choch_premium_sell", config.REQUIRE_5M_CHOCH_PREMIUM_SELL
        ),
        "PREMIUM_CROSS_LEVEL": user_config.get(
            "premium_cross_level", config.PREMIUM_CROSS_LEVEL
        ),
        "DISCOUNT_CROSS_LEVEL": user_config.get(
            "discount_cross_level", config.DISCOUNT_CROSS_LEVEL
        ),
        "ENABLE_SPREAD_FILTER": user_config.get(
            "enable_spread_filter", config.ENABLE_SPREAD_FILTER
        ),
        "MAX_SPREAD_PIPS": user_config.get("max_spread_pips", config.MAX_SPREAD_PIPS),
        "ASSUME_ZERO_SPREAD": user_config.get(
            "assume_zero_spread", config.ASSUME_ZERO_SPREAD
        ),
        "ENABLE_CHOCH_RANGE_FILTER": user_config.get(
            "enable_choch_range_filter", config.ENABLE_CHOCH_RANGE_FILTER
        ),
        "MIN_CHOCH_RANGE_PIPS": user_config.get(
            "min_choch_range_pips", config.MIN_CHOCH_RANGE_PIPS
        ),
        "REQUIRE_NO_LIQUIDITY_SWEEP": user_config.get(
            "require_no_liquidity_sweep", config.REQUIRE_NO_LIQUIDITY_SWEEP
        ),
        "REQUIRE_LIQUIDITY_SWEEP_SELL": user_config.get(
            "require_liquidity_sweep_sell", config.REQUIRE_LIQUIDITY_SWEEP_SELL
        ),
        "ACCOUNT_BALANCE_OVERRIDE": user_config.get(
            "starting_balance", config.ACCOUNT_BALANCE_OVERRIDE
        ),
    }
    return overrides


@contextmanager
def apply_config_overrides(overrides: dict[str, Any]):
    with CONFIG_LOCK:
        original = {key: getattr(config, key) for key in overrides.keys()}
        for key, value in overrides.items():
            setattr(config, key, value)
        try:
            yield
        finally:
            for key, value in original.items():
                setattr(config, key, value)
