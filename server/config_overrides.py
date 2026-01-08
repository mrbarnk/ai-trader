from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from typing import Any

from trader import config

from .models import User, UserConfig

DEFAULT_USER_CONFIG: dict[str, Any] = {
    "model_mode": "aggressive",
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
    "metaapi_account_id": "",
    "metaapi_token": "",
}

CONFIG_FLOAT_KEYS = {
    "tp1_leg_percent",
    "tp2_leg_percent",
    "tp3_leg_percent",
    "sl_extra_pips",
    "starting_balance",
    "risk_per_trade_pct",
}
CONFIG_BOOL_KEYS = {"tp3_enabled", "enable_break_even", "use_real_balance", "use_1m_entry"}
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


def load_user_config(session, user: User) -> dict[str, Any]:
    existing = session.query(UserConfig).filter_by(user_id=user.id).first()
    if not existing:
        return DEFAULT_USER_CONFIG.copy()
    try:
        stored = json.loads(existing.config_json)
    except json.JSONDecodeError:
        stored = {}
    merged = DEFAULT_USER_CONFIG.copy()
    merged.update({k: v for k, v in stored.items() if k in ALLOWED_CONFIG_KEYS})
    return merged


def save_user_config(session, user: User, config_data: dict[str, Any]) -> dict[str, Any]:
    merged = load_user_config(session, user)
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
        "ENABLE_BREAK_EVEN": user_config.get("enable_break_even", config.ENABLE_BREAK_EVEN),
        "RISK_PER_TRADE_PCT": user_config.get("risk_per_trade_pct", config.RISK_PER_TRADE_PCT),
        "USE_1M_ENTRY": user_config.get("use_1m_entry", config.USE_1M_ENTRY),
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
