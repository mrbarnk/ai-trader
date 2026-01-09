from __future__ import annotations

import base64
import hashlib
import _hashlib
import json
import secrets
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, request
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime, timedelta
from trader.backtest import load_csv_candles, resample_candles

from .backtest_jobs import (
    create_backtest_job,
    parse_date,
    parse_datetime,
    run_backtest_job,
    validate_candles,
)
from .auth_tokens import generate_token, hash_token
from .config_overrides import (
    ALLOWED_CONFIG_KEYS,
    build_config_overrides,
    load_user_config,
    save_user_config,
    sanitize_config,
)
from .http_utils import json_error, json_response, require_token, bearer_token
from .jwt_utils import create_access_token
from .mailer import send_email
from .metaapi_client import MetaApiClient, MetaApiError
from .models import (
    AccountSettings,
    ApiToken,
    Backtest,
    Mt5Account,
    Notification,
    Trade,
    TradingModel,
    User,
    db_session,
)
from .notifications import create_notification, emit_notification, serialize_notification
from .serializers import (
    serialize_account,
    serialize_backtest,
    serialize_trade,
    serialize_user,
)
from .trade_management import manage_break_even_for_account
from .settings import (
    BACKTEST_UPLOAD_DIR,
    BACKTEST_WORKER_ENABLED,
    EMAIL_VERIFY_TTL_HOURS,
    METAAPI_SYNC_LOOKBACK_DAYS,
    METAAPI_TOKEN,
    PASSWORD_HASH_METHOD,
    PASSWORD_MIN_LENGTH,
    PASSWORD_RESET_TTL_HOURS,
)

api = Blueprint("api", __name__)


def _hash_password(password: str) -> str:
    return generate_password_hash(password, method=PASSWORD_HASH_METHOD)


_UNSUPPORTED_DIGEST = getattr(_hashlib, "UnsupportedDigestmodError", ValueError)


def _check_password(stored_hash: str, password: str) -> bool | None:
    try:
        return check_password_hash(stored_hash, password)
    except (ValueError, TypeError, _UNSUPPORTED_DIGEST):
        return None


def _issue_email_verification(user: User) -> str:
    token, token_hash = generate_token()
    user.email_verify_token_hash = token_hash
    user.email_verify_expires_at = datetime.utcnow() + timedelta(
        hours=EMAIL_VERIFY_TTL_HOURS
    )
    user.email_verify_sent_at = datetime.utcnow()
    return token


def _send_verification_email(email: str, token: str) -> None:
    send_email(
        email,
        "Verify your AlgoTrade AI email",
        f"Use this verification code to confirm your email: {token}",
    )


def _validate_strategy_settings(
    data: dict[str, Any], label: str
) -> tuple[dict[str, Any], str | None]:
    invalid_keys = sorted(set(data) - ALLOWED_CONFIG_KEYS)
    if invalid_keys:
        joined = ", ".join(invalid_keys)
        return {}, f"{label} contains unsupported keys: {joined}."
    sanitized = sanitize_config(data)
    if not sanitized:
        return {}, f"{label} must include valid strategy config keys."
    return sanitized, None


_CANDLE_TIMEFRAMES_SECONDS = {
    "1M": 60,
    "5M": 5 * 60,
    "15M": 15 * 60,
    "30M": 30 * 60,
    "1H": 60 * 60,
    "4H": 4 * 60 * 60,
    "1D": 24 * 60 * 60,
}


def _infer_source_seconds(candles: list[Any]) -> int | None:
    if len(candles) < 2:
        return None
    deltas: list[int] = []
    for idx in range(1, len(candles)):
        delta = int((candles[idx].time_utc - candles[idx - 1].time_utc).total_seconds())
        if delta > 0:
            deltas.append(delta)
    if not deltas:
        return None
    return Counter(deltas).most_common(1)[0][0]


ACCOUNT_SETTINGS_BOOL_FIELDS = {
    "ai_enabled",
    "use_real_balance",
    "enable_drawdown_limit",
    "enable_consecutive_loss",
    "enable_max_daily_losses",
    "enable_profit_target",
    "enable_break_even",
    "enable_tp3",
    "london_enabled",
    "ny_enabled",
    "notify_trade_open",
    "notify_tp_hit",
    "notify_sl_hit",
    "notify_limit_hit",
}
ACCOUNT_SETTINGS_FLOAT_FIELDS = {
    "risk_per_trade",
    "balance_override",
    "min_lot",
    "max_lot",
    "max_drawdown_percent",
    "profit_target_percent",
}
ACCOUNT_SETTINGS_INT_FIELDS = {
    "max_consecutive_losses",
    "max_daily_losses",
    "sl_lookback",
    "sl_buffer",
    "be_buffer",
    "tp1_percent",
    "tp2_percent",
    "tp3_percent",
}


def _resolve_metaapi_token(user_config: dict[str, Any]) -> str | None:
    token = str(user_config.get("metaapi_token") or "").strip()
    if token:
        return token
    if METAAPI_TOKEN:
        return METAAPI_TOKEN
    return None


def _parse_settings_payload(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    if raw is None:
        return None, None
    if isinstance(raw, dict):
        return raw, None
    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            return None, None
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None, "settings must be a JSON object."
        if not isinstance(parsed, dict):
            return None, "settings must be a JSON object."
        return parsed, None
    return None, "settings must be a JSON object."


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp = timestamp / 1000.0
        return datetime.utcfromtimestamp(timestamp)
    if isinstance(value, str):
        parsed = parse_datetime(value)
        if parsed:
            return parsed
    return None


def _extract_float(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = payload.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _order_action(direction: str, order_type: str) -> str | None:
    direction = direction.upper()
    order_type = order_type.upper()
    if direction not in {"BUY", "SELL"}:
        return None
    if order_type == "MARKET":
        return "ORDER_TYPE_BUY" if direction == "BUY" else "ORDER_TYPE_SELL"
    if order_type == "LIMIT":
        return "ORDER_TYPE_BUY_LIMIT" if direction == "BUY" else "ORDER_TYPE_SELL_LIMIT"
    if order_type == "STOP":
        return "ORDER_TYPE_BUY_STOP" if direction == "BUY" else "ORDER_TYPE_SELL_STOP"
    return None




def _trade_exists(session, account_id: int, ticket_id: int | None, signature: dict[str, Any]) -> bool:
    if ticket_id is not None:
        exists = (
            session.query(Trade)
            .filter_by(account_id=account_id, mt_ticket_id=ticket_id)
            .first()
        )
        if exists:
            return True
    entry_time = signature.get("entry_time")
    entry_price = signature.get("entry_price")
    direction = signature.get("direction")
    symbol = signature.get("symbol")
    if entry_time and entry_price and direction and symbol:
        exists = (
            session.query(Trade)
            .filter_by(
                account_id=account_id,
                entry_time=entry_time,
                entry_price=entry_price,
                direction=direction,
                symbol=symbol,
            )
            .first()
        )
        return exists is not None
    return False


def _ingest_metaapi_trades(
    session, account: Mt5Account, trades: list[dict[str, Any]]
) -> dict[str, int]:
    created = 0
    skipped = 0
    for raw in trades:
        symbol = str(raw.get("symbol") or raw.get("instrument") or "").strip()
        side = str(raw.get("type") or raw.get("side") or raw.get("direction") or "").upper()
        if "BUY" in side:
            direction = "BUY"
        elif "SELL" in side:
            direction = "SELL"
        else:
            direction = None
        entry_price = _extract_float(raw, ("entryPrice", "openPrice", "price", "entry_price"))
        stop_loss = _extract_float(raw, ("stopLoss", "sl", "stop_loss", "stopLossPrice"))
        volume = _extract_float(raw, ("volume", "lots", "size", "positionSize"))
        entry_time = _coerce_datetime(
            raw.get("entryTime") or raw.get("openTime") or raw.get("time") or raw.get("createdAt")
        )
        exit_time = _coerce_datetime(raw.get("closeTime") or raw.get("exitTime") or raw.get("updatedAt"))
        ticket_id = raw.get("id") or raw.get("ticket") or raw.get("positionId") or raw.get("orderId")
        try:
            ticket_id = int(ticket_id) if ticket_id is not None else None
        except (TypeError, ValueError):
            ticket_id = None
        if not (symbol and direction and entry_price is not None and stop_loss is not None and volume is not None and entry_time):
            skipped += 1
            continue
        signature = {
            "symbol": symbol,
            "direction": direction,
            "entry_price": float(entry_price),
            "entry_time": entry_time,
        }
        if _trade_exists(session, account.id, ticket_id, signature):
            skipped += 1
            continue
        pnl = _extract_float(raw, ("profit", "pnl"))
        r_multiple = _extract_float(raw, ("rMultiple", "r_multiple"))
        pips = _extract_float(raw, ("pips",))
        tp1 = _extract_float(raw, ("takeProfit", "tp", "tp1"))
        trade = Trade(
            account_id=account.id,
            mt_ticket_id=ticket_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=_extract_float(raw, ("closePrice", "exitPrice")),
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=_extract_float(raw, ("tp2",)),
            take_profit_3=_extract_float(raw, ("tp3",)),
            position_size=volume,
            risk_amount=_extract_float(raw, ("riskAmount", "risk_amount")),
            pips=pips,
            pnl=pnl,
            r_multiple=r_multiple,
            outcome=raw.get("outcome"),
            balance_before=_extract_float(raw, ("balanceBefore",)),
            balance_after=_extract_float(raw, ("balanceAfter",)),
            session=str(raw.get("session") or "").upper() or None,
            entry_time=entry_time,
            exit_time=exit_time,
            duration_seconds=(
                int((exit_time - entry_time).total_seconds())
                if exit_time and entry_time
                else None
            ),
            model_used=account.settings.selected_model if account.settings else None,
            model_tag=account.trade_tag,
            model_magic=account.magic_number,
            is_live=True,
        )
        session.add(trade)
        created += 1
    return {"created": created, "skipped": skipped}

def _apply_account_settings(settings: AccountSettings, payload: dict[str, Any]) -> None:
    for key, value in payload.items():
        if key in ACCOUNT_SETTINGS_BOOL_FIELDS:
            setattr(settings, key, bool(value))
        elif key in ACCOUNT_SETTINGS_FLOAT_FIELDS:
            try:
                setattr(settings, key, float(value))
            except (TypeError, ValueError):
                continue
        elif key in ACCOUNT_SETTINGS_INT_FIELDS:
            try:
                setattr(settings, key, int(value))
            except (TypeError, ValueError):
                continue
        elif key == "selected_model":
            if value is None:
                settings.selected_model = None
            else:
                settings.selected_model = str(value).strip() or None


def _issue_tokens(session, user: User) -> dict[str, str]:
    refresh_token = secrets.token_urlsafe(32)
    session.add(ApiToken(user_id=user.id, token=refresh_token))
    access_token = create_access_token(user.id)
    return {"access_token": access_token, "refresh_token": refresh_token}


def _ensure_default_models(session) -> None:
    defaults = [
        {
            "id": "aggressive",
            "name": "Aggressive",
            "badge": "High Frequency",
            "description": "Optimized for frequent entries with minimal filtering.",
            "entry_style": "Fast confirmation",
            "filters": "Minimal",
            "tp_strategy": "TP1 then TP2",
            "risk_description": "0.5-1% per trade",
            "trading_description": "London/NY sessions",
            "best_for": "Active traders",
        },
        {
            "id": "passive",
            "name": "Passive",
            "badge": "Conservative",
            "description": "Selective entries with stricter filters.",
            "entry_style": "Conservative confirmation",
            "filters": "Higher quality",
            "tp_strategy": "TP1 then TP2",
            "risk_description": "0.5% per trade",
            "trading_description": "London session focus",
            "best_for": "Lower frequency",
        },
    ]
    existing = {model.id for model in session.query(TradingModel).all()}
    for item in defaults:
        if item["id"] in existing:
            continue
        session.add(
            TradingModel(
                id=item["id"],
                name=item["name"],
                badge=item["badge"],
                description=item["description"],
                entry_style=item["entry_style"],
                filters=item["filters"],
                tp_strategy=item["tp_strategy"],
                risk_description=item["risk_description"],
                trading_description=item["trading_description"],
                best_for=item["best_for"],
            )
        )


@api.route("/api/signup", methods=["POST"])
def api_signup() -> Response:
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", "")).strip()
    if not email or not password:
        return json_error("Email and password are required.", 400)
    if len(password) < PASSWORD_MIN_LENGTH:
        return json_error(f"Password must be at least {PASSWORD_MIN_LENGTH} characters.", 400)
    with db_session() as session:
        if session.query(User).filter_by(email=email).first():
            return json_error("Email already registered.", 409)
        user = User(
            email=email,
            password_hash=_hash_password(password),
            email_verified=False,
        )
        session.add(user)
        verify_token = _issue_email_verification(user)
        session.flush()
        _send_verification_email(email, verify_token)
        tokens = _issue_tokens(session, user)
        save_user_config(session, user, {})
        return json_response(
            {"access_token": tokens["access_token"], "refresh_token": tokens["refresh_token"], "user": serialize_user(user)},
            201,
        )


@api.route("/api/login", methods=["POST"])
def api_login() -> Response:
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", "")).strip()
    if not email or not password:
        return json_error("Email and password are required.", 400)
    with db_session() as session:
        user = session.query(User).filter_by(email=email).first()
        if not user:
            return json_error("Invalid credentials.", 401)
        verified = _check_password(user.password_hash, password)
        if verified is None:
            return json_error("Password reset required.", 403)
        if not verified:
            return json_error("Invalid credentials.", 401)
        tokens = _issue_tokens(session, user)
        return json_response(
            {"access_token": tokens["access_token"], "refresh_token": tokens["refresh_token"], "user": serialize_user(user)}
        )


@api.route("/api/logout", methods=["POST"])
def api_logout() -> Response:
    with db_session() as session:
        data = request.get_json(silent=True) or {}
        token = data.get("refresh_token") or bearer_token()
        if not token:
            return json_response({"ok": True})
        token_row = session.query(ApiToken).filter_by(token=token).first()
        if token_row:
            session.delete(token_row)
        return json_response({"ok": True})


@api.route("/api/me", methods=["GET"])
def api_me() -> Response:
    with db_session() as session:
        user = require_token(session)
        if not user:
            return json_error("Unauthorized.", 401)
        return json_response(serialize_user(user))


@api.route("/auth/signup", methods=["POST"])
def auth_signup() -> Response:
    return api_signup()


@api.route("/auth/login", methods=["POST"])
def auth_login() -> Response:
    return api_login()


@api.route("/auth/logout", methods=["POST"])
def auth_logout() -> Response:
    return api_logout()


@api.route("/auth/refresh", methods=["POST"])
def auth_refresh() -> Response:
    with db_session() as session:
        data = request.get_json(silent=True) or {}
        token = data.get("refresh_token") or bearer_token()
        if not token:
            return json_error("Missing refresh token.", 401)
        token_row = session.query(ApiToken).filter_by(token=token).first()
        if not token_row:
            return json_error("Invalid refresh token.", 401)
        user = token_row.user
        new_tokens = _issue_tokens(session, user)
        session.delete(token_row)
        return json_response(
            {
                "access_token": new_tokens["access_token"],
                "refresh_token": new_tokens["refresh_token"],
                "user": serialize_user(user),
            }
        )


@api.route("/auth/forgot-password", methods=["POST"])
def auth_forgot_password() -> Response:
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    if not email:
        return json_error("Email is required.", 400)
    with db_session() as session:
        user = session.query(User).filter_by(email=email).first()
        if user:
            reset_token, reset_hash = generate_token()
            user.reset_token_hash = reset_hash
            user.reset_token_expires_at = datetime.utcnow() + timedelta(
                hours=PASSWORD_RESET_TTL_HOURS
            )
            user.reset_token_sent_at = datetime.utcnow()
            send_email(
                email,
                "Reset your AlgoTrade AI password",
                f"Use this reset code to set a new password: {reset_token}",
            )
    return json_response({"ok": True})


@api.route("/auth/reset-password", methods=["POST"])
def auth_reset_password() -> Response:
    data = request.get_json(silent=True) or {}
    token = str(data.get("token", "")).strip()
    password = str(data.get("password", "")).strip()
    if not token or not password:
        return json_error("Token and password are required.", 400)
    if len(password) < PASSWORD_MIN_LENGTH:
        return json_error(f"Password must be at least {PASSWORD_MIN_LENGTH} characters.", 400)
    token_hash = hash_token(token)
    with db_session() as session:
        user = session.query(User).filter_by(reset_token_hash=token_hash).first()
        if not user:
            return json_error("Invalid or expired reset token.", 400)
        if user.reset_token_expires_at and user.reset_token_expires_at < datetime.utcnow():
            return json_error("Reset token has expired.", 400)
        user.password_hash = _hash_password(password)
        user.reset_token_hash = None
        user.reset_token_expires_at = None
        user.reset_token_sent_at = None
        session.query(ApiToken).filter_by(user_id=user.id).delete()
        return json_response({"ok": True})


@api.route("/auth/verify-email", methods=["POST"])
def auth_verify_email() -> Response:
    with db_session() as session:
        data = request.get_json(silent=True) or {}
        token = str(data.get("token", "")).strip()
        user = None
        if token:
            token_hash = hash_token(token)
            user = session.query(User).filter_by(email_verify_token_hash=token_hash).first()
            if not user:
                return json_error("Invalid verification token.", 400)
            if user.email_verify_expires_at and user.email_verify_expires_at < datetime.utcnow():
                return json_error("Verification token expired.", 400)
        else:
            user = require_token(session)
            if not user:
                return json_error("Unauthorized.", 401)
            if user.email_verified:
                return json_response(
                    {"ok": True, "message": "Email already verified.", "user": serialize_user(user)}
                )
            verify_token = _issue_email_verification(user)
            _send_verification_email(user.email, verify_token)
            return json_response(
                {"ok": True, "message": "Verification email sent.", "user": serialize_user(user)}
            )
        user.email_verified = True
        user.email_verify_token_hash = None
        user.email_verify_expires_at = None
        user.email_verify_sent_at = None
        return json_response({"ok": True, "user": serialize_user(user)})


@api.route("/auth/me", methods=["GET", "PATCH"])
def auth_me() -> Response:
    with db_session() as session:
        user = require_token(session)
        if not user:
            return json_error("Unauthorized.", 401)
        if request.method == "GET":
            return json_response(serialize_user(user))
        payload = request.get_json(silent=True) or {}
        if "full_name" in payload:
            user.full_name = str(payload.get("full_name") or "").strip() or None
        if "avatar_url" in payload:
            user.avatar_url = str(payload.get("avatar_url") or "").strip() or None
        if "timezone" in payload:
            user.timezone = str(payload.get("timezone") or "UTC").strip() or "UTC"
        return json_response(serialize_user(user))


@api.route("/api/config", methods=["GET", "PUT"])
def api_config() -> Response:
    with db_session() as session:
        user = require_token(session)
        if not user:
            return json_error("Unauthorized.", 401)
        if request.method == "GET":
            return json_response({"config": load_user_config(session, user)})
        payload = request.get_json(silent=True) or {}
        sanitized, error = _validate_strategy_settings(payload, "config")
        if error:
            return json_error(error, 400)
        updated = save_user_config(session, user, sanitized)
        return json_response({"config": updated})


@api.route("/api/accounts", methods=["GET"])
def api_accounts() -> Response:
    with db_session() as session:
        user = require_token(session)
        if not user:
            return json_error("Unauthorized.", 401)
        accounts = session.query(Mt5Account).filter_by(user_id=user.id).all()
        return json_response({"accounts": [serialize_account(acc) for acc in accounts]})


@api.route("/api/accounts/connect", methods=["POST"])
def api_accounts_connect() -> Response:
    with db_session() as session:
        user = require_token(session)
        if not user:
            return json_error("Unauthorized.", 401)
        data = request.get_json(silent=True) or {}
        required = ["platform", "type", "broker", "account_number", "server", "password"]
        if any(not data.get(field) for field in required):
            return json_error("Missing required account fields.", 400)
        account_type = data.get("type") or data.get("account_type")
        magic_number = data.get("magic_number")
        if magic_number is not None:
            try:
                magic_number = int(magic_number)
            except (TypeError, ValueError):
                magic_number = None
        account = Mt5Account(
            user_id=user.id,
            nickname=str(data.get("nickname") or "").strip() or None,
            account_number=str(data.get("account_number")).strip(),
            platform=str(data.get("platform")).strip(),
            account_type=str(account_type).strip(),
            broker=str(data.get("broker")).strip(),
            server=str(data.get("server")).strip(),
            password_encrypted=str(data.get("password")).strip(),
            metaapi_account_id=str(data.get("metaapi_account_id") or "").strip() or None,
            metaapi_region=str(data.get("metaapi_region") or "").strip() or None,
            trade_tag=str(data.get("trade_tag") or "").strip() or None,
            magic_number=magic_number,
            status="connected",
        )
        session.add(account)
        session.flush()
        session.add(AccountSettings(account_id=account.id))
        provision_metaapi = bool(
            data.get("provision_metaapi") or data.get("metaapi_provision")
        )
        if provision_metaapi and not account.metaapi_account_id:
            user_config = load_user_config(session, user)
            token = _resolve_metaapi_token(user_config)
            if not token:
                return json_error("MetaApi token missing.", 400)
            client = MetaApiClient(token, region=account.metaapi_region)
            payload = data.get("metaapi_payload") or {
                "name": f"{user.email}-{account.account_number}",
                "type": "cloud",
                "login": account.account_number,
                "password": data.get("password"),
                "server": account.server,
                "platform": account.platform,
            }
            try:
                response = client.create_account(payload)
                account_id = (
                    response.get("id")
                    or response.get("accountId")
                    or response.get("account_id")
                )
                if not account_id:
                    raise MetaApiError("MetaApi account ID not returned.")
                account.metaapi_account_id = str(account_id)
                if bool(data.get("deploy_metaapi", True)):
                    client.deploy_account(account.metaapi_account_id)
            except MetaApiError as exc:
                account.status = "error"
                account.error_message = str(exc)
                return json_error(str(exc), 400)
        return json_response({"success": True, "account": serialize_account(account)})


@api.route("/api/accounts/<int:account_id>/settings", methods=["PATCH"])
def api_account_settings(account_id: int) -> Response:
    with db_session() as session:
        user = require_token(session)
        if not user:
            return json_error("Unauthorized.", 401)
        account = (
            session.query(Mt5Account)
            .filter_by(id=account_id, user_id=user.id)
            .first()
        )
        if not account:
            return json_error("Account not found.", 404)
        payload = request.get_json(silent=True) or {}
        if account.settings is None:
            account.settings = AccountSettings(account_id=account.id)
        if "selected_model" in payload:
            current_model = account.settings.selected_model
            new_model_raw = payload.get("selected_model")
            new_model = None if new_model_raw is None else str(new_model_raw).strip()
            if new_model == "":
                new_model = None
            if current_model and new_model and new_model != current_model:
                return json_error(
                    f"Account already assigned to model '{current_model}'. Clear it first.",
                    409,
                )
        _apply_account_settings(account.settings, payload)
        return json_response({"account": serialize_account(account)})


@api.route("/api/accounts/<int:account_id>", methods=["DELETE"])
def api_account_delete(account_id: int) -> Response:
    with db_session() as session:
        user = require_token(session)
        if not user:
            return json_error("Unauthorized.", 401)
        account = (
            session.query(Mt5Account)
            .filter_by(id=account_id, user_id=user.id)
            .first()
        )
        if not account:
            return json_error("Account not found.", 404)
        session.delete(account)
        return json_response({"ok": True})


@api.route("/api/accounts/<int:account_id>/sync", methods=["POST"])
def api_account_sync(account_id: int) -> Response:
    with db_session() as session:
        user = require_token(session)
        if not user:
            return json_error("Unauthorized.", 401)
        payload = request.get_json(silent=True) or {}
        account = (
            session.query(Mt5Account)
            .filter_by(id=account_id, user_id=user.id)
            .first()
        )
        if not account:
            return json_error("Account not found.", 404)
        user_config = load_user_config(session, user)
        token = _resolve_metaapi_token(user_config)
        if not token:
            return json_error("MetaApi token missing.", 400)
        if not account.metaapi_account_id:
            return json_error("MetaApi account ID missing.", 400)
        client = MetaApiClient(token, region=account.metaapi_region)
        try:
            info = client.get_account_information(account.metaapi_account_id)
            state = client.get_account_state(account.metaapi_account_id)
        except MetaApiError as exc:
            account.status = "error"
            account.error_message = str(exc)
            return json_response({"account": serialize_account(account), "error": str(exc)}, 400)
        account.balance = info.get("balance", account.balance)
        account.equity = info.get("equity", account.equity)
        account.margin = info.get("margin", account.margin)
        account.free_margin = info.get("freeMargin", account.free_margin)
        account.leverage = info.get("leverage", account.leverage)
        account.currency = info.get("currency", account.currency)
        account.status = "connected"
        account.error_message = None
        previous_sync = account.last_sync_at
        now = datetime.utcnow()
        sync_trades = bool(
            payload.get("sync_trades") or request.args.get("sync_trades") == "1"
        )
        manage_trades = bool(
            payload.get("manage_trades") or request.args.get("manage_trades") == "1"
        )
        trade_summary = None
        if sync_trades:
            start_time = payload.get("start_time") or None
            end_time = payload.get("end_time") or None
            if not start_time and previous_sync:
                start_dt = previous_sync - timedelta(days=METAAPI_SYNC_LOOKBACK_DAYS)
                start_time = start_dt.isoformat() + "Z"
            try:
                deals = client.get_deals(
                    account.metaapi_account_id, start_time=start_time, end_time=end_time
                )
                trade_summary = _ingest_metaapi_trades(session, account, deals)
            except MetaApiError as exc:
                account.status = "error"
                account.error_message = str(exc)
                return json_response(
                    {"account": serialize_account(account), "error": str(exc)}, 400
                )
        trade_actions = None
        if manage_trades:
            trade_actions = manage_break_even_for_account(session, account, client)
        account.last_sync_at = now
        payload = {"account": serialize_account(account)}
        if trade_summary is not None:
            payload["trades"] = trade_summary
        if trade_actions is not None:
            payload["trade_actions"] = trade_actions
        return json_response(payload)


@api.route("/api/accounts/<int:account_id>/orders", methods=["POST"])
def api_account_place_order(account_id: int) -> Response:
    with db_session() as session:
        user = require_token(session)
        if not user:
            return json_error("Unauthorized.", 401)
        account = (
            session.query(Mt5Account)
            .filter_by(id=account_id, user_id=user.id)
            .first()
        )
        if not account:
            return json_error("Account not found.", 404)
        user_config = load_user_config(session, user)
        token = _resolve_metaapi_token(user_config)
        if not token:
            return json_error("MetaApi token missing.", 400)
        if not account.metaapi_account_id:
            return json_error("MetaApi account ID missing.", 400)

        data = request.get_json(silent=True) or {}
        symbol = str(data.get("symbol") or "").strip().upper()
        direction = str(data.get("direction") or "").strip().upper()
        order_type = str(data.get("order_type") or "LIMIT").strip().upper()
        volume = _extract_float(data, ("volume", "lots", "size"))
        entry_price = _extract_float(data, ("entry_price", "price"))
        stop_loss = _extract_float(data, ("stop_loss", "sl"))
        tp1_price = _extract_float(data, ("tp1_price", "tp1", "take_profit", "tp"))
        tp2_price = _extract_float(data, ("tp2_price", "tp2"))
        tp3_price = _extract_float(data, ("tp3_price", "tp3"))
        tp1_percent = _extract_float(data, ("tp1_percent",))
        if not symbol or not direction or volume is None:
            return json_error("symbol, direction, and volume are required.", 400)
        action_type = _order_action(direction, order_type)
        if not action_type:
            return json_error("Invalid order_type or direction.", 400)
        if order_type in {"LIMIT", "STOP"} and entry_price is None:
            return json_error("entry_price is required for limit/stop orders.", 400)

        duplicate = (
            session.query(Trade)
            .filter_by(
                account_id=account.id,
                symbol=symbol,
                direction=direction,
                entry_price=entry_price if entry_price is not None else 0,
                exit_time=None,
            )
            .first()
        )
        if duplicate:
            return json_error("Duplicate open trade detected.", 409)

        payload: dict[str, Any] = {
            "actionType": action_type,
            "type": action_type,
            "symbol": symbol,
            "volume": volume,
        }
        if entry_price is not None:
            payload["price"] = entry_price
        if stop_loss is not None:
            payload["stopLoss"] = stop_loss
        final_tp = tp3_price or tp2_price or tp1_price
        if final_tp is not None:
            payload["takeProfit"] = final_tp
        comment = str(data.get("comment") or account.trade_tag or "").strip()
        if comment:
            payload["comment"] = comment
        if account.magic_number:
            payload["magic"] = account.magic_number

        client = MetaApiClient(token, region=account.metaapi_region)
        try:
            response = client.place_order(account.metaapi_account_id, payload)
        except MetaApiError as exc:
            return json_error(str(exc), 400)

        ticket_id = response.get("orderId") or response.get("positionId") or response.get("id")
        try:
            ticket_id = int(ticket_id) if ticket_id is not None else None
        except (TypeError, ValueError):
            ticket_id = None

        settings = account.settings
        meta = {
            "tp1_percent": (
                tp1_percent
                if tp1_percent is not None
                else float(settings.tp1_percent) if settings else 50.0
            ),
            "tp2_percent": float(settings.tp2_percent) if settings else None,
            "tp3_percent": float(settings.tp3_percent) if settings else None,
            "enable_break_even": bool(settings.enable_break_even) if settings else False,
            "be_buffer_pips": float(settings.be_buffer) if settings else 0.0,
        }
        trade = Trade(
            account_id=account.id,
            mt_ticket_id=ticket_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price or 0,
            stop_loss=stop_loss or 0,
            take_profit_1=tp1_price,
            take_profit_2=tp2_price,
            take_profit_3=tp3_price,
            position_size=volume,
            outcome="PENDING",
            entry_time=datetime.utcnow(),
            model_used=account.settings.selected_model if account.settings else None,
            model_tag=account.trade_tag,
            model_magic=account.magic_number,
            is_live=True,
            entry_reasoning=_dump_trade_meta(meta),
        )
        session.add(trade)
        session.flush()
        note = create_notification(
            session,
            user.id,
            "trade_executed",
            "Trade order submitted",
            f"{symbol} {direction} {order_type} submitted.",
            {"trade_id": trade.id, "symbol": symbol, "direction": direction},
        )
        emit_notification(user.id, serialize_notification(note))
        return json_response({"order": response, "trade": serialize_trade(trade)}, 201)


@api.route("/api/notifications", methods=["GET"])
def api_notifications() -> Response:
    with db_session() as session:
        user = require_token(session)
        if not user:
            return json_error("Unauthorized.", 401)
        query = session.query(Notification).filter_by(user_id=user.id)
        read_filter = request.args.get("read")
        if read_filter in {"true", "false"}:
            query = query.filter(Notification.read.is_(read_filter == "true"))
        limit = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))
        total = query.count()
        notes = (
            query.order_by(Notification.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return json_response(
            {
                "notifications": [serialize_notification(note) for note in notes],
                "pagination": {"total": total, "limit": limit, "offset": offset},
            }
        )


@api.route("/api/notifications/<int:notification_id>/read", methods=["PATCH"])
def api_notification_read(notification_id: int) -> Response:
    with db_session() as session:
        user = require_token(session)
        if not user:
            return json_error("Unauthorized.", 401)
        note = (
            session.query(Notification)
            .filter_by(id=notification_id, user_id=user.id)
            .first()
        )
        if not note:
            return json_error("Notification not found.", 404)
        note.read = True
        return json_response({"notification": serialize_notification(note)})


@api.route("/api/notifications/read-all", methods=["POST"])
def api_notifications_read_all() -> Response:
    with db_session() as session:
        user = require_token(session)
        if not user:
            return json_error("Unauthorized.", 401)
        session.query(Notification).filter_by(user_id=user.id).update({"read": True})
        return json_response({"ok": True})


@api.route("/api/accounts/<int:account_id>/trades", methods=["GET"])
def api_account_trades(account_id: int) -> Response:
    with db_session() as session:
        user = require_token(session)
        if not user:
            return json_error("Unauthorized.", 401)
        account = (
            session.query(Mt5Account)
            .filter_by(id=account_id, user_id=user.id)
            .first()
        )
        if not account:
            return json_error("Account not found.", 404)
        query = session.query(Trade).filter_by(account_id=account.id)
        direction = request.args.get("direction")
        outcome = request.args.get("outcome")
        session_filter = request.args.get("session")
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        if direction:
            query = query.filter(Trade.direction == direction)
        if outcome:
            query = query.filter(Trade.outcome == outcome)
        if session_filter:
            query = query.filter(Trade.session == session_filter)
        start_dt = parse_datetime(start_date)
        end_dt = parse_datetime(end_date)
        if start_dt:
            query = query.filter(Trade.entry_time >= start_dt)
        if end_dt:
            query = query.filter(Trade.entry_time <= end_dt)
        limit = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))
        total = query.count()
        trades = query.order_by(Trade.entry_time.desc()).offset(offset).limit(limit).all()
        total_pnl = sum(float(t.pnl or 0) for t in trades)
        total_pips = sum(float(t.pips or 0) for t in trades)
        avg_r = (
            sum(float(t.r_multiple or 0) for t in trades) / len(trades)
            if trades
            else 0
        )
        return json_response(
            {
                "trades": [serialize_trade(trade) for trade in trades],
                "pagination": {"total": total, "limit": limit, "offset": offset},
                "summary": {"total_pnl": total_pnl, "total_pips": total_pips, "avg_r": avg_r},
            }
        )


@api.route("/api/trades/<int:trade_id>", methods=["GET"])
def api_trade_detail(trade_id: int) -> Response:
    with db_session() as session:
        user = require_token(session)
        if not user:
            return json_error("Unauthorized.", 401)
        trade = (
            session.query(Trade)
            .join(Mt5Account, Trade.account_id == Mt5Account.id)
            .filter(Trade.id == trade_id, Mt5Account.user_id == user.id)
            .first()
        )
        if not trade:
            return json_error("Trade not found.", 404)
        return json_response(serialize_trade(trade))


@api.route("/api/accounts/<int:account_id>/stats", methods=["GET"])
def api_account_stats(account_id: int) -> Response:
    with db_session() as session:
        user = require_token(session)
        if not user:
            return json_error("Unauthorized.", 401)
        account = (
            session.query(Mt5Account)
            .filter_by(id=account_id, user_id=user.id)
            .first()
        )
        if not account:
            return json_error("Account not found.", 404)
        trades = session.query(Trade).filter_by(account_id=account.id).all()
        total_trades = len(trades)
        wins = [t for t in trades if t.pnl and t.pnl > 0]
        losses = [t for t in trades if t.pnl and t.pnl < 0]
        net_pnl = sum(float(t.pnl or 0) for t in trades)
        starting_balance = float(account.balance or 0)
        win_rate = (len(wins) / total_trades * 100) if total_trades else 0
        gross_profit = sum(float(t.pnl or 0) for t in wins)
        gross_loss = abs(sum(float(t.pnl or 0) for t in losses))
        avg_win = (gross_profit / len(wins)) if wins else 0
        avg_loss = (gross_loss / len(losses)) if losses else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss else None
        durations = [t.duration_seconds for t in trades if t.duration_seconds]
        avg_duration_minutes = int(sum(durations) / len(durations) / 60) if durations else 0
        today = datetime.utcnow().date()
        trades_today = [t for t in trades if t.entry_time and t.entry_time.date() == today]
        pnl_today = sum(float(t.pnl or 0) for t in trades_today)
        return json_response(
            {
                "period": "all_time",
                "balance": float(account.balance or 0),
                "starting_balance": starting_balance,
                "net_pnl": net_pnl,
                "net_pnl_percent": (net_pnl / starting_balance * 100) if starting_balance else 0,
                "total_trades": total_trades,
                "winning_trades": len(wins),
                "losing_trades": len(losses),
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "max_drawdown": None,
                "avg_trade_duration_minutes": avg_duration_minutes,
                "trades_today": len(trades_today),
                "pnl_today": pnl_today,
            }
        )


@api.route("/api/models", methods=["GET"])
def api_models() -> Response:
    with db_session() as session:
        _ensure_default_models(session)
        models = session.query(TradingModel).all()
        return json_response(
            {
                "models": [
                    {
                        "id": model.id,
                        "name": model.name,
                        "badge": model.badge,
                        "description": model.description,
                        "characteristics": {
                            "entry_style": model.entry_style,
                            "filters": model.filters,
                            "tp_strategy": model.tp_strategy,
                            "risk": model.risk_description,
                            "trading": model.trading_description,
                            "best_for": model.best_for,
                        },
                        "six_month_performance": None,
                        "monthly_returns": [],
                        "active_accounts": model.active_accounts,
                    }
                    for model in models
                ]
            }
        )


@api.route("/api/models/<model_id>", methods=["GET"])
def api_model_detail(model_id: str) -> Response:
    with db_session() as session:
        _ensure_default_models(session)
        model = session.query(TradingModel).filter_by(id=model_id).first()
        if not model:
            return json_error("Model not found.", 404)
        return json_response(
            {
                "id": model.id,
                "name": model.name,
                "badge": model.badge,
                "description": model.description,
                "characteristics": {
                    "entry_style": model.entry_style,
                    "filters": model.filters,
                    "tp_strategy": model.tp_strategy,
                    "risk": model.risk_description,
                    "trading": model.trading_description,
                    "best_for": model.best_for,
                },
                "six_month_performance": None,
                "monthly_returns": [],
                "active_accounts": model.active_accounts,
            }
        )


@api.route("/api/models/<model_id>/backtests", methods=["GET"])
def api_model_backtests(model_id: str) -> Response:
    with db_session() as session:
        user = require_token(session)
        if not user:
            return json_error("Unauthorized.", 401)
        backtests = (
            session.query(Backtest)
            .filter_by(user_id=user.id, model=model_id)
            .order_by(Backtest.created_at.desc())
            .all()
        )
        return json_response(
            {"backtests": [serialize_backtest(bt, include_rows=False) for bt in backtests]}
        )


@api.route("/api/models/<model_id>/copy", methods=["POST"])
def api_model_copy(model_id: str) -> Response:
    with db_session() as session:
        user = require_token(session)
        if not user:
            return json_error("Unauthorized.", 401)
        data = request.get_json(silent=True) or {}
        account_id = data.get("account_id")
        if not account_id:
            return json_error("account_id is required.", 400)
        account = (
            session.query(Mt5Account)
            .filter_by(id=account_id, user_id=user.id)
            .first()
        )
        if not account:
            return json_error("Account not found.", 404)
        if account.settings is None:
            account.settings = AccountSettings(account_id=account.id)
        current_model = account.settings.selected_model
        if current_model and current_model != model_id:
            return json_error(
                f"Account already assigned to model '{current_model}'. Clear it first.",
                409,
            )
        account.settings.ai_enabled = True
        account.settings.selected_model = model_id
        return json_response({"ok": True, "account": serialize_account(account)})


@api.route("/api/backtests", methods=["GET", "POST"])
def api_backtests() -> Response:
    with db_session() as session:
        user = require_token(session)
        if not user:
            return json_error("Unauthorized.", 401)
        if request.method == "GET":
            backtests = (
                session.query(Backtest)
                .filter_by(user_id=user.id)
                .order_by(Backtest.created_at.desc())
                .all()
            )
            return json_response(
                {"backtests": [serialize_backtest(bt, include_rows=False) for bt in backtests]}
            )

        payload = request.form.to_dict() if request.form else {}
        if request.is_json:
            payload = request.get_json(silent=True) or payload

        name = payload.get("name")
        model = payload.get("model")
        date_start = parse_date(payload.get("date_start"))
        date_end = parse_date(payload.get("date_end"))
        starting_balance = payload.get("starting_balance")
        symbol = payload.get("symbol")
        settings_raw = payload.get("settings")
        settings, settings_error = _parse_settings_payload(settings_raw)
        if settings_error:
            return json_error(settings_error, 400)
        settings_json = json.dumps(settings) if settings else None

        BACKTEST_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        upload = request.files.get("csv")
        csv_filename = f"{secrets.token_urlsafe(12)}.csv"
        csv_path = BACKTEST_UPLOAD_DIR / csv_filename
        if upload and upload.filename:
            upload.save(csv_path)
        else:
            csv_base64 = payload.get("csv_base64")
            if not csv_base64:
                csv_path.unlink(missing_ok=True)
                return json_error("CSV file is required.", 400)
            try:
                csv_bytes = base64.b64decode(csv_base64)
            except (TypeError, ValueError):
                csv_path.unlink(missing_ok=True)
                return json_error("Invalid csv_base64 payload.", 400)
            csv_path.write_bytes(csv_bytes)

        error = validate_candles(csv_path)
        if error:
            csv_path.unlink(missing_ok=True)
            return json_error(error, 400)

        user_config = load_user_config(session, user, model_mode=model)
        if settings:
            sanitized, error = _validate_strategy_settings(settings, "settings")
            if error:
                return json_error(error, 400)
            user_config.update(sanitized)
        if model:
            user_config["model_mode"] = model
        overrides = build_config_overrides(user_config)
        if starting_balance is not None:
            try:
                overrides["ACCOUNT_BALANCE_OVERRIDE"] = float(starting_balance)
            except (TypeError, ValueError):
                pass

        job_id = create_backtest_job(
            session,
            user.id,
            model or overrides.get("MODEL_MODE"),
            name=name,
            date_start=date_start,
            date_end=date_end,
            starting_balance=overrides.get("ACCOUNT_BALANCE_OVERRIDE"),
            symbol=symbol,
            settings_json=settings_json or json.dumps(user_config),
            csv_path=csv_path,
        )
        if not BACKTEST_WORKER_ENABLED:
            thread = threading.Thread(
                target=run_backtest_job,
                args=(job_id, csv_path, overrides),
                daemon=True,
            )
            thread.start()

        return json_response({"id": job_id, "status": "pending", "progress": 0}, 202)


@api.route("/api/backtests/<backtest_id>", methods=["GET"])
def api_backtest_detail(backtest_id: str) -> Response:
    with db_session() as session:
        user = require_token(session)
        if not user:
            return json_error("Unauthorized.", 401)
        backtest = (
            session.query(Backtest)
            .filter_by(id=backtest_id, user_id=user.id)
            .first()
        )
        if not backtest:
            return json_error("Backtest not found.", 404)
        return json_response(serialize_backtest(backtest))


@api.route("/api/backtests/<backtest_id>/candles", methods=["GET"])
def api_backtest_candles(backtest_id: str) -> Response:
    timeframe = (request.args.get("timeframe") or "").upper()
    target_seconds = _CANDLE_TIMEFRAMES_SECONDS.get(timeframe)
    if not target_seconds:
        allowed = ", ".join(sorted(_CANDLE_TIMEFRAMES_SECONDS))
        return json_error(f"Invalid timeframe. Allowed: {allowed}.", 400)

    with db_session() as session:
        user = require_token(session)
        if not user:
            return json_error("Unauthorized.", 401)
        backtest = (
            session.query(Backtest)
            .filter_by(id=backtest_id, user_id=user.id)
            .first()
        )
        if not backtest:
            return json_error("Backtest not found.", 404)
        if not backtest.csv_path:
            return json_error("Backtest CSV is not available.", 404)
        csv_path = Path(backtest.csv_path)

    if not csv_path.exists():
        return json_error("Backtest CSV is not available.", 404)

    candles = load_csv_candles(csv_path)
    if not candles:
        return json_response({"candles": []})

    source_seconds = _infer_source_seconds(candles)
    if not source_seconds:
        return json_error("Unable to infer CSV candle timeframe.", 400)
    if target_seconds < source_seconds:
        return json_error("Requested timeframe is smaller than CSV timeframe.", 400)
    if target_seconds == source_seconds:
        resampled = candles
    elif target_seconds % source_seconds != 0:
        return json_error("Requested timeframe must be a multiple of CSV timeframe.", 400)
    else:
        resampled = resample_candles(candles, source_seconds, target_seconds)

    payload = [
        {
            "time": int(candle.time_utc.timestamp()),
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": None,
        }
        for candle in resampled
    ]
    return json_response({"candles": payload})


@api.route("/api/backtest", methods=["POST"])
def api_backtest() -> Response:
    with db_session() as session:
        user = require_token(session)
        if not user:
            return json_error("Unauthorized.", 401)

        payload = request.form.to_dict() if request.form else {}
        if request.is_json:
            payload = request.get_json(silent=True) or payload

        upload = request.files.get("csv")
        if not upload or upload.filename == "":
            return json_error("No CSV file uploaded.", 400)

        BACKTEST_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        csv_filename = f"{secrets.token_urlsafe(12)}.csv"
        csv_path = BACKTEST_UPLOAD_DIR / csv_filename
        upload.save(csv_path)

        error = validate_candles(csv_path)
        if error:
            csv_path.unlink(missing_ok=True)
            return json_error(error, 400)

        model = payload.get("model")
        starting_balance = payload.get("starting_balance")
        settings_raw = payload.get("settings")
        settings, settings_error = _parse_settings_payload(settings_raw)
        if settings_error:
            return json_error(settings_error, 400)

        user_config = load_user_config(session, user, model_mode=model)
        if settings:
            sanitized, error = _validate_strategy_settings(settings, "settings")
            if error:
                return json_error(error, 400)
            user_config.update(sanitized)
        if model:
            user_config["model_mode"] = model
        overrides = build_config_overrides(user_config)
        if starting_balance is not None:
            try:
                overrides["ACCOUNT_BALANCE_OVERRIDE"] = float(starting_balance)
            except (TypeError, ValueError):
                pass
        job_id = create_backtest_job(
            session,
            user.id,
            model or overrides.get("MODEL_MODE"),
            starting_balance=overrides.get("ACCOUNT_BALANCE_OVERRIDE"),
            settings_json=json.dumps(user_config),
            csv_path=csv_path,
        )
        if not BACKTEST_WORKER_ENABLED:
            thread = threading.Thread(
                target=run_backtest_job,
                args=(job_id, csv_path, overrides),
                daemon=True,
            )
            thread.start()

        return json_response(
            {"job_id": job_id, "status": "pending", "model_mode": overrides.get("MODEL_MODE")},
            202,
        )


@api.route("/api/backtest/<job_id>", methods=["GET"])
def api_backtest_status(job_id: str) -> Response:
    with db_session() as session:
        user = require_token(session)
        if not user:
            return json_error("Unauthorized.", 401)

        job = (
            session.query(Backtest)
            .filter_by(id=job_id, user_id=user.id)
            .first()
        )
        if not job:
            return json_error("Backtest not found.", 404)
        payload = serialize_backtest(job)
        return json_response(payload)
