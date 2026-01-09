from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any

from trader import config as trader_config

from .config_overrides import apply_config_overrides, build_config_overrides, load_user_config
from .metaapi_client import MetaApiClient, MetaApiError
from .models import LiveSignal, Mt5Account, Trade, db_session
from .notifications import create_notification, emit_notification, serialize_notification
from .settings import (
    LIVE_TRADE_POLL_SECONDS,
    LIVE_TRADE_WORKER_ENABLED,
    METAAPI_SYNC_LOOKBACK_DAYS,
)

logger = logging.getLogger(__name__)

DEFAULT_LIVE_MODELS = ("aggressive", "passive")
_worker_thread: threading.Thread | None = None
_worker_started = False


def _round_price(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 5)


def _signal_identity(signal) -> tuple | None:
    if signal.decision != "TRADE":
        return None
    return (
        signal.model_mode,
        signal.direction,
        _round_price(signal.entry),
        _round_price(signal.stop_loss),
        _round_price(signal.tp1_price),
        _round_price(signal.tp2_price),
        _round_price(signal.tp3_price),
    )


def _parse_signal_time(timestamp: str | None) -> datetime | None:
    if not timestamp:
        return None
    try:
        return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


def _risk_fraction(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(float(value), 0.0) / 100.0


def _resolve_order_volume(
    entry_price: float,
    stop_loss: float,
    account: Mt5Account,
    balance: float,
) -> float | None:
    stop_distance_pips = abs(entry_price - stop_loss) / trader_config.PIP_SIZE
    if stop_distance_pips <= 0:
        return None
    settings = account.settings
    risk_pct = _risk_fraction(float(settings.risk_per_trade) if settings else None)
    if risk_pct <= 0:
        return None
    risk_amount = balance * risk_pct
    if trader_config.PIP_VALUE_PER_LOT <= 0:
        return None
    raw_lots = risk_amount / (stop_distance_pips * trader_config.PIP_VALUE_PER_LOT)
    if raw_lots <= 0:
        return None
    step = trader_config.LOT_STEP
    stepped = (raw_lots // step) * step
    min_lot = float(settings.min_lot) if settings else trader_config.MIN_LOT_SIZE
    max_lot = float(settings.max_lot) if settings else trader_config.MAX_LOT_SIZE
    if stepped < min_lot or stepped > max_lot:
        return None
    return round(float(stepped), 4)


def _resolve_balance(account: Mt5Account, client: MetaApiClient) -> float | None:
    settings = account.settings
    if settings and not settings.use_real_balance:
        return float(settings.balance_override or 0)
    balance = float(account.balance or 0)
    if balance > 0:
        return balance
    try:
        info = client.get_account_information(account.metaapi_account_id)
    except MetaApiError:
        return None
    balance = float(info.get("balance") or 0)
    if balance > 0:
        account.balance = balance
    return balance if balance > 0 else None


def _order_action(direction: str, order_type: str) -> str | None:
    direction = direction.upper()
    order_type = order_type.upper()
    if direction not in {"BUY", "SELL"}:
        return None
    if order_type == "LIMIT":
        return "ORDER_TYPE_BUY_LIMIT" if direction == "BUY" else "ORDER_TYPE_SELL_LIMIT"
    return None


def _build_signal_record(account: Mt5Account, signal) -> LiveSignal:
    return LiveSignal(
        account_id=account.id,
        model=signal.model_mode,
        decision=signal.decision,
        symbol=str(signal.pair or "").upper(),
        direction=signal.direction,
        entry_price=_round_price(signal.entry) if isinstance(signal.entry, (int, float)) else None,
        stop_loss=_round_price(signal.stop_loss) if signal.stop_loss is not None else None,
        take_profit_1=_round_price(signal.tp1_price) if signal.tp1_price is not None else None,
        take_profit_2=_round_price(signal.tp2_price) if signal.tp2_price is not None else None,
        take_profit_3=_round_price(signal.tp3_price) if signal.tp3_price is not None else None,
        rules_passed=json.dumps(signal.rules_passed),
        rules_failed=json.dumps(signal.rules_failed),
        status="created",
        signal_time=_parse_signal_time(signal.timestamp_utc),
    )


def _is_duplicate_trade(session, account: Mt5Account, signal_key: tuple) -> bool:
    if not signal_key:
        return False
    model, direction, entry, stop, tp1, tp2, tp3 = signal_key
    query = session.query(Trade).filter_by(
        account_id=account.id,
        exit_time=None,
        model_used=model,
        direction=direction,
        entry_price=entry or 0,
        stop_loss=stop or 0,
    )
    if tp1 is not None:
        query = query.filter(Trade.take_profit_1 == tp1)
    if tp2 is not None:
        query = query.filter(Trade.take_profit_2 == tp2)
    if tp3 is not None:
        query = query.filter(Trade.take_profit_3 == tp3)
    return session.query(query.exists()).scalar()


def _sync_account_state(
    session, account: Mt5Account, client: MetaApiClient, now_utc: datetime
) -> None:
    try:
        info = client.get_account_information(account.metaapi_account_id)
    except MetaApiError as exc:
        logger.warning("Account info sync failed for %s: %s", account.id, exc)
        return
    account.balance = info.get("balance", account.balance)
    account.equity = info.get("equity", account.equity)
    account.margin = info.get("margin", account.margin)
    account.free_margin = info.get("freeMargin", account.free_margin)
    account.leverage = info.get("leverage", account.leverage)
    account.currency = info.get("currency", account.currency)
    account.status = "connected"
    account.error_message = None
    start_time = None
    if account.last_sync_at:
        start_dt = account.last_sync_at - timedelta(days=METAAPI_SYNC_LOOKBACK_DAYS)
        start_time = start_dt.isoformat() + "Z"
    try:
        deals = client.get_deals(
            account.metaapi_account_id, start_time=start_time, end_time=None
        )
    except MetaApiError as exc:
        logger.warning("Trade sync failed for %s: %s", account.id, exc)
        account.last_sync_at = now_utc
        return
    try:
        from .routes_api import _ingest_metaapi_trades
    except Exception as exc:
        logger.warning("Trade ingest unavailable: %s", exc)
    else:
        _ingest_metaapi_trades(session, account, deals)
    account.last_sync_at = now_utc


def start_live_trade_worker() -> None:
    global _worker_thread, _worker_started
    if _worker_started or not LIVE_TRADE_WORKER_ENABLED:
        return
    _worker_started = True
    _worker_thread = threading.Thread(
        target=run_worker, name="LiveTradeWorker", daemon=True
    )
    _worker_thread.start()


def run_worker() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        from trader.engine_factory import get_engine
        from trader.live_data import LiveCandleProvider
        from trader.mt5_client import get_tick_time, initialize, resolve_symbol, shutdown
        from trader.time_utils import broker_epoch_to_utc
    except Exception as exc:
        logger.error("Live trade worker missing MT5 dependencies: %s", exc)
        return
    try:
        initialize()
    except Exception as exc:
        logger.error("MT5 initialize failed for live worker: %s", exc)
        return
    symbol = resolve_symbol()
    candle_provider = LiveCandleProvider()
    engine_cache: dict[tuple[int, str], Any] = {}
    last_signal_keys: dict[tuple[int, str], tuple] = {}

    try:
        logger.info("Live trade worker started.")
        while True:
            if not LIVE_TRADE_WORKER_ENABLED:
                time.sleep(LIVE_TRADE_POLL_SECONDS)
                continue
            try:
                tick_time = get_tick_time(symbol)
            except Exception as exc:
                logger.warning("Live worker tick fetch failed: %s", exc)
                time.sleep(LIVE_TRADE_POLL_SECONDS)
                continue
            now_utc = broker_epoch_to_utc(tick_time)
            with db_session() as session:
                synced_accounts: set[int] = set()
                accounts = (
                    session.query(Mt5Account)
                    .filter(Mt5Account.metaapi_account_id.isnot(None))
                    .all()
                )
                for account in accounts:
                    settings = account.settings
                    if not settings or not settings.ai_enabled:
                        continue
                    selected_model = (settings.selected_model or "").lower()
                    models = [selected_model] if selected_model else list(DEFAULT_LIVE_MODELS)
                    for model_mode in models:
                        user_config = load_user_config(session, account.user, model_mode)
                        overrides = build_config_overrides(user_config)
                        overrides["MODEL_MODE"] = model_mode
                        with apply_config_overrides(overrides):
                            engine_key = (account.id, model_mode)
                            engine = engine_cache.get(engine_key)
                            if engine is None:
                                engine = get_engine(
                                    symbol=symbol,
                                    candle_provider=candle_provider,
                                    mode=model_mode,
                                )
                                engine_cache[engine_key] = engine
                            signal = engine.evaluate(now_utc)
                        signal_row = _build_signal_record(account, signal)
                        session.add(signal_row)
                        session.flush()
                        if signal.decision != "TRADE":
                            signal_row.status = "no_trade"
                            continue
                        signal_key = _signal_identity(signal)
                        if signal_key is None:
                            signal_row.status = "invalid_signal"
                            continue
                        last_key = last_signal_keys.get((account.id, model_mode))
                        if last_key == signal_key or _is_duplicate_trade(
                            session, account, signal_key
                        ):
                            signal_row.status = "duplicate"
                            continue
                        token = str(user_config.get("metaapi_token") or "").strip()
                        if not token:
                            signal_row.status = "missing_token"
                            continue
                        client = MetaApiClient(token, region=account.metaapi_region)
                        if account.id not in synced_accounts:
                            _sync_account_state(session, account, client, now_utc)
                            synced_accounts.add(account.id)
                        entry_price = _round_price(signal.entry)
                        stop_loss = _round_price(signal.stop_loss)
                        if entry_price is None or stop_loss is None:
                            signal_row.status = "missing_prices"
                            continue
                        balance = _resolve_balance(account, client)
                        if balance is None:
                            signal_row.status = "balance_unavailable"
                            continue
                        volume = _resolve_order_volume(
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            account=account,
                            balance=balance,
                        )
                        if volume is None:
                            signal_row.status = "volume_invalid"
                            continue
                        order_type = "LIMIT"
                        action_type = _order_action(signal.direction or "", order_type)
                        if not action_type:
                            signal_row.status = "invalid_direction"
                            continue
                        tp_price = signal.tp3_price or signal.tp2_price or signal.tp1_price
                        payload: dict[str, Any] = {
                            "actionType": action_type,
                            "type": action_type,
                            "symbol": symbol,
                            "volume": volume,
                            "price": entry_price,
                            "stopLoss": stop_loss,
                        }
                        if tp_price is not None:
                            payload["takeProfit"] = float(tp_price)
                        comment = str(account.trade_tag or "").strip()
                        if comment:
                            payload["comment"] = comment
                        if account.magic_number:
                            payload["magic"] = int(account.magic_number)
                        try:
                            response = client.place_order(
                                str(account.metaapi_account_id), payload
                            )
                        except MetaApiError as api_exc:
                            signal_row.status = "order_failed"
                            signal_row.error_message = str(api_exc)
                            continue
                        ticket_id = (
                            response.get("orderId")
                            or response.get("positionId")
                            or response.get("id")
                            if response
                            else None
                        )
                        try:
                            ticket_id = int(ticket_id) if ticket_id is not None else None
                        except (TypeError, ValueError):
                            ticket_id = None
                        trade = Trade(
                            account_id=account.id,
                            mt_ticket_id=ticket_id,
                            symbol=symbol,
                            direction=signal.direction or "",
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit_1=signal.tp1_price,
                            take_profit_2=signal.tp2_price,
                            take_profit_3=signal.tp3_price,
                            position_size=volume,
                            risk_amount=balance * _risk_fraction(
                                float(settings.risk_per_trade) if settings else 0
                            ),
                            outcome="PENDING",
                            entry_time=now_utc,
                            model_used=model_mode,
                            model_tag=account.trade_tag,
                            model_magic=account.magic_number,
                            rules_passed=json.dumps(signal.rules_passed),
                            is_live=True,
                            entry_reasoning=json.dumps(
                                {"signal_rules": signal.rules_passed}, separators=(",", ":")
                            ),
                        )
                        session.add(trade)
                        session.flush()
                        signal_row.status = "submitted"
                        signal_row.trade_id = trade.id
                        last_signal_keys[(account.id, model_mode)] = signal_key
                        note = create_notification(
                            session,
                            account.user_id,
                            "trade_executed",
                            "Trade order submitted",
                            f"{symbol} {signal.direction} LIMIT submitted.",
                            {"trade_id": trade.id, "symbol": symbol, "direction": signal.direction},
                        )
                        emit_notification(account.user_id, serialize_notification(note))
            time.sleep(LIVE_TRADE_POLL_SECONDS)
    finally:
        shutdown()


def main() -> None:
    run_worker()


if __name__ == "__main__":
    main()
