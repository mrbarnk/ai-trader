from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from trader import config as trader_config

from .metaapi_client import MetaApiClient, MetaApiError
from .models import Mt5Account, Trade


def _parse_trade_meta(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return meta if isinstance(meta, dict) else {}


def _dump_trade_meta(meta: dict[str, Any]) -> str:
    return json.dumps(meta, separators=(",", ":"))


def _normalize_percent(value: Any, default: float) -> float:
    try:
        percent = float(value)
    except (TypeError, ValueError):
        percent = float(default)
    if percent > 1:
        percent /= 100.0
    if percent < 0:
        return 0.0
    if percent > 1:
        return 1.0
    return percent


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


def _position_direction(position: dict[str, Any]) -> str | None:
    side = str(
        position.get("type")
        or position.get("side")
        or position.get("direction")
        or position.get("orderType")
        or ""
    ).upper()
    if "BUY" in side:
        return "BUY"
    if "SELL" in side:
        return "SELL"
    return None


def _position_id(position: dict[str, Any]) -> str | None:
    for key in ("id", "positionId", "ticket", "orderId"):
        value = position.get(key)
        if value is not None and value != "":
            return str(value)
    return None


def _position_price(position: dict[str, Any]) -> float | None:
    return _extract_float(position, ("openPrice", "entryPrice", "price", "open_price"))


def _position_volume(position: dict[str, Any]) -> float | None:
    return _extract_float(position, ("volume", "lots", "size", "positionSize"))


def _match_position(trade: Trade, positions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if trade.mt_ticket_id is not None:
        for pos in positions:
            pos_id = _position_id(pos)
            if pos_id and pos_id == str(trade.mt_ticket_id):
                return pos
    entry_price = float(trade.entry_price or 0)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for pos in positions:
        pos_symbol = str(pos.get("symbol") or pos.get("instrument") or "").upper()
        if pos_symbol != str(trade.symbol).upper():
            continue
        if _position_direction(pos) != trade.direction:
            continue
        pos_price = _position_price(pos)
        delta = abs((pos_price if pos_price is not None else entry_price) - entry_price)
        candidates.append((delta, pos))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def manage_break_even_for_account(
    session,
    account: Mt5Account,
    client: MetaApiClient,
    price_map: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    settings = account.settings
    if not settings or not settings.enable_break_even:
        return {"checked": 0, "moved_to_be": 0, "skipped": 0}
    try:
        positions = client.get_positions(account.metaapi_account_id)
    except MetaApiError as exc:
        return {"checked": 0, "moved_to_be": 0, "skipped": 0, "error": str(exc)}

    open_trades = (
        session.query(Trade)
        .filter_by(account_id=account.id, exit_time=None, is_live=True)
        .all()
    )
    moved = 0
    checked = 0
    skipped = 0
    for trade in open_trades:
        if trade.take_profit_1 is None:
            skipped += 1
            continue
        meta = _parse_trade_meta(trade.entry_reasoning)
        if meta.get("moved_to_be"):
            skipped += 1
            continue
        if meta.get("enable_break_even") is False:
            skipped += 1
            continue
        position = _match_position(trade, positions)
        if not position:
            skipped += 1
            continue
        tp1_price = float(trade.take_profit_1 or 0)
        bid = ask = None
        if price_map:
            price_pair = price_map.get(str(trade.symbol).upper())
            if price_pair:
                bid, ask = price_pair
        if bid is None or ask is None:
            price_data = client.get_symbol_price(
                account.metaapi_account_id, trade.symbol
            )
            bid = _extract_float(price_data, ("bid", "bidPrice", "price"))
            ask = _extract_float(price_data, ("ask", "askPrice", "price"))
            if bid is None and ask is not None:
                bid = ask
            if ask is None and bid is not None:
                ask = bid
        if bid is None or ask is None:
            skipped += 1
            continue
        if trade.direction == "BUY":
            tp1_hit = bid >= tp1_price
        else:
            tp1_hit = ask <= tp1_price
        if not tp1_hit:
            checked += 1
            continue
        position_volume = _position_volume(position)
        if position_volume is None:
            skipped += 1
            continue
        tp1_fraction = _normalize_percent(
            meta.get("tp1_percent", settings.tp1_percent), settings.tp1_percent
        )
        close_volume = position_volume * tp1_fraction
        min_lot = float(settings.min_lot or 0)
        if min_lot:
            close_volume = max(close_volume, min_lot)
        if close_volume <= 0:
            skipped += 1
            continue
        if close_volume > position_volume:
            close_volume = position_volume
        position_id = _position_id(position)
        if not position_id:
            skipped += 1
            continue
        be_buffer_pips = meta.get("be_buffer_pips", settings.be_buffer)
        buffer_price = float(be_buffer_pips or 0) * trader_config.PIP_SIZE
        entry_price = float(trade.entry_price or 0)
        if trade.direction == "BUY":
            new_sl = entry_price + buffer_price
            if float(trade.stop_loss or 0) >= new_sl:
                meta["moved_to_be"] = True
                trade.entry_reasoning = _dump_trade_meta(meta)
                checked += 1
                continue
        else:
            new_sl = entry_price - buffer_price
            if float(trade.stop_loss or 0) <= new_sl:
                meta["moved_to_be"] = True
                trade.entry_reasoning = _dump_trade_meta(meta)
                checked += 1
                continue
        try:
            client.close_position(account.metaapi_account_id, position_id, close_volume)
            client.modify_position(
                account.metaapi_account_id, position_id, stop_loss=new_sl
            )
        except MetaApiError:
            skipped += 1
            continue
        meta["moved_to_be"] = True
        meta["be_moved_at"] = datetime.utcnow().isoformat() + "Z"
        meta["tp1_percent"] = tp1_fraction
        meta["be_buffer_pips"] = float(be_buffer_pips or 0)
        trade.entry_reasoning = _dump_trade_meta(meta)
        trade.stop_loss = new_sl
        trade.outcome = "TP1_THEN_BE"
        moved += 1
        checked += 1
    return {"checked": checked, "moved_to_be": moved, "skipped": skipped}
