from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from metaapi_cloud_sdk import MetaApi

from .config_overrides import load_user_config
from .metaapi_client import MetaApiClient
from .models import Mt5Account, Trade, db_session
from .settings import (
    METAAPI_STREAMING_CHECK_SECONDS,
    METAAPI_STREAMING_ENABLED,
    METAAPI_STREAMING_REFRESH_SECONDS,
    METAAPI_TOKEN,
)
from .trade_management import manage_break_even_for_account


logger = logging.getLogger(__name__)


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_token(session, account: Mt5Account) -> str | None:
    user_config = load_user_config(
        session,
        account.user,
        account.settings.selected_model if account.settings else None,
    )
    token = str(user_config.get("metaapi_token") or "").strip()
    if token:
        return token
    return METAAPI_TOKEN


@dataclass
class AccountStream:
    account_id: int
    metaapi_account_id: str
    token: str
    region: str | None
    refresh_seconds: int
    check_seconds: float
    price_map: dict[str, tuple[float, float]] = field(default_factory=dict)
    subscribed_symbols: set[str] = field(default_factory=set)
    _last_check: float = field(default=0.0)
    _connection: Any | None = field(default=None)
    _client: MetaApiClient | None = field(default=None)

    async def start(self) -> None:
        logger.info("Streaming: connecting account %s", self.account_id)
        self._client = MetaApiClient(self.token, self.region)
        metaapi = MetaApi(self.token)
        account_api = metaapi.metatrader_account_api
        account = await account_api.get_account(self.metaapi_account_id)
        try:
            if getattr(account, "state", "") != "DEPLOYED":
                await account.deploy()
        except Exception:
            pass
        try:
            await account.wait_connected()
        except Exception:
            pass
        connection = account.get_streaming_connection()
        await connection.connect()
        await connection.wait_synchronized()
        self._connection = connection
        self._attach_listener(connection)
        await self._refresh_subscriptions()
        await self._refresh_loop()

    def _attach_listener(self, connection: Any) -> None:
        listener = _StreamListener(self)
        for method_name in ("add_market_data_listener", "add_synchronization_listener"):
            method = getattr(connection, method_name, None)
            if callable(method):
                method(listener)

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(self.refresh_seconds)
            await self._refresh_subscriptions()

    async def _refresh_subscriptions(self) -> None:
        symbols = await asyncio.to_thread(self._load_symbols)
        if not symbols or not self._connection:
            return
        new_symbols = symbols - self.subscribed_symbols
        for symbol in new_symbols:
            await self._subscribe_symbol(symbol)
        self.subscribed_symbols = symbols

    def _load_symbols(self) -> set[str]:
        with db_session() as session:
            trades = (
                session.query(Trade)
                .filter_by(account_id=self.account_id, exit_time=None, is_live=True)
                .all()
            )
        return {str(trade.symbol).upper() for trade in trades if trade.symbol}

    async def _subscribe_symbol(self, symbol: str) -> None:
        if not self._connection:
            return
        subscribe = getattr(self._connection, "subscribe_to_market_data", None)
        if not callable(subscribe):
            return
        try:
            result = subscribe(symbol)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception("Streaming: subscribe failed for %s", symbol)

    async def on_price(self, symbol: str, bid: float, ask: float) -> None:
        self.price_map[symbol.upper()] = (bid, ask)
        loop_time = asyncio.get_running_loop().time()
        if loop_time - self._last_check < self.check_seconds:
            return
        self._last_check = loop_time
        price_snapshot = dict(self.price_map)
        await asyncio.to_thread(self._handle_break_even, price_snapshot)

    def _handle_break_even(self, price_snapshot: dict[str, tuple[float, float]]) -> None:
        with db_session() as session:
            account = session.get(Mt5Account, self.account_id)
            if not account or not account.metaapi_account_id:
                return
            client = self._client or MetaApiClient(self.token, account.metaapi_region)
            manage_break_even_for_account(
                session, account, client, price_map=price_snapshot
            )


class _StreamListener:
    def __init__(self, stream: AccountStream) -> None:
        self.stream = stream

    async def on_symbol_price_updated(self, _instance_index: int, price: dict[str, Any]) -> None:
        symbol = str(price.get("symbol") or price.get("instrument") or "").upper()
        if not symbol:
            return
        bid = _safe_float(price.get("bid") or price.get("bidPrice"))
        ask = _safe_float(price.get("ask") or price.get("askPrice"))
        if bid is None and ask is not None:
            bid = ask
        if ask is None and bid is not None:
            ask = bid
        if bid is None or ask is None:
            return
        await self.stream.on_price(symbol, bid, ask)


def _load_stream_accounts() -> list[AccountStream]:
    streams: list[AccountStream] = []
    with db_session() as session:
        accounts = (
            session.query(Mt5Account)
            .filter(Mt5Account.metaapi_account_id.isnot(None))
            .all()
        )
        for account in accounts:
            if not account.settings or not account.settings.ai_enabled:
                continue
            if not account.settings.enable_break_even:
                continue
            token = _resolve_token(session, account)
            if not token:
                logger.warning("Streaming: missing MetaApi token for account %s", account.id)
                continue
            streams.append(
                AccountStream(
                    account_id=account.id,
                    metaapi_account_id=str(account.metaapi_account_id),
                    token=token,
                    region=account.metaapi_region,
                    refresh_seconds=METAAPI_STREAMING_REFRESH_SECONDS,
                    check_seconds=METAAPI_STREAMING_CHECK_SECONDS,
                )
            )
    return streams


async def run_streaming_worker() -> None:
    if not METAAPI_STREAMING_ENABLED:
        logger.warning("MetaApi streaming worker disabled via METAAPI_STREAMING_ENABLED.")
        return
    if not METAAPI_TOKEN:
        logger.warning("MetaApi streaming worker missing METAAPI_TOKEN.")
    streams = _load_stream_accounts()
    if not streams:
        logger.warning("MetaApi streaming worker found no eligible accounts.")
        return
    tasks = [asyncio.create_task(stream.start()) for stream in streams]
    await asyncio.gather(*tasks)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(run_streaming_worker())


if __name__ == "__main__":
    main()
