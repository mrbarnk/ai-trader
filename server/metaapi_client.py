from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    from metaapi_cloud_sdk import MetaApi
except Exception:
    MetaApi = None

from .settings import METAAPI_SYNC_LOOKBACK_DAYS


class MetaApiError(RuntimeError):
    pass


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _ensure_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {
            key: val for key, val in value.__dict__.items() if not key.startswith("_")
        }
    return {"raw": value}


def _account_to_dict(account: Any | None) -> dict[str, Any]:
    if account is None:
        return {}
    if isinstance(account, dict):
        return account
    return {
        "id": getattr(account, "id", None),
        "state": getattr(account, "state", None),
        "connectionStatus": getattr(account, "connection_status", None),
        "name": getattr(account, "name", None),
        "type": getattr(account, "type", None),
        "server": getattr(account, "server", None),
        "login": getattr(account, "login", None),
        "region": getattr(account, "region", None),
    }


def _trade_options(payload: dict[str, Any]) -> dict[str, Any] | None:
    options: dict[str, Any] = {}
    comment = payload.get("comment")
    if comment:
        options["comment"] = str(comment)
    magic = payload.get("magic")
    if magic not in (None, ""):
        options["magic"] = str(magic)
    return options or None


async def _safe_close(obj: Any) -> None:
    close_method = getattr(obj, "close", None)
    if not callable(close_method):
        return
    result = close_method()
    if asyncio.iscoroutine(result):
        await result


class MetaApiClient:
    def __init__(self, token: str, region: str | None = None) -> None:
        self.token = token
        self.region = region

    def _metaapi_opts(self) -> dict[str, Any] | None:
        if self.region:
            return {"region": self.region}
        return None

    def _run(self, coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                return asyncio.run(coro)
            except Exception as exc:
                raise MetaApiError(self._format_error("MetaApi SDK error", exc)) from exc
        return self._run_in_thread(coro)

    def _run_in_thread(self, coro):
        result: dict[str, Any] = {}

        def runner() -> None:
            try:
                result["value"] = asyncio.run(coro)
            except Exception as exc:
                result["error"] = exc

        thread = threading.Thread(target=runner, name="MetaApiSDKRunner")
        thread.start()
        thread.join()

        error = result.get("error")
        if error:
            raise MetaApiError(self._format_error("MetaApi SDK error", error))
        return result.get("value")

    def _format_error(self, prefix: str, exc: Exception) -> str:
        details = getattr(exc, "details", None)
        if details:
            return f"{prefix}: {exc}. Details: {details}"
        return f"{prefix}: {exc}"

    async def _with_metaapi(self, fn):
        if MetaApi is None:
            raise MetaApiError("MetaApi SDK not available.")
        metaapi = MetaApi(self.token, self._metaapi_opts())
        try:
            return await fn(metaapi)
        finally:
            await _safe_close(metaapi)

    async def _with_rpc_connection(self, account_id: str, fn):
        async def _execute(metaapi):
            account_api = metaapi.metatrader_account_api
            account = await account_api.get_account(account_id)
            try:
                if getattr(account, "state", "") != "DEPLOYED":
                    await account.deploy()
            except Exception:
                pass
            try:
                await account.wait_connected()
            except Exception:
                pass
            connection = account.get_rpc_connection()
            await connection.connect()
            await connection.wait_synchronized()
            try:
                return await fn(connection, account)
            finally:
                await _safe_close(connection)

        return await self._with_metaapi(_execute)

    async def _with_streaming_connection(self, account_id: str, fn):
        async def _execute(metaapi):
            account_api = metaapi.metatrader_account_api
            account = await account_api.get_account(account_id)
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
            try:
                return await fn(connection, account)
            finally:
                await _safe_close(connection)

        return await self._with_metaapi(_execute)

    def create_account(self, payload: dict[str, Any]) -> dict[str, Any]:
        async def _call(metaapi):
            account = await metaapi.metatrader_account_api.create_account(payload)
            return _account_to_dict(account)

        return self._run(self._with_metaapi(_call))

    def get_account(self, account_id: str) -> dict[str, Any]:
        async def _call(metaapi):
            account = await metaapi.metatrader_account_api.get_account(account_id)
            return _account_to_dict(account)

        return self._run(self._with_metaapi(_call))

    def deploy_account(self, account_id: str) -> None:
        async def _call(metaapi):
            account = await metaapi.metatrader_account_api.get_account(account_id)
            await account.deploy()

        self._run(self._with_metaapi(_call))

    def get_account_information(self, account_id: str) -> dict[str, Any]:
        async def _call(connection, _account):
            info = await connection.get_account_information()
            return _ensure_dict(info)

        return self._run(self._with_rpc_connection(account_id, _call))

    def get_account_state(self, account_id: str) -> dict[str, Any]:
        async def _call(metaapi):
            account = await metaapi.metatrader_account_api.get_account(account_id)
            return {
                "state": getattr(account, "state", None),
                "connectionStatus": getattr(account, "connection_status", None),
            }

        return self._run(self._with_metaapi(_call))

    def get_deals(
        self, account_id: str, start_time: str | None = None, end_time: str | None = None
    ) -> list[dict[str, Any]]:
        async def _call(connection, _account):
            end_dt = _parse_iso_datetime(end_time) or datetime.now(timezone.utc)
            start_dt = _parse_iso_datetime(start_time)
            if start_dt is None:
                start_dt = end_dt - timedelta(days=METAAPI_SYNC_LOOKBACK_DAYS)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
            if start_dt > end_dt:
                start_dt, end_dt = end_dt, start_dt
            deals = await connection.get_deals_by_time_range(
                start_dt, end_dt, offset=0, limit=1000
            )
            if isinstance(deals, dict):
                items = deals.get("deals")
                if isinstance(items, list):
                    return items
            if isinstance(deals, list):
                return deals
            return []

        return self._run(self._with_rpc_connection(account_id, _call))

    def place_order(self, account_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        async def _call(connection, _account):
            action_type = str(payload.get("actionType") or payload.get("type") or "").upper()
            symbol = str(payload.get("symbol") or "").upper()
            volume = payload.get("volume")
            price = payload.get("price") or payload.get("openPrice")
            stop_loss = payload.get("stopLoss")
            take_profit = payload.get("takeProfit")
            options = _trade_options(payload)

            if not symbol:
                raise MetaApiError("MetaApi order missing symbol.")
            if volume is None:
                raise MetaApiError("MetaApi order missing volume.")

            if action_type in {"ORDER_TYPE_BUY_LIMIT", "BUY_LIMIT"}:
                if price is None:
                    raise MetaApiError("Limit order requires price.")
                result = await connection.create_limit_buy_order(
                    symbol, float(volume), float(price), stop_loss, take_profit, options
                )
                return _ensure_dict(result)
            if action_type in {"ORDER_TYPE_SELL_LIMIT", "SELL_LIMIT"}:
                if price is None:
                    raise MetaApiError("Limit order requires price.")
                result = await connection.create_limit_sell_order(
                    symbol, float(volume), float(price), stop_loss, take_profit, options
                )
                return _ensure_dict(result)
            if action_type in {"ORDER_TYPE_BUY_STOP", "BUY_STOP"}:
                if price is None:
                    raise MetaApiError("Stop order requires price.")
                result = await connection.create_stop_buy_order(
                    symbol, float(volume), float(price), stop_loss, take_profit, options
                )
                return _ensure_dict(result)
            if action_type in {"ORDER_TYPE_SELL_STOP", "SELL_STOP"}:
                if price is None:
                    raise MetaApiError("Stop order requires price.")
                result = await connection.create_stop_sell_order(
                    symbol, float(volume), float(price), stop_loss, take_profit, options
                )
                return _ensure_dict(result)
            if action_type in {"ORDER_TYPE_BUY", "BUY"}:
                result = await connection.create_market_buy_order(
                    symbol, float(volume), stop_loss, take_profit, options
                )
                return _ensure_dict(result)
            if action_type in {"ORDER_TYPE_SELL", "SELL"}:
                result = await connection.create_market_sell_order(
                    symbol, float(volume), stop_loss, take_profit, options
                )
                return _ensure_dict(result)
            if action_type == "POSITION_CLOSE":
                position_id = payload.get("positionId") or payload.get("position_id")
                if not position_id:
                    raise MetaApiError("Position close requires positionId.")
                if volume is not None:
                    result = await connection.close_position_partially(
                        str(position_id), float(volume)
                    )
                else:
                    result = await connection.close_position(str(position_id))
                return _ensure_dict(result)
            if action_type == "POSITION_MODIFY":
                position_id = payload.get("positionId") or payload.get("position_id")
                if not position_id:
                    raise MetaApiError("Position modify requires positionId.")
                result = await connection.modify_position(
                    str(position_id), stop_loss=stop_loss, take_profit=take_profit
                )
                return _ensure_dict(result)

            raise MetaApiError(f"Unsupported MetaApi actionType: {action_type or 'n/a'}.")

        try:
            return self._run(self._with_streaming_connection(account_id, _call))
        except MetaApiError as exc:
            last_error = exc
        try:
            return self._run(self._with_rpc_connection(account_id, _call))
        except MetaApiError as exc:
            raise MetaApiError(
                f"MetaApi streaming order failed: {last_error}. RPC failed: {exc}."
            ) from exc

    def get_positions(self, account_id: str) -> list[dict[str, Any]]:
        async def _call(connection, _account):
            positions = await connection.get_positions()
            if not positions:
                return []
            return [_ensure_dict(pos) for pos in positions]

        return self._run(self._with_rpc_connection(account_id, _call))

    def get_symbol_price(self, account_id: str, symbol: str) -> dict[str, Any]:
        async def _call(connection, _account):
            subscribe = getattr(connection, "subscribe_to_market_data", None)
            if callable(subscribe):
                result = subscribe(str(symbol).upper())
                if asyncio.iscoroutine(result):
                    await result
            data = await connection.get_symbol_price(str(symbol), keep_subscription=False)
            return _ensure_dict(data)

        try:
            return self._run(self._with_streaming_connection(account_id, _call))
        except MetaApiError as exc:
            last_error = exc
        try:
            return self._run(self._with_rpc_connection(account_id, _call))
        except MetaApiError as exc:
            raise MetaApiError(
                f"MetaApi streaming price failed: {last_error}. RPC failed: {exc}."
            ) from exc

    def close_position(
        self, account_id: str, position_id: str, volume: float | None
    ) -> dict[str, Any]:
        async def _call(connection, _account):
            if volume is None:
                result = await connection.close_position(str(position_id))
            else:
                result = await connection.close_position_partially(
                    str(position_id), float(volume)
                )
            return _ensure_dict(result)

        return self._run(self._with_rpc_connection(account_id, _call))

    def modify_position(
        self, account_id: str, position_id: str, stop_loss: float | None = None
    ) -> dict[str, Any]:
        async def _call(connection, _account):
            result = await connection.modify_position(str(position_id), stop_loss=stop_loss)
            return _ensure_dict(result)

        return self._run(self._with_rpc_connection(account_id, _call))
