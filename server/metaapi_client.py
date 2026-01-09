from __future__ import annotations

import json
from typing import Any

import requests

from .settings import (
    METAAPI_CLIENT_URL,
    METAAPI_DEALS_PATH,
    METAAPI_PROVISIONING_URL,
    METAAPI_TRADE_PATH,
    METAAPI_SSL_VERIFY,
    METAAPI_TIMEOUT_SECONDS,
)


class MetaApiError(RuntimeError):
    pass


def _format_base_url(template: str, region: str | None) -> str:
    if "{region}" in template:
        return template.format(region=region or "").rstrip("/")
    return template.rstrip("/")


def _join_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


class MetaApiClient:
    def __init__(self, token: str, region: str | None = None) -> None:
        self.token = token
        self.region = region
        self.provisioning_url = _format_base_url(METAAPI_PROVISIONING_URL, region)
        self.client_url = _format_base_url(METAAPI_CLIENT_URL, region)

    def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        headers = kwargs.pop("headers", {})
        headers.setdefault("Authorization", f"Bearer {self.token}")
        headers.setdefault("Content-Type", "application/json")
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                timeout=METAAPI_TIMEOUT_SECONDS,
                verify=METAAPI_SSL_VERIFY,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise MetaApiError(f"MetaApi request failed: {exc}") from exc
        if response.status_code >= 400:
            detail = response.text.strip()
            raise MetaApiError(
                f"MetaApi {response.status_code} error: {detail or 'request failed'}"
            )
        if not response.text:
            return None
        try:
            return response.json()
        except json.JSONDecodeError:
            return response.text

    def create_account(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = _join_url(self.provisioning_url, "/users/current/accounts")
        data = self._request("POST", url, json=payload)
        return data or {}

    def get_account(self, account_id: str) -> dict[str, Any]:
        url = _join_url(
            self.provisioning_url, f"/users/current/accounts/{account_id}"
        )
        data = self._request("GET", url)
        return data or {}

    def deploy_account(self, account_id: str) -> None:
        url = _join_url(
            self.provisioning_url, f"/users/current/accounts/{account_id}/deploy"
        )
        self._request("POST", url)

    def get_account_information(self, account_id: str) -> dict[str, Any]:
        url = _join_url(
            self.client_url, f"/users/current/accounts/{account_id}/account-information"
        )
        data = self._request("GET", url)
        return data or {}

    def get_account_state(self, account_id: str) -> dict[str, Any]:
        url = _join_url(
            self.client_url, f"/users/current/accounts/{account_id}/account-state"
        )
        data = self._request("GET", url)
        return data or {}

    def get_deals(
        self, account_id: str, start_time: str | None = None, end_time: str | None = None
    ) -> list[dict[str, Any]]:
        path = METAAPI_DEALS_PATH.format(account_id=account_id)
        url = _join_url(self.client_url, path)
        params: dict[str, Any] = {}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        data = self._request("GET", url, params=params)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "deals" in data and isinstance(data["deals"], list):
            return data["deals"]
        return []

    def place_order(self, account_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        path = METAAPI_TRADE_PATH.format(account_id=account_id)
        url = _join_url(self.client_url, path)
        data = self._request("POST", url, json=payload)
        if isinstance(data, dict):
            return data
        return {"raw": data}

    def get_positions(self, account_id: str) -> list[dict[str, Any]]:
        url = _join_url(self.client_url, f"/users/current/accounts/{account_id}/positions")
        data = self._request("GET", url)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "positions" in data and isinstance(data["positions"], list):
            return data["positions"]
        return []

    def get_symbol_price(self, account_id: str, symbol: str) -> dict[str, Any]:
        url = _join_url(
            self.client_url, f"/users/current/accounts/{account_id}/symbols/{symbol}/price"
        )
        data = self._request("GET", url)
        return data or {}

    def close_position(self, account_id: str, position_id: str, volume: float) -> dict[str, Any]:
        path = METAAPI_TRADE_PATH.format(account_id=account_id)
        url = _join_url(self.client_url, path)
        payload = {"actionType": "POSITION_CLOSE", "positionId": position_id, "volume": volume}
        data = self._request("POST", url, json=payload)
        if isinstance(data, dict):
            return data
        return {"raw": data}

    def modify_position(
        self, account_id: str, position_id: str, stop_loss: float | None = None
    ) -> dict[str, Any]:
        path = METAAPI_TRADE_PATH.format(account_id=account_id)
        url = _join_url(self.client_url, path)
        payload: dict[str, Any] = {"actionType": "POSITION_MODIFY", "positionId": position_id}
        if stop_loss is not None:
            payload["stopLoss"] = stop_loss
        data = self._request("POST", url, json=payload)
        if isinstance(data, dict):
            return data
        return {"raw": data}
