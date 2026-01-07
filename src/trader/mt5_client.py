from __future__ import annotations

import MetaTrader5 as mt5

from . import config


def initialize() -> None:
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")


def shutdown() -> None:
    mt5.shutdown()


def resolve_symbol() -> str:
    for symbol in config.SYMBOL_VARIANTS:
        info = mt5.symbol_info(symbol)
        if info is None:
            continue
        if not info.visible:
            mt5.symbol_select(symbol, True)
        return symbol
    raise RuntimeError("No broker symbol found for GU variants")


def get_tick_time(symbol: str) -> int:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"No tick data for {symbol}")
    return int(tick.time)


def get_rates(symbol: str, timeframe: int, count: int):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        raise RuntimeError(f"Failed to fetch rates for {symbol}: {mt5.last_error()}")
    return rates
