from __future__ import annotations

import time as time_module

from . import config
from .logger import SignalLogger
from .rules_engine import SignalOutput
from .mt5_client import get_tick_time, initialize, resolve_symbol, shutdown
from .engine_factory import get_engine
from .time_utils import session_from_utc
from .time_utils import broker_epoch_to_utc


def _round_price(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 5)


def _signal_identity(signal: SignalOutput) -> tuple | None:
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


def main() -> None:
    initialize()
    symbol = resolve_symbol()
    logger = SignalLogger("logs/signals.jsonl")
    engine = get_engine(symbol, mode=config.MODEL_MODE)
    last_trade_key: tuple | None = None

    try:
        while True:
            tick_time = get_tick_time(symbol)
            now_utc = broker_epoch_to_utc(tick_time)
            if session_from_utc(now_utc) is None:
                time_module.sleep(config.POLL_SECONDS)
                continue
            signal = engine.evaluate(now_utc)
            signal_key = _signal_identity(signal)
            if signal_key is not None and signal_key == last_trade_key:
                time_module.sleep(config.POLL_SECONDS)
                continue
            if signal_key is not None:
                last_trade_key = signal_key
            logger.log(signal)
            time_module.sleep(config.POLL_SECONDS)
    finally:
        shutdown()


if __name__ == "__main__":
    main()
