from __future__ import annotations

import time as time_module

from . import config
from .logger import SignalLogger
from .mt5_client import get_tick_time, initialize, resolve_symbol, shutdown
from .rules_engine import SignalEngine
from .time_utils import session_from_utc
from .time_utils import broker_epoch_to_utc


def main() -> None:
    initialize()
    symbol = resolve_symbol()
    logger = SignalLogger("logs/signals.jsonl")
    engine = SignalEngine(symbol)

    try:
        while True:
            tick_time = get_tick_time(symbol)
            now_utc = broker_epoch_to_utc(tick_time)
            if session_from_utc(now_utc) is None:
                time_module.sleep(config.POLL_SECONDS)
                continue
            signal = engine.evaluate(now_utc)
            logger.log(signal)
            time_module.sleep(config.POLL_SECONDS)
    finally:
        shutdown()


if __name__ == "__main__":
    main()
