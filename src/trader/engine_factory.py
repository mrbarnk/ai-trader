from __future__ import annotations

from typing import Literal

from . import config
from .strategies.aggressive.rules_engine import SignalEngine as AggressiveEngine
from .strategies.passive.rules_engine import SignalEngine as PassiveEngine


ModelMode = Literal["aggressive", "passive"]


def get_engine(symbol: str, candle_provider=None, mode: str | None = None):
    selected = (mode or config.MODEL_MODE or "aggressive").lower()
    engine_cls = PassiveEngine if selected == "passive" else AggressiveEngine
    return engine_cls(symbol, candle_provider=candle_provider)
