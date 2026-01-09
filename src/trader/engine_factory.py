from __future__ import annotations

from typing import Literal

from . import config
from .strategies.aggressive.rules_engine import SignalEngine as AggressiveEngine
from .strategies.passive.rules_engine import SignalEngine as PassiveEngine
from .strategies.sniper.rules_engine import SignalEngine as SniperEngine


ModelMode = Literal["aggressive", "passive", "sniper"]


def get_engine(symbol: str, candle_provider=None, mode: str | None = None):
    selected = (mode or config.MODEL_MODE or "aggressive").lower()
    if selected == "passive":
        engine_cls = PassiveEngine
    elif selected == "sniper":
        engine_cls = SniperEngine
    else:
        engine_cls = AggressiveEngine
    return engine_cls(symbol, candle_provider=candle_provider)
