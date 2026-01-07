from __future__ import annotations

import json
from datetime import date, datetime
from dataclasses import asdict
from pathlib import Path

from .rules_engine import SignalOutput


class SignalLogger:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, signal: SignalOutput | dict) -> None:
        if isinstance(signal, dict):
            record = signal
        else:
            record = asdict(signal)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=self._json_default) + "\n")

    @staticmethod
    def _json_default(value: object) -> str:
        if isinstance(value, datetime):
            return value.isoformat() + "Z"
        if isinstance(value, date):
            return value.isoformat()
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")
