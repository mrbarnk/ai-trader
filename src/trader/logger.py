from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .rules_engine import SignalOutput


class SignalLogger:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, signal: SignalOutput) -> None:
        record = asdict(signal)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
