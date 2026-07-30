"""Result serialisation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from experiments import RESULTS_DIR


def save_report(name: str, payload: dict) -> Path:
    payload = dict(payload)
    payload.setdefault("_meta", {})
    payload["_meta"].setdefault("name", name)
    payload["_meta"].setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
    out = RESULTS_DIR / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")
    return out


def _default(obj: Any):
    try:
        return float(obj)
    except Exception:
        return str(obj)


def print_header(name: str) -> None:
    bar = "=" * (len(name) + 4)
    print(f"\n{bar}\n  {name}\n{bar}")
