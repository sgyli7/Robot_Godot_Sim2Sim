"""Stable, inspectable configuration and evidence identities."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
from enum import Enum
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(value):
    """Keep actual config values, callable identity and callable source bytes."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return {"nonfinite": str(value)}
        return value
    if isinstance(value, Enum):
        return {"enum": qualified(type(value)), "value": snapshot(value.value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if isinstance(value, (set, frozenset)):
        return sorted((snapshot(item) for item in value), key=lambda item: canonical_bytes(item))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"type": qualified(type(value)), "fields": {
            field.name: snapshot(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }}
    if isinstance(value, dict):
        return {str(key): snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(item) for item in value]
    if callable(value):
        result = {"callable": qualified(value)}
        try:
            path = Path(inspect.getsourcefile(value))
            result.update(source=str(path), sha256=sha256_file(path))
        except (TypeError, OSError):
            pass
        return result
    if hasattr(value, "tolist"):
        return snapshot(value.tolist())
    # Torch dtypes are configuration constants, not tensors or hidden state.
    if type(value).__name__ == "dtype":
        return str(value)
    raise TypeError(f"Unserializable configuration value: {type(value)!r}")


def qualified(value) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
