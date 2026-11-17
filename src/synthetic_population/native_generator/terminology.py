from __future__ import annotations
from typing import Any, Iterable


def iter_codings(value: Any, path: str = "") -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        if isinstance(value.get("system"), str) and isinstance(value.get("code"), str):
            yield path, value
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            yield from iter_codings(child, child_path)
    elif isinstance(value, list):
        for child in value:
            yield from iter_codings(child, f"{path}[*]")
