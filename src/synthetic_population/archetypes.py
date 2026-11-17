from __future__ import annotations

import hashlib
from functools import lru_cache
from datetime import datetime, timedelta
from typing import Any

from .catalogs import load_catalog
from .nested_allocation import nested, block_position

_BLOCK_MULTIPLIERS = tuple(value for value in range(1, 1000) if value % 2 and value % 5)


@lru_cache(maxsize=256)
def _permutation(seed: int, block: int) -> tuple[int, int]:
    digest = hashlib.sha256(f"{seed}|archetype-block|{block}".encode()).digest()
    return (_BLOCK_MULTIPLIERS[int.from_bytes(digest[:4], "big") % len(_BLOCK_MULTIPLIERS)],
            int.from_bytes(digest[4:8], "big") % 1000)


def _archetype_slot(spec: dict[str, Any], index: int) -> int:
    block, local = block_position(index)
    if not block:
        return local
    multiplier, offset = _permutation(spec["seed"], block)
    return ((local - 1) * multiplier + offset) % 1000 + 1


def distribution(spec: dict[str, Any]) -> list[dict[str, Any]]:
    catalog = load_catalog(spec["archetypes"]["catalog"])
    rows = catalog["primaryDistribution"]
    if spec["populationSize"] == 1000:
        return rows
    exact = [spec["populationSize"] * row["count"] / 1000 for row in rows]
    counts = [int(value) for value in exact]
    for index in sorted(range(len(rows)), key=lambda i: (exact[i] - counts[i], -i), reverse=True)[:spec["populationSize"] - sum(counts)]:
        counts[index] += 1
    return [{**row, "count": count} for row, count in zip(rows, counts)]


def primary_archetype(spec: dict[str, Any], patient_index: int) -> str:
    if nested(spec):
        patient_index = _archetype_slot(spec, patient_index)
        spec = {**spec, "populationSize": 1000}
    cursor = 0
    for row in distribution(spec):
        cursor += row["count"]
        if patient_index <= cursor:
            return row["id"]
    raise ValueError(f"patient index outside archetype distribution: {patient_index}")


def archetype_ordinal(spec: dict[str, Any], index: int, archetype: str) -> int:
    """Zero-based ordinal within a fixed block's archetype quota."""
    if nested(spec):
        index = _archetype_slot(spec, index)
        spec = {**spec, "populationSize": 1000}
    start = 1
    for row in distribution(spec):
        if row["id"] == archetype:
            return index - start
        start += row["count"]
    raise ValueError(f"unknown archetype: {archetype}")


def archetype_rule(spec: dict[str, Any], archetype: str) -> dict[str, Any]:
    return next(row for row in distribution(spec) if row["id"] == archetype)


def source_event_time(spec: dict[str, Any], organization: str, patient_index: int) -> datetime:
    anchor = datetime.fromisoformat(spec["timelineAnchor"].replace("Z", "+00:00"))
    if "sourceModels" not in spec:
        legacy_offsets = {"hmo": -30, "hospital-a": -20, "hospital-b": -10}
        jitter = (patient_index * 37 + len(organization) * 11) % 21
        return anchor + timedelta(days=legacy_offsets[organization] - jitter, hours=8 + patient_index % 8)
    model = load_catalog(spec["sourceModels"]["catalog"])["sources"][organization]
    source_factor = {"hmo": 37, "hospital-a": 61, "hospital-b": 89}[organization]
    jitter = (patient_index * source_factor + len(organization) * 11) % 241
    return anchor + timedelta(days=model["eventOffsetDays"] - jitter, hours=8 + patient_index % 8)


def combined_event_dates(spec: dict[str, Any], patient_index: int) -> list[datetime]:
    archetype = primary_archetype(spec, patient_index)
    rule = archetype_rule(spec, archetype)
    anchor = datetime.fromisoformat(spec["timelineAnchor"].replace("Z", "+00:00"))
    span = max(rule["minimumSpanDays"], 400)
    count = max(rule["minimumDates"], 2)
    return [anchor - timedelta(days=span - round(position * span / (count - 1))) for position in range(count)]
