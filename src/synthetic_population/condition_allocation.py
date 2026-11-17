"""Deterministic S18 coding classes for emitted source Condition records."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any

from .condition_coding import (CLASS_SYSTEM, SNOMED, TERMINOLOGY_CATALOG,
                               condition_code_inventory, select_condition_coding)


CLASS_ORDER = ("local-only", "standard-plus-local", "standard-only")
CLASS_PERCENT = (5, 30, 65)
ARCHETYPE_SYSTEM = "urn:hdp-test-bundle:synthetic-archetype"


def hamilton_counts(total: int, weights: tuple[int, ...] = CLASS_PERCENT) -> tuple[int, ...]:
    """Use integer Hamilton arithmetic and declaration-order tie breaking."""
    if total < 0 or not weights or any(weight < 0 for weight in weights) or sum(weights) == 0:
        raise ValueError("invalid Condition class apportionment")
    denominator = sum(weights)
    floor = [total * weight // denominator for weight in weights]
    remainders = [total * weight % denominator for weight in weights]
    for index in sorted(range(len(weights)), key=lambda item: (-remainders[item], item))[:total - sum(floor)]:
        floor[index] += 1
    return tuple(floor)


def _rank(seed: int, source: str, full_url: str, category: str) -> str:
    return hashlib.sha256(f"{seed}|condition-class|{category}|{source}|{full_url}".encode()).hexdigest()


def _apportion(target: int, capacities: dict[str, int]) -> dict[str, int]:
    """Proportional Hamilton apportionment with deterministic capacity repair."""
    if target > sum(capacities.values()):
        raise ValueError(f"mapped ordinary Condition capacity {sum(capacities.values())} below target {target}")
    denominator = sum(capacities.values())
    result = {key: target * value // denominator for key, value in capacities.items()}
    remainder = target - sum(result.values())
    order = sorted(capacities, key=lambda key: (-(target * capacities[key] % denominator), key))
    for key in order:
        if remainder == 0:
            break
        if result[key] < capacities[key]:
            result[key] += 1
            remainder -= 1
    if remainder:
        for key in sorted(capacities):
            while remainder and result[key] < capacities[key]:
                result[key] += 1
                remainder -= 1
    if remainder:
        raise AssertionError("Condition apportionment failed capacity repair")
    return result


def allocate_condition_coding(per_org: dict[str, list[tuple[str, dict[str, Any]]]], spec: dict) -> dict[str, Any]:
    """Classify every emitted source Condition once, using source identity ranks.

    Existing problem fixtures keep their standard coding and problem identity.
    Local classes use only exact ordinary mapped facts in approved archetypes.
    """
    if spec["terminology"]["catalog"] != TERMINOLOGY_CATALOG:
        raise ValueError("S18 allocation requires the accepted local terminology catalog")
    inventory = condition_code_inventory()
    by_standard = {
        (row["mapping"]["system"], row["mapping"]["code"], row["mapping"]["display"]): row
        for row in inventory.values()
    }
    metadata = spec["metadata"]
    all_records: dict[str, list[tuple[int, str, dict]]] = defaultdict(list)
    eligible: dict[str, dict[str, list[tuple[int, str, dict, dict]]]] = defaultdict(lambda: defaultdict(list))
    ineligible_reasons = Counter()
    identities = set()
    for source, records in per_org.items():
        for position, (url, resource) in enumerate(records):
            if resource.get("resourceType") != "Condition":
                continue
            if (source, url) in identities:
                raise ValueError(f"duplicate source Condition identity: {source}/{url}")
            identities.add((source, url))
            all_records[source].append((position, url, resource))
            tags = resource.get("meta", {}).get("tag", [])
            codings = resource.get("code", {}).get("coding", [])
            if len(codings) != 1 or codings[0].get("system") != SNOMED:
                raise ValueError(f"source Condition lacks one original SNOMED coding: {source}/{url}")
            if any(tag.get("system") == metadata["problemMarkerSystem"] for tag in tags):
                ineligible_reasons[(source, "declared-fixture")] += 1
                continue
            standard = codings[0]
            concept = by_standard.get((standard.get("system"), standard.get("code"), standard.get("display")))
            if concept is None:
                ineligible_reasons[(source, "unmapped-or-display-drift")] += 1
                continue
            archetypes = [tag["code"] for tag in tags if tag.get("system") == ARCHETYPE_SYSTEM]
            if (len(archetypes) != 1 or archetypes[0] not in concept["eligibility"]["archetypes"]
                    or source not in concept["eligibility"]["sources"]
                    or not set(resource.get("meta", {}).get("profile", [])) & set(concept["profiles"])):
                ineligible_reasons[(source, "profile-source-or-archetype")] += 1
                continue
            eligible[source][archetypes[0]].append((position, url, resource, concept))
    total = sum(map(len, all_records.values()))
    targets = dict(zip(CLASS_ORDER, hamilton_counts(total)))
    source_totals = {source: len(rows) for source, rows in all_records.items()}
    local_source = _apportion(targets["local-only"], source_totals)
    plus_source = _apportion(targets["standard-plus-local"], source_totals)
    assignments: dict[tuple[str, str], tuple[str, dict | None]] = {}
    for source in sorted(all_records):
        groups = eligible[source]
        capacities = {archetype: len(rows) for archetype, rows in groups.items()}
        if sum(capacities.values()) < local_source[source] + plus_source[source]:
            raise ValueError(f"mapped ordinary Condition capacity shortfall for {source}: {sum(capacities.values())} < {local_source[source] + plus_source[source]}")
        local_by_archetype = _apportion(local_source[source], capacities)
        remaining = {key: capacities[key] - local_by_archetype[key] for key in capacities}
        plus_by_archetype = _apportion(plus_source[source], remaining)
        for archetype, rows in sorted(groups.items()):
            local_rows = sorted(rows, key=lambda row: (_rank(spec["seed"], source, row[1], "local-only"), row[1]))[:local_by_archetype[archetype]]
            local_urls = {row[1] for row in local_rows}
            for _position, url, _resource, concept in local_rows:
                assignments[(source, url)] = ("local-only", concept)
            plus_rows = sorted((row for row in rows if row[1] not in local_urls), key=lambda row: (_rank(spec["seed"], source, row[1], "standard-plus-local"), row[1]))[:plus_by_archetype[archetype]]
            for _position, url, _resource, concept in plus_rows:
                assignments[(source, url)] = ("standard-plus-local", concept)
    actual = Counter()
    by_source = Counter()
    by_archetype = Counter()
    for source, rows in all_records.items():
        for position, url, resource in rows:
            kind, concept = assignments.get((source, url), ("standard-only", None))
            selection = {"class": kind}
            if concept is not None:
                selection["localCode"] = concept["code"]
            if kind == "local-only":
                selection["problemId"] = "s18-local-only-" + hashlib.sha256(f"{source}|{url}".encode()).hexdigest()[:20]
            selected = select_condition_coding(resource, selection, metadata)
            per_org[source][position] = (url, selected)
            actual[kind] += 1
            by_source[(source, kind)] += 1
            archetype = next((tag["code"] for tag in selected["meta"]["tag"] if tag.get("system") == ARCHETYPE_SYSTEM), "unknown")
            by_archetype[(source, archetype, kind)] += 1
    if any(actual[kind] != targets[kind] for kind in CLASS_ORDER):
        raise AssertionError("Condition allocation differs from Hamilton target")
    return {
        "schemaVersion": "1.0.0", "denominator": total,
        "target": targets, "actual": dict(actual),
        "eligibleOrdinaryMappedCount": sum(len(rows) for groups in eligible.values() for rows in groups.values()),
        "ineligible": [{"source": source, "reason": reason, "count": count} for (source, reason), count in sorted(ineligible_reasons.items())],
        "bySource": [{"source": source, "class": kind, "count": count} for (source, kind), count in sorted(by_source.items())],
        "bySourceArchetype": [{"source": source, "archetype": archetype, "class": kind, "count": count} for (source, archetype, kind), count in sorted(by_archetype.items())],
        "selectionBasis": "Stable source/fullUrl hash rank within eligible source/archetype strata; no code-frequency ranking",
    }
