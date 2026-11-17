from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from contextvars import ContextVar
from functools import wraps
from typing import Any, Iterable


def walk(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def codings(value: Any) -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        if isinstance(value.get("system"), str) and isinstance(value.get("code"), str):
            yield value["system"], value["code"]
        for child in value.values():
            yield from codings(child)
    elif isinstance(value, list):
        for child in value:
            yield from codings(child)


def references(value: Any) -> Iterable[str]:
    for key, child in walk(value):
        if key == "reference" and isinstance(child, str):
            yield child


def is_supplemental_capture(resource: dict[str, Any]) -> bool:
    """Identify the separately audited source-local capture episode layer."""
    return any(
        tag.get("system") == "urn:hdp-test-bundle:synthetic-capture-variant"
        for tag in resource.get("meta", {}).get("tag", [])
    )


def _build_patient_graphs(
    bundles: dict[str, dict[str, Any]], spec: dict[str, Any], *, include_demographics: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    system = spec["identity"]["cohortIdentifierSystem"]
    url_to_cohort: dict[tuple[str, str], str] = {}
    patients: dict[str, dict[str, Any]] = {}
    for source, bundle in bundles.items():
        for entry in bundle["entry"]:
            resource = entry["resource"]
            if resource.get("resourceType") != "Patient":
                continue
            cohort = next((
                identifier.get("value") for identifier in resource.get("identifier", [])
                if identifier.get("system") == system
            ), None)
            if cohort:
                url_to_cohort[(source, entry["fullUrl"])] = cohort
                patients.setdefault(cohort, resource)
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source, bundle in bundles.items():
        for entry in bundle["entry"]:
            resource = entry["resource"]
            cohort = url_to_cohort.get((source, entry["fullUrl"]))
            if cohort is None:
                cohort = next((
                    url_to_cohort[(source, reference)] for reference in references(resource)
                    if (source, reference) in url_to_cohort
                ), None)
            if cohort is None:
                cohort = next((
                    value.get("value") for _key, value in walk(resource)
                    if isinstance(value, dict) and value.get("system") == system
                    and value.get("value") in patients
                ), None)
            if cohort and (include_demographics or resource.get("resourceType") not in {"Patient", "RelatedPerson"}):
                result[cohort].append({**resource, "_source": source, "_fullUrl": entry["fullUrl"]})
    return dict(result), patients


def strip_generator_metadata(resource: dict[str, Any]) -> dict[str, Any]:
    """Keep observable clinical content while removing all generator evidence."""
    excluded = {"id", "identifier", "text", "note", "contained", "_source", "_fullUrl"}

    def visit(value: Any, key: str | None = None) -> Any:
        if isinstance(value, dict):
            output = {}
            for child_key, child in value.items():
                if child_key in excluded:
                    continue
                if child_key == "tag":
                    kept = [
                        item for item in child
                        if not item.get("system", "").startswith("urn:hdp-test-bundle:synthetic")
                    ]
                    if kept:
                        output[child_key] = visit(kept, child_key)
                    continue
                if child_key == "lastUpdated" and key == "meta":
                    continue
                if child_key == "source" and key == "meta":
                    continue
                output[child_key] = visit(child, child_key)
            return output
        if isinstance(value, list):
            return [visit(child, key) for child in value]
        return value

    return visit(resource)


_GRAPH_INDEX = ContextVar("synthetic_population_verification_graph_index", default=None)


def with_graph_index(function):
    """Share read-only parsed graph indexes for one independent disk verification."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        token = _GRAPH_INDEX.set({})
        try:
            return function(*args, **kwargs)
        finally:
            _GRAPH_INDEX.reset(token)
    return wrapped


def patient_graphs(bundles, spec, *, include_demographics=True):
    cache = _GRAPH_INDEX.get()
    if cache is None:
        return deepcopy(_build_patient_graphs(bundles, spec, include_demographics=include_demographics))
    key = (id(bundles), id(spec), include_demographics)
    if key not in cache:
        # Strong input references prevent id reuse within the invocation. Return
        # isolated payloads so a consumer cannot mutate inputs or cached graphs.
        cache[key] = (bundles, spec, _build_patient_graphs(bundles, spec, include_demographics=include_demographics))
    return deepcopy(cache[key][2])
