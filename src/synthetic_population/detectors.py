from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from datetime import date
import re
from typing import Any, Iterable

from .catalogs import default_detector_catalog, load_catalog
from .predicates import evaluate


_SEMANTICS = load_catalog(default_detector_catalog())
SNOMED_SEMANTIC_FAMILIES = dict(_SEMANTICS["snomedSemanticFamilies"])
RESOURCE_ALLOWED_SEMANTIC_FAMILIES = {
    resource_type: set(families)
    for resource_type, families in _SEMANTICS["resourceAllowedSemanticFamilies"].items()
}
MUTUALLY_EXCLUSIVE_DIAGNOSIS_FAMILIES = tuple(
    frozenset(codes) for codes in _SEMANTICS["mutuallyExclusiveDiagnosisFamilies"]
)
ANALYTE_UNITS = {
    code: dict(row["canonicalUnits"])
    for code, row in _SEMANTICS["analytes"].items()
}
ANALYTE_DISCORDANCE = {
    code: float(row["discordanceThreshold"])
    for code, row in _SEMANTICS["analytes"].items()
}


def _codes(value: Any) -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        if isinstance(value.get("system"), str) and isinstance(value.get("code"), str):
            yield value["system"], value["code"]
        for item in value.values():
            yield from _codes(item)
    elif isinstance(value, list):
        for item in value:
            yield from _codes(item)


def _instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def detects(resource: dict, kind: str) -> bool:
    """Resource-local detector. Group-only cases intentionally return False."""
    codes = set(_codes(resource.get("code")))
    if kind in {"conflicting-diagnosis", "discordant-lab", "probable-duplicate", "cross-source-clinical-risk"}:
        return False
    if kind == "timestamp-order":
        return bool(resource.get("issued") and resource.get("effectiveDateTime") and _instant(resource["issued"]) < _instant(resource["effectiveDateTime"]))
    if kind == "semantically-wrong-code":
        allowed = RESOURCE_ALLOWED_SEMANTIC_FAMILIES.get(resource.get("resourceType"))
        if not allowed:
            return False
        families = {
            SNOMED_SEMANTIC_FAMILIES.get(code)
            for system, code in codes if system == "http://snomed.info/sct"
        }
        return bool({family for family in families if family} - allowed)
    if kind == "unit-mismatch":
        quantity = resource.get("valueQuantity", {})
        loinc = next((code for system, code in codes if system == "http://loinc.org"), None)
        return resource.get("resourceType") == "Observation" and bool(
            quantity.get("unit") and quantity.get("code") and (
                _SEMANTICS["unitAliases"].get(quantity["unit"], quantity["unit"]) != quantity["code"]
                or (loinc in ANALYTE_UNITS and quantity["code"] not in ANALYTE_UNITS[loinc])
            )
        )
    if kind == "stale-demographics":
        return False
    raise ValueError(f"unknown detector kind: {kind}")


def normalized_linkage_features(patient: dict[str, Any]) -> dict[str, Any]:
    normalize = lambda value: re.sub(r"[^a-z0-9]", "", str(value).lower())
    name = patient.get("name", [{}])[0]
    phone = next((row.get("value", "") for row in patient.get("telecom", []) if row.get("system") in {None, "phone"}), "")
    address = patient.get("address", [{}])[-1]
    birth = patient.get("birthDate", "")
    return {"family": normalize(name.get("family", "")), "given": normalize("".join(name.get("given", []))).rstrip("h"),
            "birth": birth, "birthDay": date.fromisoformat(birth).toordinal() if birth else None,
            "gender": patient.get("gender", ""), "phone": re.sub(r"\D", "", phone.split("x", 1)[0]),
            "city": normalize(address.get("city", "")), "line": normalize("".join(address.get("line", [])))}


def linkage_score_features(a: dict, b: dict) -> tuple[float, dict[str, bool]]:
    matches = {field: bool(a[field] and a[field] == b[field]) for field in ("family", "given", "gender", "phone", "city", "line")}
    matches["birthClose"] = a["birthDay"] is not None and b["birthDay"] is not None and abs(a["birthDay"] - b["birthDay"]) <= 1
    weights = {"family": 2.0, "given": 2.0, "birthClose": 2.5, "gender": .5, "phone": 2.0, "city": .5, "line": .5}
    return sum(weights[key] for key, matched in matches.items() if matched), matches


def duplicate_linkage_score(left: dict[str, Any], right: dict[str, Any]) -> tuple[float, dict[str, bool]]:
    return linkage_score_features(normalized_linkage_features(left), normalized_linkage_features(right))


def blocked_duplicate_pairs(patients: list[dict], work_stats: dict | None = None):
    """Exact required-name and +/- one birth-day blocking; no approximate pruning."""
    features = [normalized_linkage_features(patient) for patient in patients]
    blocks = defaultdict(list)
    for position, row in enumerate(features):
        if row["family"] and row["given"] and row["birthDay"] is not None:
            blocks[(row["family"], row["given"], row["birthDay"])].append(position)
    if work_stats is not None:
        work_stats["normalizedPatients"] = work_stats.get("normalizedPatients", 0) + len(features)
        work_stats.setdefault("duplicateScoreComparisons", 0)
    for left, row in enumerate(features):
        if not row["family"] or not row["given"] or row["birthDay"] is None:
            continue
        candidates = sorted(right for day in (row["birthDay"] - 1, row["birthDay"], row["birthDay"] + 1)
                            for right in blocks.get((row["family"], row["given"], day), []) if right > left)
        for right in candidates:
            if work_stats is not None:
                work_stats["duplicateScoreComparisons"] += 1
            score, evidence = linkage_score_features(row, features[right])
            if score >= 8.0:
                yield left, right, score, evidence


def normalized_quantity(resource: dict[str, Any]) -> tuple[str, float] | None:
    code = next(
        (code for system, code in _codes(resource.get("code")) if system == "http://loinc.org"),
        None,
    )
    quantity = resource.get("valueQuantity", {})
    value = quantity.get("value")
    unit = quantity.get("code")
    if code not in ANALYTE_UNITS or unit not in ANALYTE_UNITS[code] or not isinstance(value, (int, float)):
        return None
    return code, float(value) * ANALYTE_UNITS[code][unit]


def stale_snapshot_evidence(snapshot: dict[str, Any], newer: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare provenance dates and element periods, preserving ended history controls."""
    updated_raw = snapshot.get("meta", {}).get("lastUpdated")
    if not updated_raw:
        return {"stale": False, "reasons": []}
    updated = _instant(updated_raw).date()
    newer_dates = [
        _instant(row.get("meta", {}).get("lastUpdated")).date()
        for row in newer if row.get("meta", {}).get("lastUpdated")
    ]
    if not newer_dates or (max(newer_dates) - updated).days <= 365:
        return {"stale": False, "reasons": []}
    reasons = []
    for field in ("name", "address", "telecom"):
        for item in snapshot.get(field, []):
            period = item.get("period", {})
            start = date.fromisoformat(period["start"][:10]) if period.get("start") else None
            end = date.fromisoformat(period["end"][:10]) if period.get("end") else None
            # A correctly ended historical element is evidence, not a stale fact.
            if end and end <= updated:
                continue
            if start and start > updated:
                reasons.append(f"{field}-starts-after-snapshot")
        def current_values(patient: dict[str, Any]) -> set[str]:
            values = set()
            for item in patient.get(field, []):
                if item.get("period", {}).get("end"):
                    continue
                if field == "name":
                    values.add("|".join([
                        str(item.get("use", "")), str(item.get("family", "")).lower(),
                        "".join(item.get("given", [])).lower(),
                    ]))
                elif field == "address":
                    values.add("|".join([
                        *[str(line).lower() for line in item.get("line", [])],
                        str(item.get("city", "")).lower(),
                    ]))
                else:
                    values.add(re.sub(r"\D", "", str(item.get("value", ""))))
            return {value for value in values if value}

        snapshot_values = current_values(snapshot)
        newer_values = set().union(*(current_values(row) for row in newer))
        if snapshot_values and newer_values and snapshot_values.isdisjoint(newer_values):
            reasons.append(f"{field}-conflicts-with-newer-snapshot")
    return {"stale": bool(reasons), "reasons": sorted(set(reasons))}


def detect_group(resources: list[dict], kind: str) -> bool:
    """Infer a data-quality issue without inspecting project problem tags or IDs."""
    if kind in {"conflicting-diagnosis", "discordant-lab", "unit-mismatch"}:
        return bool(quality_variant_evidence(resources, kind))
    if kind == "probable-duplicate":
        patients = [resource for resource in resources if resource.get("resourceType") == "Patient"]
        if len(patients) < 2:
            return False
        return next(blocked_duplicate_pairs(patients), None) is not None
    if kind == "cross-source-clinical-risk":
        return any(evaluate(name, resources) for name in (
            "renal-medication-risk", "bleeding-risk", "allergy-treatment-risk",
        ))
    return any(detects(resource, kind) for resource in resources)


def quality_variant_evidence(resources: list[dict], kind: str, *, anchor: str | None = None) -> set[str]:
    """Observable quality conclusions; no problem labels or answer-key input."""
    found = set()
    if kind == "timestamp-order":
        births = [row["birthDate"] for row in resources if row.get("resourceType") == "Patient" and row.get("birthDate")]
        for row in resources:
            effective = row.get("effectiveDateTime")
            if not effective:
                continue
            if detects(row, kind):
                found.add("issued-before-effective")
            if anchor and _instant(effective) > _instant(anchor):
                found.add("future-result")
            if births and effective[:10] < min(births):
                found.add("pre-birth-event")
    elif kind == "unit-mismatch":
        for row in resources:
            if detects(row, kind):
                quantity = row.get("valueQuantity", {})
                code = next((c for system, c in _codes(row.get("code")) if system == "http://loinc.org"), None)
                found.add("wrong-unit-family" if code in ANALYTE_UNITS and quantity.get("code") not in ANALYTE_UNITS[code] else "display-code-mismatch")
        for left in resources:
            targets = {row.get("reference") for row in left.get("derivedFrom", [])}
            a = normalized_quantity(left)
            for right in resources:
                if not targets.intersection({right.get("_fullUrl"), right.get("id")}):
                    continue
                b = normalized_quantity(right)
                if a and b and a[0] == b[0] and left.get("valueQuantity", {}).get("code") != right.get("valueQuantity", {}).get("code"):
                    if abs(a[1] - b[1]) > max(.02, abs(b[1]) * .02):
                        found.add("scale-mismatch")
    elif kind == "conflicting-diagnosis":
        conditions = [row for row in resources if row.get("resourceType") == "Condition"]
        for pos, left in enumerate(conditions):
            for right in conditions[pos + 1:]:
                a, b = set(_codes(left.get("code"))), set(_codes(right.get("code")))
                ls = {code for _, code in _codes(left.get("clinicalStatus"))}
                rs = {code for _, code in _codes(right.get("clinicalStatus"))}
                codes = {code for _, code in a | b}
                if "active" in ls & rs and any(group <= codes for group in MUTUALLY_EXCLUSIVE_DIAGNOSIS_FAMILIES):
                    found.add("code-family-conflict")
                same_assertion = bool(a & b) and bool({(row.get("system"), row.get("value")) for row in left.get("identifier", [])} & {(row.get("system"), row.get("value")) for row in right.get("identifier", [])}) and left.get("recordedDate") == right.get("recordedDate")
                if same_assertion:
                    if ("active" in ls and "resolved" in rs) or ("resolved" in ls and "active" in rs):
                        found.add("status-lifecycle-conflict")
                    if left.get("onsetDateTime") and right.get("onsetDateTime") and left["onsetDateTime"] != right["onsetDateTime"]:
                        found.add("onset-provenance-conflict")
    elif kind == "discordant-lab":
        labs = [row for row in resources if row.get("resourceType") == "Observation" and row.get("status") == "final"]
        for pos, left in enumerate(labs):
            for right in labs[pos + 1:]:
                a, b = normalized_quantity(left), normalized_quantity(right)
                if not a or not b or a[0] != b[0] or not left.get("effectiveDateTime") or not right.get("effectiveDateTime"):
                    continue
                if left.get("derivedFrom") or right.get("derivedFrom"):
                    continue
                if not left.get("specimen") or left.get("specimen") != right.get("specimen"):
                    continue
                hours = abs((_instant(left["effectiveDateTime"]) - _instant(right["effectiveDateTime"])).total_seconds()) / 3600
                if hours > _SEMANTICS["labComparisonMaximumHours"] or abs(a[1] - b[1]) + 1e-9 < ANALYTE_DISCORDANCE[a[0]]:
                    continue
                if left.get("valueQuantity", {}).get("code") != right.get("valueQuantity", {}).get("code"):
                    found.add("unit-normalized-disagreement")
                elif left.get("method") != right.get("method"):
                    found.add("near-time-method-disagreement")
                else:
                    found.add("same-specimen-disagreement")
    return found
