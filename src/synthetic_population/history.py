from __future__ import annotations

from collections import Counter, defaultdict
import math
from datetime import date
from typing import Any

from .observable import patient_graphs
from .temporal import ClinicalInterval, ClinicalOccurrence, clinical_occurrences, history_decision, parse_fhir_time


INTENSIVE = {"chronic-care", "medication-intensive", "oncology", "rare-complex"}


def build_history_audit(
    bundles: dict[str, dict[str, Any]], crosswalk: dict[str, Any], spec: dict[str, Any],
    ledger: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    graphs, patients = patient_graphs(bundles, spec, include_demographics=False)
    anchor = date.fromisoformat(spec["timelineAnchor"][:10])
    rows = []
    intensive_failures = []
    eligible = 0
    multiyear_passed = 0
    exclusions: Counter[str] = Counter()
    strata: dict[str, Counter[str]] = defaultdict(Counter)
    spans: Counter[int] = Counter()
    offsets: Counter[int] = Counter()
    latent_by_cohort: dict[str, list[ClinicalOccurrence]] = defaultdict(list)
    for item in ledger or []:
        if item.get("rowType") == "latent-event" and item.get("eventRole") not in {"condition"}:
            interval = parse_fhir_time(item["occurrence"]["start"])
            latent_by_cohort[item["cohortId"]].append(ClinicalOccurrence(
                "latent-event", item["eventRole"], interval,
                resource_id=item["eventId"], latent_event_id=item["eventId"],
            ))
        elif item.get("rowType") == "supplemental-source-capture" and item.get("resourceType") not in {"Condition", "AllergyIntolerance"} and item.get("occurrence"):
            occurrence = item["occurrence"]
            start = parse_fhir_time(occurrence["start"])
            end = parse_fhir_time(occurrence["end"])
            interval = ClinicalInterval(start.start, end.start, start.precision)
            latent_by_cohort[item["cohortId"]].append(ClinicalOccurrence(
                item["resourceType"], "supplemental-source-capture", interval,
                resource_id=item["resourceFullUrl"],
            ))
    latent_mismatches = []

    def age_band(age_days: int) -> str:
        if age_days < 730:
            return "under-2"
        if age_days < 18 * 365:
            return "2-17"
        if age_days < 65 * 365:
            return "18-64"
        return "65-plus"

    def add_stratum(name: str, key: str, one_year: dict[str, Any], multi_year: dict[str, Any], is_eligible: bool) -> None:
        bucket = strata[f"{name}:{key}"]
        bucket["patients"] += 1
        bucket["oneYearPassed"] += int(one_year["passed"])
        bucket["multiYearEligible"] += int(is_eligible)
        bucket["multiYearPassed"] += int(is_eligible and multi_year["passed"])
        bucket["spanDaysTotal"] += multi_year["spanDays"]
    for declared in crosswalk["patients"]:
        cohort = declared["cohortId"]
        patient = patients[cohort]
        born = date.fromisoformat(patient["birthDate"])
        age_days = (anchor - born).days
        occurrences = clinical_occurrences(graphs.get(cohort, []))
        one_year = history_decision(occurrences, 365)
        multi_year = history_decision(occurrences, 730)
        is_eligible = age_days >= 730
        if is_eligible:
            eligible += 1
            multiyear_passed += int(multi_year["passed"])
        else:
            exclusions["younger-than-730-days"] += 1
        if declared["primaryArchetype"] in INTENSIVE and not one_year["passed"]:
            intensive_failures.append(cohort)
        participation = "+".join(sorted({resource["_source"] for resource in graphs.get(cohort, [])})) or "none"
        add_stratum("archetype", declared["primaryArchetype"], one_year, multi_year, is_eligible)
        add_stratum("variant", declared.get("pathwayVariant", "undeclared"), one_year, multi_year, is_eligible)
        add_stratum("ageBand", age_band(age_days), one_year, multi_year, is_eligible)
        add_stratum("sourceParticipation", participation, one_year, multi_year, is_eligible)
        if multi_year["passed"]:
            spans[multi_year["spanDays"]] += 1
        event_dates = sorted({row.interval.start.date() for row in occurrences})
        offsets.update((right - left).days for left, right in zip(event_dates, event_dates[1:]) if right > left)
        latent = history_decision(latent_by_cohort.get(cohort, []), 730) if ledger is not None else None
        if ledger is not None and declared.get("ordinaryPathwayEligible", True) and (
            latent["eventCount"] != multi_year["eventCount"]
            or latent["spanDays"] != multi_year["spanDays"]
            or latent["passed"] != multi_year["passed"]
        ):
            latent_mismatches.append({
                "cohortId": cohort,
                "fhir": {key: multi_year[key] for key in ("eventCount", "spanDays", "passed")},
                "latent": {key: latent[key] for key in ("eventCount", "spanDays", "passed")},
            })
        rows.append({
            "cohortId": cohort, "archetype": declared["primaryArchetype"],
            "variant": declared.get("pathwayVariant"), "ageDays": age_days,
            "multiYearEligible": is_eligible, "sourceParticipation": participation,
            "oneYear": one_year, "multiYear": multi_year,
            **({"latentMultiYear": latent} if latent is not None else {}),
        })
    rate = multiyear_passed / eligible if eligible else 0.0
    passing_denominator = sum(spans.values())
    pair_denominator = sum(offsets.values())
    maximum_span_share = max(spans.values(), default=0) / max(1, passing_denominator)
    maximum_offset_share = max(offsets.values(), default=0) / max(1, pair_denominator)
    canonical = spec["populationSize"] >= 1000

    def quantiles(values: list[int]) -> dict[str, int]:
        ordered = sorted(values)
        return {
            label: ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)] if ordered else 0
            for label, fraction in (("p05", .05), ("p50", .50), ("p95", .95))
        }

    stratum_rows = {}
    for key, value in sorted(strata.items()):
        row = dict(value)
        row["multiYearRate"] = round(row["multiYearPassed"] / max(1, row["multiYearEligible"]), 6)
        stratum_rows[key] = row
    failures = []
    if intensive_failures:
        failures.append("one-year-intensive gate failed")
    if rate < .20:
        failures.append("multi-year rate below 20%")
    if latent_mismatches:
        failures.append("FHIR and latent history decisions differ")
    if canonical and maximum_span_share > .10:
        failures.append("exact non-protocol span concentration exceeds 10%")
    if canonical and maximum_offset_share > .20:
        failures.append("exact inter-event offset concentration exceeds 20%")
    return {
        "schemaVersion": "4.0.0", "registry": "explicit-primary-clinical-occurrence-v1",
        "totalPopulation": len(rows), "multiYearEligible": eligible,
        "multiYearPassed": multiyear_passed, "multiYearRate": round(rate, 6),
        "excludedCount": sum(exclusions.values()), "exclusionReasons": dict(sorted(exclusions.items())),
        "intensivePatientCount": sum(row["primaryArchetype"] in INTENSIVE for row in crosswalk["patients"]),
        "intensiveFailureCount": len(intensive_failures), "intensiveFailures": intensive_failures,
        "strata": stratum_rows,
        "spanQuantiles": quantiles([row["multiYear"]["spanDays"] for row in rows]),
        "exactSpanConcentration": {"denominator": passing_denominator, "counts": {str(key): value for key, value in sorted(spans.items())}, "maximumShare": round(maximum_span_share, 6)},
        "exactOffsetConcentration": {"denominator": pair_denominator, "counts": {str(key): value for key, value in sorted(offsets.items())}, "maximumShare": round(maximum_offset_share, 6)},
        "latentComparison": {"ordinaryMismatchCount": len(latent_mismatches), "mismatches": latent_mismatches},
        "patients": rows,
        "failureCount": len(failures), "failures": failures,
        "status": "pass" if not failures else "fail",
    }
