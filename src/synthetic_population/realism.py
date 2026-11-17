from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from math import isfinite, log2
from typing import Any, Callable, Iterable

from .catalogs import default_detector_catalog, load_catalog
from .clinical_catalog import age_applicability, birth_date_or_none, completed_age_years, unresolved_age_applicability
from .temporal import parse_fhir_time


PROJECT_TAG_PREFIX = "urn:hdp-test-bundle:synthetic"
_DETECTOR_SEMANTICS = load_catalog(default_detector_catalog())
_SNOMED_FAMILIES = _DETECTOR_SEMANTICS["snomedSemanticFamilies"]


def _codings(value: Any) -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        if isinstance(value.get("system"), str) and isinstance(value.get("code"), str):
            yield value["system"], value["code"]
        for child in value.values():
            yield from _codings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _codings(child)


def _references(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        if isinstance(value.get("reference"), str):
            yield value["reference"]
        for child in value.values():
            yield from _references(child)
    elif isinstance(value, list):
        for child in value:
            yield from _references(child)


def _status(resource: dict[str, Any]) -> tuple[str, ...]:
    values = [str(resource.get("status", ""))]
    for field in ("clinicalStatus", "verificationStatus"):
        values.extend(code for _system, code in _codings(resource.get(field)))
    return tuple(sorted(value for value in values if value))


def _clinical_codes(resource: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    fields = {
        key: resource.get(key)
        for key in ("code", "reasonCode", "medicationCodeableConcept", "vaccineCode", "category", "class", "type")
    }
    return tuple(sorted(set(_codings(fields))))


def _resource_signature(resource: dict[str, Any], reference_types: tuple[str, ...]) -> tuple[Any, ...]:
    """Transport-, date-, number-, text-, display-, and project-tag-free signature."""
    return (
        resource["resourceType"],
        tuple(profile.rsplit("/", 1)[-1] for profile in resource.get("meta", {}).get("profile", [])),
        _status(resource),
        _clinical_codes(resource),
        tuple(sorted({
            _SNOMED_FAMILIES[code]
            for system, code in _clinical_codes(resource)
            if system == "http://snomed.info/sct" and code in _SNOMED_FAMILIES
        })),
        tuple(sorted(reference_types)),
        (
            "quantity" if "valueQuantity" in resource else
            "coded-value" if "valueCodeableConcept" in resource else
            "components" if resource.get("component") else
            "no-value"
        ),
        tuple(sorted(
            key for key in (
                "abatement" if any(key.startswith("abatement") for key in resource) else None,
                "effective-period" if resource.get("effectivePeriod") else None,
                "dosage" if resource.get("dosage") or resource.get("dosageInstruction") else None,
                "course" if resource.get("courseOfTherapyType") or any(
                    extension.get("url", "").endswith("course-of-therapy-type")
                    for extension in resource.get("extension", [])
                ) else None,
                "specimen" if resource.get("specimen") else None,
                "based-on" if resource.get("basedOn") else None,
                "has-member" if resource.get("hasMember") else None,
                "method" if resource.get("method") else None,
                "reaction" if resource.get("reaction") else None,
            ) if key
        )),
        tuple(sorted({
            (quantity.get("system"), quantity.get("code"))
            for quantity in [
                resource.get("valueQuantity", {}),
                *[
                    component.get("valueQuantity", {})
                    for component in resource.get("component", [])
                ],
            ]
            if quantity.get("system") or quantity.get("code")
        })),
    )


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(raw.encode()).hexdigest()


def _frequency_metrics(rows: Iterable[Any], *, sequence: bool = False) -> dict[str, Any]:
    """Return normalized frequency and pairwise near-duplicate evidence.

    A near duplicate differs in exactly one normalized top-level component.  For
    sequence grains, insertion or removal of one normalized event also counts.
    IDs, dates, values, displays, notes, and project tags have already been
    removed by the signature builder before this function is called.
    """
    materialized = list(rows)
    counts = Counter(_digest(row) for row in materialized)
    representatives = {_digest(row): row for row in materialized}
    total = sum(counts.values())
    entropy = -sum((count / total) * log2(count / total) for count in counts.values()) if total else 0.0
    total_pairs = total * (total - 1) // 2
    exact_pairs = sum(count * (count - 1) // 2 for count in counts.values())

    substitution_buckets: dict[tuple[Any, ...], list[tuple[str, int]]] = defaultdict(list)
    for row_digest, count in counts.items():
        row = representatives[row_digest]
        parts = tuple(_digest(part) for part in row) if isinstance(row, (tuple, list)) else (_digest(row),)
        for position in range(len(parts)):
            substitution_buckets[(len(parts), position, *parts[:position], *parts[position + 1:])].append(
                (row_digest, count)
            )
    substitution_pairs = 0
    for bucket in substitution_buckets.values():
        if len(bucket) < 2:
            continue
        bucket_total = sum(count for _row_digest, count in bucket)
        substitution_pairs += (
            bucket_total * bucket_total - sum(count * count for _row_digest, count in bucket)
        ) // 2

    insertion_pairs = 0
    if sequence:
        for longer_digest, longer_count in counts.items():
            longer = representatives[longer_digest]
            if not isinstance(longer, (tuple, list)) or not longer:
                continue
            shorter_digests = {
                _digest(tuple(longer[:position]) + tuple(longer[position + 1:]))
                for position in range(len(longer))
            }
            insertion_pairs += sum(
                longer_count * counts[shorter_digest]
                for shorter_digest in shorter_digests
                if shorter_digest in counts
            )
    near_pairs = substitution_pairs + insertion_pairs
    return {
        "count": total,
        "uniqueCount": len(counts),
        "topShare": round(max(counts.values(), default=0) / max(1, total), 6),
        "shannonEntropyBits": round(entropy, 6),
        "exactDuplicateRate": round(1 - len(counts) / max(1, total), 6),
        "pairCount": total_pairs,
        "exactDuplicatePairCount": exact_pairs,
        "nearDuplicatePairCount": near_pairs,
        "nearDuplicateSubstitutionPairCount": substitution_pairs,
        "nearDuplicateInsertionRemovalPairCount": insertion_pairs,
        "pairwiseNearDuplicateRate": round(near_pairs / max(1, total_pairs), 6),
        "nearDuplicateDefinition": "one normalized component substitution or, for sequence grains, one event insertion/removal",
    }


def _numeric_summary(values: Iterable[float | int]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "minimum": None, "p25": None, "median": None, "p75": None, "maximum": None}

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(len(ordered) - 1, lower + 1)
        weight = position - lower
        return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)

    return {
        "count": len(ordered), "minimum": ordered[0], "p25": percentile(.25),
        "median": percentile(.5), "p75": percentile(.75), "maximum": ordered[-1],
    }


def _pearson(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs = [row[0] for row in pairs]
    ys = [row[1] for row in pairs]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_scale = sum((x - x_mean) ** 2 for x in xs) ** .5
    y_scale = sum((y - y_mean) ** 2 for y in ys) ** .5
    if not x_scale or not y_scale:
        return None
    return numerator / (x_scale * y_scale)


def _tagged(resource: dict[str, Any], system: str) -> bool:
    return any(tag.get("system") == system for tag in resource.get("meta", {}).get("tag", []))


def build_realism_audit(bundles: dict[str, dict[str, Any]], crosswalk: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    cohort_system = spec["identity"]["cohortIdentifierSystem"]
    url_to_cohort: dict[tuple[str, str], str] = {}
    url_to_type: dict[tuple[str, str], str] = {}
    patient_resources: dict[str, dict[str, Any]] = {}
    cohort_archetype = {row["cohortId"]: row["primaryArchetype"] for row in crosswalk["patients"]}
    cohort_secondary = {row["cohortId"]: row.get("secondaryTrait", "none") for row in crosswalk["patients"]}
    content = load_catalog(spec["remediation"]["clinicalContentCatalog"]) if spec.get("remediation") else {"modelDefinitions": {}}
    resources: list[tuple[str, str, dict[str, Any]]] = []
    for organization, bundle in bundles.items():
        for entry in bundle["entry"]:
            resource = entry["resource"]
            url_to_type[(organization, entry["fullUrl"])] = resource["resourceType"]
            if resource["resourceType"] == "Patient":
                cohort = next((row.get("value") for row in resource.get("identifier", []) if row.get("system") == cohort_system), None)
                if cohort:
                    url_to_cohort[(organization, entry["fullUrl"])] = cohort
                    patient_resources.setdefault(cohort, resource)
            resources.append((organization, entry["fullUrl"], resource))

    by_cohort: dict[str, list[tuple[str, tuple[Any, ...]]]] = defaultdict(list)
    source_rows: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    source_patients: dict[str, set[str]] = defaultdict(set)
    capture_counts: dict[str, Counter[str]] = defaultdict(Counter)
    reason_counts: dict[str, Counter[str]] = defaultdict(Counter)
    reason_counts_by_class: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    encounter_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    event_dates_by_cohort: dict[str, set[date]] = defaultdict(set)
    event_dates_by_source_cohort: dict[tuple[str, str], set[date]] = defaultdict(set)
    documentation_lags: dict[str, list[float]] = defaultdict(list)
    condition_codes_by_cohort: dict[str, set[tuple[str, str]]] = defaultdict(set)
    medication_codes_by_cohort: dict[str, set[tuple[str, str]]] = defaultdict(set)
    lisinopril_orders: dict[str, list[tuple[datetime, datetime | None]]] = defaultdict(list)
    hypertension_onsets: dict[str, list[datetime]] = defaultdict(list)
    episode_rows: dict[tuple[str, str], list[tuple[Any, ...]]] = defaultdict(list)
    event_rows: list[tuple[Any, ...]] = []
    problem_event_rows: list[tuple[Any, ...]] = []
    scenario_event_rows: list[tuple[Any, ...]] = []
    problem_variant_rows: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    scenario_variant_rows: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    problem_contexts: dict[str, set[tuple[str, str]]] = defaultdict(set)
    scenario_contexts: set[str] = set()
    ordinary_event_rows: list[tuple[Any, ...]] = []
    encounter_durations: list[float] = []
    event_offsets: list[int] = []
    laboratory_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    laboratory_counts_by_patient: dict[str, Counter[str]] = defaultdict(Counter)
    for organization, full_url, resource in resources:
        cohort = None
        for reference in _references(resource):
            cohort = cohort or url_to_cohort.get((organization, reference))
        if resource["resourceType"] == "Patient":
            cohort = url_to_cohort.get((organization, full_url))
        if not cohort or resource["resourceType"] in {"Patient", "RelatedPerson"}:
            continue
        if any(tag.get("system") == "urn:hdp-test-bundle:synthetic-archetype" and tag.get("code") == "compact-coverage" for tag in resource.get("meta", {}).get("tag", [])):
            continue
        reference_types = tuple(
            url_to_type[(organization, reference)]
            for reference in _references(resource)
            if (organization, reference) in url_to_type
        )
        signature = _resource_signature(resource, reference_types)
        event_counts[cohort] += 1
        by_cohort[cohort].append((organization, signature))
        source_rows[organization].append(signature)
        episode_rows[(cohort, organization)].append(signature)
        event_rows.append(signature)
        problem_system = spec["metadata"]["problemMarkerSystem"]
        scenario_system = "urn:hdp-test-bundle:synthetic-scenario-role"
        if _tagged(resource, problem_system):
            problem_event_rows.append(signature)
            for problem_id in {
                tag.get("code") for tag in resource.get("meta", {}).get("tag", [])
                if tag.get("system") == spec["metadata"]["problemIdSystem"] and tag.get("code")
            }:
                problem_variant_rows[problem_id].append(signature)
                problem_contexts[problem_id].add((cohort, organization))
        elif _tagged(resource, scenario_system):
            scenario_event_rows.append(signature)
            scenario_variant_rows[cohort].append(signature)
            scenario_contexts.add(cohort)
        else:
            ordinary_event_rows.append(signature)
        source_patients[organization].add(cohort)
        if resource["resourceType"] == "Encounter":
            encounter_counts[cohort] += 1
            reasons = [code for system, code in _codings(resource.get("reasonCode")) if system == "http://snomed.info/sct"]
            reason_counts[organization].update(reasons)
            encounter_class = resource.get("class", {}).get("code", "unknown")
            reason_counts_by_class[organization][encounter_class].update(reasons)
            capture_counts[organization].update(
                tag.get("code") for tag in resource.get("meta", {}).get("tag", [])
                if tag.get("system") == "urn:hdp-test-bundle:synthetic-capture-variant"
            )
            period = resource.get("period", {})
            if period.get("start") and period.get("end"):
                start = datetime.fromisoformat(period["start"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(period["end"].replace("Z", "+00:00"))
                encounter_durations.append(round((end - start).total_seconds() / 60, 3))
                event_dates_by_cohort[cohort].update({start.date(), end.date()})
                event_dates_by_source_cohort[(organization, cohort)].update({start.date(), end.date()})
        if resource["resourceType"] == "Condition":
            condition_codes_by_cohort[cohort].update(_clinical_codes(resource))
            if ("http://snomed.info/sct", "38341003") in set(_codings(resource.get("code"))) \
                    and any(code == "active" for _system, code in _codings(resource.get("clinicalStatus"))):
                instant = resource.get("onsetDateTime") or resource.get("recordedDate")
                if instant:
                    hypertension_onsets[cohort].append(parse_fhir_time(instant).start)
        if resource["resourceType"] in {"MedicationRequest", "MedicationStatement", "MedicationAdministration", "MedicationDispense"}:
            medication_codes_by_cohort[cohort].update(_clinical_codes(resource))
        if resource["resourceType"] == "MedicationRequest" and resource.get("status") == "active" \
                and ("http://www.nlm.nih.gov/research/umls/rxnorm", "29046") in set(_codings(resource.get("medicationCodeableConcept"))) \
                and not _tagged(resource, problem_system) and not _tagged(resource, scenario_system):
            repeat = next((item.get("timing", {}).get("repeat", {}) for item in resource.get("dosageInstruction", [])
                           if item.get("timing", {}).get("repeat", {}).get("boundsPeriod")), {})
            bounds = repeat.get("boundsPeriod", {})
            if bounds.get("start"):
                start = parse_fhir_time(bounds["start"]).start
                end = parse_fhir_time(bounds["end"]).end if bounds.get("end") else None
                lisinopril_orders[cohort].append((start, end))
        if resource["resourceType"] == "Observation" and isinstance(resource.get("valueQuantity", {}).get("value"), (int, float)):
            loinc = next((code for system, code in _codings(resource.get("code")) if system == "http://loinc.org"), None)
            if loinc:
                quantity = resource["valueQuantity"]
                reference = resource.get("referenceRange", [{}])[0]
                low = reference.get("low", {}).get("value")
                high = reference.get("high", {}).get("value")
                value = float(quantity["value"])
                interpretation = next((code for system, code in _codings(resource.get("interpretation")) if system == "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation"), None)
                birth = birth_date_or_none(patient_resources[cohort])
                observation_date = date.fromisoformat(resource["effectiveDateTime"][:10]) if resource.get("effectiveDateTime") else None
                age_years = completed_age_years(birth, observation_date) if birth and observation_date else None
                applicability = age_applicability(content, loinc, age_years) if age_years is not None else unresolved_age_applicability(loinc)
                intentional = _tagged(resource, problem_system) or _tagged(resource, scenario_system)
                trajectory = next((
                    tag.get("code") for tag in resource.get("meta", {}).get("tag", [])
                    if tag.get("system") == "urn:hdp-test-bundle:synthetic-trajectory-class"
                ), None)
                laboratory_rows[loinc].append({
                    "cohort": cohort, "source": organization, "value": value,
                    "unit": quantity.get("code"), "low": low, "high": high,
                    "interpretation": interpretation, "intentional": intentional,
                    "status": resource.get("status"),
                    "eventTime": resource.get("effectiveDateTime"),
                    "trajectoryClass": trajectory,
                    "ageYears": age_years,
                    "ageApplicability": applicability,
                })
                if not intentional:
                    laboratory_counts_by_patient[cohort][loinc] += 1
        dates = []
        for key in ("effectiveDateTime", "issued", "authoredOn", "recordedDate", "occurrenceDateTime", "performedDateTime"):
            if isinstance(resource.get(key), str) and len(resource[key]) >= 10:
                dates.append(date.fromisoformat(resource[key][:10]))
        event_dates_by_cohort[cohort].update(dates)
        event_dates_by_source_cohort[(organization, cohort)].update(dates)
        effective = resource.get("effectiveDateTime")
        issued = resource.get("issued")
        if effective and issued:
            effective_instant = datetime.fromisoformat(effective.replace("Z", "+00:00"))
            issued_instant = datetime.fromisoformat(issued.replace("Z", "+00:00"))
            if effective_instant.tzinfo is None:
                effective_instant = effective_instant.replace(tzinfo=timezone.utc)
            if issued_instant.tzinfo is None:
                issued_instant = issued_instant.replace(tzinfo=timezone.utc)
            documentation_lags[organization].append(round((issued_instant - effective_instant).total_seconds() / 60, 3))
    # Inter-event timing is measured across each patient's actual event stream,
    # not between fields of a single resource.
    for dates in event_dates_by_cohort.values():
        ordered = sorted(dates)
        event_offsets.extend((right - left).days for left, right in zip(ordered, ordered[1:]) if right != left)
    source_event_offsets: dict[str, list[int]] = defaultdict(list)
    for (source, _cohort), dates in event_dates_by_source_cohort.items():
        ordered = sorted(dates)
        source_event_offsets[source].extend(
            (right - left).days for left, right in zip(ordered, ordered[1:]) if right != left
        )

    patient_signatures = {
        cohort: _digest(sorted(rows, key=lambda row: json.dumps(row, default=list)))
        for cohort, rows in by_cohort.items()
    }
    signature_counts = Counter(patient_signatures.values())
    patient_signature_metrics = _frequency_metrics(
        (
            tuple(sorted(rows, key=lambda row: json.dumps(row, default=list)))
            for rows in by_cohort.values()
        ),
        sequence=True,
    )
    archetype_signatures: dict[str, Counter[str]] = defaultdict(Counter)
    for cohort, signature in patient_signatures.items():
        archetype_signatures[cohort_archetype[cohort]][signature] += 1
    total = max(1, len(crosswalk["patients"]))
    entropy = -sum((count / total) * log2(count / total) for count in signature_counts.values())
    source_metrics = {}
    realism_catalog = load_catalog(spec["clinicalRealism"]["catalog"])
    capture_catalog = load_catalog(realism_catalog["sourceCaptureCatalog"])
    workflow_variant_map = {
        source: {
            family: {
                branch["id"] for branch in source_row["branches"]
                if branch["workflowFamily"] == family
            }
            for family in capture_catalog["requiredWorkflowFamilies"][source]
        }
        for source, source_row in capture_catalog["sources"].items()
    }
    for source in ("hmo", "hospital-a", "hospital-b"):
        reasons = reason_counts[source]
        encounters = sum(reasons.values())
        captures = capture_counts[source]
        active_count = len(source_patients[source])
        source_metrics[source] = {
            "clinicallyActivePatients": len(source_patients[source]),
            "demographicsOnlyPatients": total - len(source_patients[source]),
            "reasonFamilyCount": len(reasons),
            "reasonCounts": dict(sorted(reasons.items())),
            "topReasonShare": round(max(reasons.values(), default=0) / encounters, 6) if encounters else 0.0,
            "reasonFamiliesByEncounterClass": {
                encounter_class: {
                    "encounterCount": sum(counts.values()),
                    "reasonFamilyCount": len(counts),
                    "topReasonShare": round(max(counts.values(), default=0) / max(1, sum(counts.values())), 6),
                }
                for encounter_class, counts in sorted(reason_counts_by_class[source].items())
            },
            "captureVariantCounts": dict(sorted(captures.items())),
            "captureVariantRates": {
                name: round(count / max(1, active_count), 6)
                for name, count in sorted(captures.items())
            },
            "workflowFamilyCounts": {
                family: sum(captures[name] for name in variants)
                for family, variants in workflow_variant_map[source].items()
            },
            "documentationLagMinutes": _numeric_summary(documentation_lags[source]),
            "resourceSignatureCount": len({_digest(row) for row in source_rows[source]}),
        }
    protocol_durations = {30.0, 45.0, 60.0, 75.0, 120.0, 180.0}
    protocol_offsets = {1, 7, 10, 11, 12, 13, 14, 30, 90, 365}
    non_protocol_durations = [value for value in encounter_durations if value not in protocol_durations]
    non_protocol_offsets = [value for value in event_offsets if value not in protocol_offsets]
    duration_counts = Counter(non_protocol_durations)
    offset_counts = Counter(non_protocol_offsets)
    grain_metrics = {
        "sourceLocalEpisodes": _frequency_metrics(
            (tuple(sorted(rows, key=lambda row: json.dumps(row, default=list))) for rows in episode_rows.values()),
            sequence=True,
        ),
        "clinicalEvents": _frequency_metrics(event_rows),
        "ordinaryClinicalEvents": _frequency_metrics(ordinary_event_rows),
        "intentionalProblemEvents": _frequency_metrics(problem_event_rows),
        "crossSourceScenarioEvents": _frequency_metrics(scenario_event_rows),
        "intentionalProblemVariants": _frequency_metrics(
            (
                tuple(sorted(
                    [
                        row for context in sorted(problem_contexts[problem_id])
                        for row in episode_rows[context]
                    ],
                    key=lambda row: json.dumps(row, default=list),
                ))
                for problem_id in problem_variant_rows
            ),
            sequence=True,
        ),
        "crossSourceScenarioVariants": _frequency_metrics(
            (
                tuple(sorted(
                    [
                        row for source in ("hmo", "hospital-a", "hospital-b")
                        for row in episode_rows[(cohort, source)]
                    ],
                    key=lambda row: json.dumps(row, default=list),
                ))
                for cohort in sorted(scenario_contexts)
            ),
            sequence=True,
        ),
    }
    guards = load_catalog(spec["clinicalDistributions"]["catalog"])["guardrails"]
    analytes = {}
    for loinc, rows in sorted(laboratory_rows.items()):
        ordinary = [row for row in rows if not row["intentional"]]
        applicable = [
            row for row in ordinary
            if row["ageApplicability"].get("ordinaryDistributionGuardApplicable", True)
        ]
        age_excluded = [row for row in ordinary if row not in applicable]
        abnormal = [row for row in applicable if row["low"] is not None and row["high"] is not None and not row["low"] <= row["value"] <= row["high"]]
        abnormal_values = Counter(row["value"] for row in abnormal)
        guard = guards.get(loinc, {})
        age_excluded_metadata_violations = sum(
            (
                (row["ageApplicability"].get("referenceRangeHandling") == "omit"
                 and (row["low"] is not None or row["high"] is not None))
                or (row["ageApplicability"].get("referenceRangeHandling") == "required"
                    and (row["low"] is None or row["high"] is None))
                or (row["ageApplicability"].get("interpretationHandling") == "omit" and row["interpretation"] is not None)
                or (row["ageApplicability"].get("interpretationHandling") == "required" and row["interpretation"] is None)
            )
            for row in age_excluded
        )
        age_excluded_safety_violations = sum(
            row["ageApplicability"].get("requiresPositiveFiniteValue")
            and (not isfinite(row["value"]) or row["value"] <= 0)
            for row in age_excluded
        )
        analytes[loinc] = {
            "count": len(rows), "ordinaryCount": len(ordinary), "ordinaryAbnormalCount": len(abnormal),
            "applicableOrdinaryCount": len(applicable), "ageExcludedOrdinaryCount": len(age_excluded),
            "ageApplicabilityCounts": dict(sorted(Counter(row["ageApplicability"].get("id", "unknown") for row in ordinary).items())),
            "ordinaryAbnormalRate": round(len(abnormal) / len(applicable), 6) if applicable else None,
            "ordinaryAbnormalRateStandardError": round(
                ((len(abnormal) / max(1, len(applicable))) * (1 - len(abnormal) / max(1, len(applicable))) / max(1, len(applicable))) ** .5,
                6,
            ),
            "distinctOrdinaryAbnormalValues": len(abnormal_values),
            "topOrdinaryAbnormalPointMassShare": round(max(abnormal_values.values(), default=0) / max(1, len(abnormal)), 6),
            "units": sorted({row["unit"] for row in ordinary}),
            "ordinaryUnitCodes": sorted({row["unit"] for row in ordinary}),
            "ordinaryStatuses": sorted({row["status"] for row in ordinary}),
            "ordinaryEventTimeRate": round(sum(bool(row["eventTime"]) for row in ordinary) / max(1, len(ordinary)), 6),
            "completeMetadataRate": round(sum(
                row["unit"] == guard.get("unit") and row["low"] == guard.get("referenceLow")
                and row["high"] == guard.get("referenceHigh") and row["interpretation"] in {"N", "H", "L"}
                and row["status"] in guard.get("allowedStatuses", ["final"])
                and bool(row["eventTime"])
                and (
                    (row["interpretation"] == "L" and row["value"] < row["low"])
                    or (row["interpretation"] == "H" and row["value"] > row["high"])
                    or (row["interpretation"] == "N" and row["low"] <= row["value"] <= row["high"])
                )
                for row in applicable
            ) / len(applicable), 6) if applicable else None,
            "physiologicBoundViolationCount": sum(
                row["value"] < guard.get("abnormalLow", float("-inf")) or row["value"] > guard.get("maximumUntaggedValue", guard.get("abnormalHigh", float("inf")))
                for row in applicable
            ),
            "ageExcludedReferenceMetadataViolationCount": age_excluded_metadata_violations,
            "ageExcludedDerivedSafetyViolationCount": age_excluded_safety_violations,
            "unknownAgeCount": sum(row["ageApplicability"].get("ageResolutionRequired", False) for row in ordinary),
        }
    anchor = date.fromisoformat(spec["timelineAnchor"][:10])
    cohort_age = {}
    for cohort, patient in patient_resources.items():
        born = birth_date_or_none(patient)
        cohort_age[cohort] = completed_age_years(born, anchor) if born is not None else None

    def actual_lisinopril_exposure(row: dict[str, Any]) -> bool:
        if row["cohort"] not in hypertension_onsets or not row.get("eventTime"):
            return False
        when = parse_fhir_time(row["eventTime"]).start
        return any(onset <= start <= when and (end is None or when <= end)
                   for onset in hypertension_onsets[row["cohort"]]
                   for start, end in lisinopril_orders[row["cohort"]])

    def subgroup_rate(loinc: str, selected: Callable[[dict[str, Any]], bool], direction: str) -> dict[str, Any]:
        rows = [row for row in laboratory_rows.get(loinc, []) if not row["intentional"]]
        selected_rows = [row for row in rows if selected(row)]
        comparator_rows = [row for row in rows if not selected(row)]

        def event(row: dict[str, Any]) -> bool:
            return row["value"] < row["low"] if direction == "low" else row["value"] > row["high"]

        selected_rate = sum(event(row) for row in selected_rows) / max(1, len(selected_rows))
        comparator_rate = sum(event(row) for row in comparator_rows) / max(1, len(comparator_rows))
        return {
            "direction": direction, "selectedCount": len(selected_rows), "comparatorCount": len(comparator_rows),
            "selectedRate": round(selected_rate, 6), "comparatorRate": round(comparator_rate, 6),
            "effectDifference": round(selected_rate - comparator_rate, 6),
        }

    correlations = {
        "oncology-low-hemoglobin": subgroup_rate("718-7", lambda row: cohort_archetype[row["cohort"]] == "oncology", "low"),
        "intensive-high-potassium": subgroup_rate("6298-4", lambda row: cohort_archetype[row["cohort"]] in {"chronic-care", "medication-intensive", "rare-complex"}, "high"),
        "older-high-creatinine": subgroup_rate(
            "2160-0", lambda row: cohort_age.get(row["cohort"]) is not None and cohort_age[row["cohort"]] >= 65, "high",
        ),
        "medication-exposure-high-potassium": subgroup_rate(
            "6298-4", lambda row: cohort_secondary.get(row["cohort"]) == "medication-exposure"
            and actual_lisinopril_exposure(row), "high",
        ),
    }
    ordinary_labs = [
        {**row, "loinc": loinc}
        for loinc, rows in laboratory_rows.items()
        for row in rows if not row["intentional"]
    ]
    trajectory_counts = Counter(row["trajectoryClass"] or "missing" for row in ordinary_labs)

    def trajectory_strata(key: Callable[[dict[str, Any]], str]) -> dict[str, dict[str, int]]:
        output: dict[str, Counter[str]] = defaultdict(Counter)
        for row in ordinary_labs:
            output[key(row)][row["trajectoryClass"] or "missing"] += 1
        return {name: dict(sorted(counts.items())) for name, counts in sorted(output.items())}

    age_band = {
        cohort: "unknown" if age is None else "child" if age < 18 else "adult" if age < 65 else "older-adult"
        for cohort, age in cohort_age.items()
    }
    gender_by_cohort = {cohort: patient.get("gender", "unknown") for cohort, patient in patient_resources.items()}
    normalized_by_cohort_analyte: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in ordinary_labs:
        if row["low"] is None or row["high"] is None or row["high"] == row["low"]:
            continue
        normalized_by_cohort_analyte[(row["cohort"], row["loinc"])].append(
            (row["value"] - (row["low"] + row["high"]) / 2) / (row["high"] - row["low"])
        )
    renal_pairs = []
    for cohort in cohort_archetype:
        potassium = normalized_by_cohort_analyte.get((cohort, "6298-4"))
        creatinine = normalized_by_cohort_analyte.get((cohort, "2160-0"))
        if potassium and creatinine:
            renal_pairs.append((sum(potassium) / len(potassium), sum(creatinine) / len(creatinine)))
    joint_families = {
        "renal-potassium": {
            "analytes": ["6298-4", "2160-0"],
            "pairedPatientCount": len(renal_pairs),
            "pearsonCorrelation": (
                round(value, 6) if (value := _pearson(renal_pairs)) is not None else None
            ),
        }
    }
    intensive = {cohort for cohort, archetype in cohort_archetype.items() if archetype in {"chronic-care", "medication-intensive", "oncology", "rare-complex"}}
    repeated_intensive = {
        cohort for cohort in intensive if any(count >= 2 for count in laboratory_counts_by_patient.get(cohort, {}).values())
    }
    spans_by_cohort = {
        cohort: (max(values) - min(values)).days if len(values) >= 2 else 0
        for cohort, values in event_dates_by_cohort.items()
    }
    spans_by_archetype = {
        archetype: _numeric_summary(
            spans_by_cohort.get(cohort, 0)
            for cohort, patient_archetype in cohort_archetype.items()
            if patient_archetype == archetype
        )
        for archetype in sorted(set(cohort_archetype.values()))
    }
    spans_by_source = {
        source: _numeric_summary(
            (max(values) - min(values)).days if len(values) >= 2 else 0
            for (row_source, _cohort), values in event_dates_by_source_cohort.items()
            if row_source == source
        )
        for source in ("hmo", "hospital-a", "hospital-b")
    }
    intensive_one_year_count = sum(spans_by_cohort.get(cohort, 0) >= 365 for cohort in intensive)
    overall_one_year_count = sum(spans_by_cohort.get(cohort, 0) >= 365 for cohort in cohort_archetype)
    utilization_patterns = Counter(
        "+".join(source for source in ("hmo", "hospital-a", "hospital-b") if cohort in source_patients[source])
        for cohort in cohort_archetype
    )
    comorbid = {cohort for cohort, codes in condition_codes_by_cohort.items() if len(codes) >= 2}
    polypharmacy = {cohort for cohort, codes in medication_codes_by_cohort.items() if len(codes) >= 3}
    result = {
        "schemaVersion": "1.0.0",
        "signatureNormalizationVersion": "1.0.0",
        "patientStructuralSignatures": {
            **patient_signature_metrics,
            "frequency": dict(sorted(signature_counts.items(), key=lambda item: (-item[1], item[0]))),
        },
        "archetypes": {
            archetype: {
                "patientCount": sum(counts.values()),
                "structuralVariantCount": len(counts),
                "topVariantShare": round(max(counts.values(), default=0) / max(1, sum(counts.values())), 6),
            }
            for archetype, counts in sorted(archetype_signatures.items())
        },
        "sources": source_metrics,
        "variabilityGrains": grain_metrics,
        "timingConcentration": {
            "encounterDurationCount": len(encounter_durations),
            "declaredProtocolDurationCount": len(encounter_durations) - len(non_protocol_durations),
            "eligibleNonProtocolDurationCount": len(non_protocol_durations),
            "topEncounterDurationShare": round(max(duration_counts.values(), default=0) / max(1, len(non_protocol_durations)), 6),
            "interEventOffsetCount": len(event_offsets),
            "declaredProtocolOffsetCount": len(event_offsets) - len(non_protocol_offsets),
            "eligibleNonProtocolOffsetCount": len(non_protocol_offsets),
            "topInterEventOffsetShare": round(max(offset_counts.values(), default=0) / max(1, len(non_protocol_offsets)), 6),
            "bySource": {
                source: {
                    "count": len(values),
                    "topExactOffsetShare": round(
                        max(Counter(values).values(), default=0) / max(1, len(values)), 6
                    ),
                }
                for source, values in sorted(source_event_offsets.items())
            },
        },
        "clinicalDistributions": {
            "analytes": analytes,
            "subgroupCorrelations": correlations,
            "jointFamilies": joint_families,
            "trajectoryClasses": {
                "overallCounts": dict(sorted(trajectory_counts.items())),
                "byAnalyte": trajectory_strata(lambda row: row["loinc"]),
                "byArchetype": trajectory_strata(lambda row: cohort_archetype[row["cohort"]]),
                "bySource": trajectory_strata(lambda row: row["source"]),
                "bySex": trajectory_strata(lambda row: gender_by_cohort.get(row["cohort"], "unknown")),
                "byAgeBand": trajectory_strata(lambda row: age_band.get(row["cohort"], "unknown")),
                "bySecondaryTrait": trajectory_strata(lambda row: cohort_secondary.get(row["cohort"], "none")),
            },
            "intensivePatientCount": len(intensive),
            "intensivePatientsWithRepeatedAnalyte": len(repeated_intensive),
            "intensiveRepeatedAnalyteRate": round(len(repeated_intensive) / max(1, len(intensive)), 6),
        },
        "episodes": {
            "patientsWithTwoOrMore": sum(count >= 2 for count in encounter_counts.values()),
            "shareWithTwoOrMore": round(sum(count >= 2 for count in encounter_counts.values()) / total, 6),
            "patientsWithThreeOrMoreEventDates": sum(
                len(event_dates_by_cohort.get(cohort, set())) >= 3 for cohort in cohort_archetype
            ),
            "shareWithThreeOrMoreEventDates": round(sum(
                len(event_dates_by_cohort.get(cohort, set())) >= 3 for cohort in cohort_archetype
            ) / total, 6),
            "countPercentiles": {
                "minimum": min(encounter_counts.values(), default=0),
                "median": sorted(encounter_counts.values())[len(encounter_counts) // 2] if encounter_counts else 0,
                "maximum": max(encounter_counts.values(), default=0),
            },
        },
        "eventCounts": _numeric_summary(event_counts.get(cohort, 0) for cohort in cohort_archetype),
        "sourceUtilization": {
            "patternCounts": dict(sorted(utilization_patterns.items())),
            "patternRates": {
                pattern: round(count / total, 6)
                for pattern, count in sorted(utilization_patterns.items())
            },
        },
        "temporalSpans": {
            "overallDays": _numeric_summary(spans_by_cohort.get(cohort, 0) for cohort in cohort_archetype),
            "byArchetypeDays": spans_by_archetype,
            "bySourceDays": spans_by_source,
            "oneYearOverallCount": overall_one_year_count,
            "oneYearOverallRate": round(overall_one_year_count / total, 6),
            "oneYearIntensiveCount": intensive_one_year_count,
            "oneYearIntensiveRate": round(intensive_one_year_count / max(1, len(intensive)), 6),
        },
        "comorbidityPolypharmacyOverlap": {
            "comorbidPatientCount": len(comorbid),
            "polypharmacyPatientCount": len(polypharmacy),
            "overlapPatientCount": len(comorbid & polypharmacy),
            "overlapRateAmongPolypharmacy": round(len(comorbid & polypharmacy) / max(1, len(polypharmacy)), 6),
        },
    }
    if spec.get("remediation"):
        from .clinical_state import population_state_audit
        result["clinicalState"] = population_state_audit(bundles, spec)
        from .medication_safety import population_medication_audit
        result["medicationSafety"] = population_medication_audit(bundles, spec)
        from .clinical_semantics import population_semantic_audit
        result["clinicalSemantics"] = population_semantic_audit(bundles, spec)
    from .clinical_policy import population_branch_quantitative_findings
    result["branchQuantitativeFindings"] = population_branch_quantitative_findings(bundles, content)
    result["diagnostics"] = realism_failures(result, spec)
    result["status"] = "pass" if not result["diagnostics"] else "fail"
    return result


def realism_failures(audit: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    if not spec.get("clinicalRealism"):
        return []
    errors = [row["reason"] for row in audit.get("branchQuantitativeFindings", [])]
    for finding in audit.get("clinicalState", {}).get("findings", []) + audit.get("medicationSafety", {}).get("findings", []) + audit.get("clinicalSemantics", {}).get("findings", []):
        errors.append(f"{finding['finding']} {finding['cohortId']}: {finding['reason']}")
    canonical = spec["populationSize"] >= 1000
    realism_catalog = load_catalog(spec["clinicalRealism"]["catalog"])
    capture_catalog = load_catalog(realism_catalog["sourceCaptureCatalog"])
    minimum_signatures = 75 if canonical else min(spec["populationSize"], 20)
    structural = audit["patientStructuralSignatures"]
    if structural["uniqueCount"] < minimum_signatures:
        errors.append(f"patient structural signatures {structural['uniqueCount']} < {minimum_signatures}")
    if canonical and structural["topShare"] > .10:
        errors.append("top patient structural signature exceeds 10%")
    for archetype, row in audit["archetypes"].items():
        minimum = min(4, row["patientCount"]) if archetype == "rare-complex" else min(5, row["patientCount"])
        if row["structuralVariantCount"] < minimum:
            errors.append(f"{archetype} has {row['structuralVariantCount']} structural variants, requires {minimum}")
        if canonical and row["topVariantShare"] > .35:
            errors.append(f"{archetype} top structural variant exceeds 35%")
    for source, row in audit["sources"].items():
        if canonical and row["reasonFamilyCount"] < 12:
            errors.append(f"{source} has fewer than 12 encounter-reason families")
        if row["topReasonShare"] > .35:
            errors.append(f"{source} top encounter reason exceeds 35%")
        if canonical and len(row["captureVariantCounts"]) < 8:
            errors.append(f"{source} does not exercise all capture variants")
        if canonical:
            declared_variants = {
                branch["id"] for branch in capture_catalog["sources"][source]["branches"]
            }
            observed_variants = set(row["captureVariantCounts"])
            if observed_variants != declared_variants:
                errors.append(
                    f"{source} capture branch mismatch: missing={sorted(declared_variants - observed_variants)} "
                    f"unknown={sorted(observed_variants - declared_variants)}"
                )
            minimum_count = capture_catalog["canonicalMinimumCountPerBranch"]
            minimum_rate = capture_catalog["canonicalRateBandPerBranch"]["minimum"]
            maximum_rate = capture_catalog["canonicalRateBandPerBranch"]["maximum"]
            for variant, count in row["captureVariantCounts"].items():
                rate = row["captureVariantRates"][variant]
                if count < minimum_count:
                    errors.append(f"{source}/{variant} capture count {count} < {minimum_count}")
                if not minimum_rate <= rate <= maximum_rate:
                    errors.append(f"{source}/{variant} capture rate {rate} outside frozen band")
            for family in capture_catalog["requiredWorkflowFamilies"][source]:
                if row["workflowFamilyCounts"].get(family, 0) < minimum_count:
                    errors.append(f"{source} workflow family {family} is not represented nontrivially")
            for encounter_class, class_row in row["reasonFamiliesByEncounterClass"].items():
                if class_row["encounterCount"] >= 50 and class_row["reasonFamilyCount"] < 5:
                    errors.append(f"{source}/{encounter_class} has fewer than five encounter reasons")
        if row["captureVariantCounts"] and set(row["captureVariantCounts"]) == {"community-lab"}:
            errors.append(f"{source} universally emits the five-resource template")
    if canonical and audit["episodes"]["shareWithTwoOrMore"] < .70:
        errors.append("fewer than 70% of people have two clinical encounters")
    if canonical and audit["episodes"]["shareWithThreeOrMoreEventDates"] < .70:
        errors.append("fewer than 70% of people have three actual clinical event dates")
    if canonical and audit["temporalSpans"]["oneYearOverallRate"] < .40:
        errors.append("fewer than 40% of people have one-year actual clinical event spans")
    if canonical and audit["temporalSpans"]["oneYearIntensiveRate"] < .90:
        errors.append("fewer than 90% of intensive-pathway people have one-year event spans")
    if canonical:
        utilization = audit["sourceUtilization"]["patternRates"]
        if utilization.get("hmo", 0) < .20:
            errors.append("fewer than 20% of people remain HMO-only")
        exactly_one_hospital = utilization.get("hmo+hospital-a", 0) + utilization.get("hmo+hospital-b", 0)
        if exactly_one_hospital < .10:
            errors.append("fewer than 10% of people use exactly one hospital")
        for source in ("hospital-a", "hospital-b"):
            if audit["sources"][source]["demographicsOnlyPatients"] / spec["populationSize"] < .20:
                errors.append(f"{source} has fewer than 20% demographics-only patients")
    timing = audit["timingConcentration"]
    if canonical and timing["topEncounterDurationShare"] > .20:
        errors.append("one non-protocol encounter duration exceeds the 20% concentration limit")
    if canonical and timing["interEventOffsetCount"] and timing["topInterEventOffsetShare"] > .20:
        errors.append("one inter-event offset exceeds the 20% concentration limit")
    if canonical:
        maximum_near_duplicate_rate = realism_catalog["metricDefinitions"]["maximumPairwiseNearDuplicateRate"]["value"]
        for grain in (
            "sourceLocalEpisodes", "clinicalEvents",
            "intentionalProblemVariants", "crossSourceScenarioVariants",
        ):
            row = audit["variabilityGrains"][grain]
            if row["pairwiseNearDuplicateRate"] > maximum_near_duplicate_rate:
                errors.append(
                    f"{grain} pairwise near-duplicate rate exceeds {maximum_near_duplicate_rate:.2%}"
                )
        if structural["pairwiseNearDuplicateRate"] > maximum_near_duplicate_rate:
            errors.append(
                f"patient structural pairwise near-duplicate rate exceeds {maximum_near_duplicate_rate:.2%}"
            )
    distributions = audit["clinicalDistributions"]
    guards = load_catalog(spec["clinicalDistributions"]["catalog"])["guardrails"]
    for loinc, row in distributions["analytes"].items():
        guard = guards.get(loinc, {})
        if row["ordinaryCount"] and guard.get("unit") and row["ordinaryUnitCodes"] != [guard["unit"]]:
            errors.append(f"{loinc} has incompatible ordinary units")
        if row["ordinaryCount"] and guard.get("allowedStatuses") and any(
            status not in guard["allowedStatuses"] for status in row["ordinaryStatuses"]
        ):
            errors.append(f"{loinc} has an unreviewed ordinary status")
        if row["ordinaryCount"] and guard.get("requiresEventTime") and row["ordinaryEventTimeRate"] != 1.0:
            errors.append(f"{loinc} lacks ordinary event timing")
        if row.get("applicableOrdinaryCount", row["ordinaryCount"]) >= 100 and not (
            guard.get("minimumOrdinaryAbnormalRate", 0) <= row["ordinaryAbnormalRate"] <= guard.get("maximumOrdinaryAbnormalRate", 1)
        ):
            errors.append(f"{loinc} ordinary abnormal rate is outside its frozen band")
        if row.get("applicableOrdinaryCount", row["ordinaryCount"]) >= 500 and row["ordinaryAbnormalCount"]:
            if row["distinctOrdinaryAbnormalValues"] < 5:
                errors.append(f"{loinc} has fewer than five distinct ordinary abnormal values")
            if row["topOrdinaryAbnormalPointMassShare"] > .25:
                errors.append(f"{loinc} abnormal value point mass exceeds 25%")
        if row.get("applicableOrdinaryCount", row["ordinaryCount"]) and row["completeMetadataRate"] != 1.0:
            errors.append(f"{loinc} has incomplete or inconsistent quantitative metadata")
        if row["physiologicBoundViolationCount"]:
            errors.append(f"{loinc} has unexplained physiologic-bound violations")
        if row.get("ageExcludedReferenceMetadataViolationCount", 0):
            errors.append(f"{loinc} has age-inapplicable reference metadata")
        if row.get("ageExcludedDerivedSafetyViolationCount", 0):
            errors.append(f"{loinc} has an invalid age-excluded derived value")
        if row.get("unknownAgeCount", 0):
            errors.append(f"{loinc} has unresolved age applicability")
    if canonical:
        background_model = realism_catalog["backgroundClinicalModel"]
        correlation_targets = {row["id"]: row for row in background_model["subgroupCorrelations"]}
        for name, row in distributions["subgroupCorrelations"].items():
            target = correlation_targets[name]
            if row["selectedCount"] >= target["minimumSelectedCount"] \
                    and row["comparatorCount"] >= target["minimumComparatorCount"] \
                    and row["effectDifference"] < target["minimumEffectDifference"]:
                errors.append(f"declared subgroup correlation is not directional enough: {name}")
        joint_targets = {row["id"]: row for row in background_model["jointFamilies"]}
        for name, row in distributions["jointFamilies"].items():
            target = joint_targets[name]
            if row["pairedPatientCount"] < target["minimumPairedPatients"]:
                errors.append(f"joint family has insufficient paired patients: {name}")
            elif row["pearsonCorrelation"] is None or row["pearsonCorrelation"] < target["minimumPearsonCorrelation"]:
                errors.append(f"joint family correlation is below its frozen target: {name}")
        missing_trajectory_classes = set(background_model["trajectoryClasses"]) - set(
            distributions["trajectoryClasses"]["overallCounts"]
        )
        if missing_trajectory_classes:
            errors.append(f"declared trajectory classes are absent: {sorted(missing_trajectory_classes)}")
        if distributions["intensiveRepeatedAnalyteRate"] < background_model["minimumRepeatedAnalyteRateIntensive"]:
            errors.append("fewer than 20% of intensive-pathway patients have repeated same-analyte measurements")
    return errors
