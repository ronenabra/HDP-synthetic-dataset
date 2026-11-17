from __future__ import annotations

from datetime import date, timedelta
from itertools import combinations
from typing import Any, Iterable

from .temporal import (
    ClinicalInterval, active_during, occurrence_from_resource, parse_fhir_time,
    starts_before, strictly_before, within_window,
)
from .catalogs import load_catalog


SOURCE_ORDER = ("hmo", "hospital-a", "hospital-b")
TIMING_POLICY = load_catalog("tools/synthetic_population/data/scenario-timing.v4.0.0.json")["scenarios"]


def _days(policy: str, window: str, bound: str) -> timedelta:
    return timedelta(days=TIMING_POLICY[policy][window][bound])


def _codings(value: Any) -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        if isinstance(value.get("system"), str) and isinstance(value.get("code"), str):
            yield value["system"], value["code"]
        for item in value.values():
            yield from _codings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _codings(item)


def _has_code(resource: dict, system: str, code: str) -> bool:
    return (system, code) in set(_codings(resource))


def _variant_evidence(resources: list[dict]) -> list[dict]:
    return resources


def renal_trend_results(resources: list[dict]) -> list[dict]:
    medications = _active_medications(resources, "9997")
    return [row for row in _observations(resources, "6298-4")
            if (_observation_value(row, "mmol/L") or -999) > 5.1
            and any(_exposure_precedes(medication, row, "effectiveDateTime") for medication in medications)]


def _date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value[:10])


def _active_condition(resource: dict, snomed: str) -> bool:
    return resource.get("resourceType") == "Condition" and _has_code(resource.get("code"), "http://snomed.info/sct", snomed) and _has_code(resource.get("clinicalStatus"), "http://terminology.hl7.org/CodeSystem/condition-clinical", "active")


def _active_medications(resources: list[dict], rxnorm: str) -> list[dict]:
    return [
        resource for resource in resources
        if resource.get("resourceType") == "MedicationRequest"
        and resource.get("status") == "active"
        and _has_code(
            resource.get("medicationCodeableConcept"),
            "http://www.nlm.nih.gov/research/umls/rxnorm", rxnorm,
        )
        and _complete_medication_exposure(resource)
    ]


def _observations(resources: list[dict], loinc: str) -> list[dict]:
    return [resource for resource in resources if resource.get("resourceType") == "Observation" and resource.get("status") == "final" and _has_code(resource.get("code"), "http://loinc.org", loinc)]


def _complete_medication_exposure(resource: dict) -> bool:
    dosage = resource.get("dosageInstruction", [])
    if not dosage or not resource.get("authoredOn"):
        return False
    instruction = dosage[0]
    repeat = instruction.get("timing", {}).get("repeat", {})
    dose = instruction.get("doseAndRate", [{}])[0].get("doseQuantity", {})
    course = resource.get("courseOfTherapyType", {}).get("coding", [])
    return bool(
        repeat.get("frequency") and repeat.get("period") and repeat.get("periodUnit")
        and instruction.get("route", {}).get("coding")
        and dose.get("value") is not None and dose.get("code")
        and course
    )


def _observation_value(resource: dict, unit: str) -> float | None:
    quantity = resource.get("valueQuantity", {})
    value = quantity.get("value")
    if not isinstance(value, (int, float)) or quantity.get("code") != unit:
        return None
    interpretation = {
        code for system, code in _codings(resource.get("interpretation"))
        if system == "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation"
    }
    if not interpretation:
        return None
    return float(value)


def _exposure_precedes(resource: dict, event: dict, event_field: str, *, interval_safe: bool = True) -> bool:
    if not interval_safe:
        start = _date(resource.get("authoredOn"))
        event_date = _date(event.get(event_field))
        if not start or not event_date or start > event_date:
            return False
        bounds = resource.get("dosageInstruction", [{}])[0].get("timing", {}).get("repeat", {}).get("boundsPeriod", {})
        end = _date(bounds.get("end"))
        return end is None or event_date <= end
    exposure = occurrence_from_resource(resource)
    occurred = occurrence_from_resource(event)
    if exposure is None or occurred is None or not starts_before(exposure.interval, occurred.interval):
        return False
    bounds = resource.get("dosageInstruction", [{}])[0].get("timing", {}).get("repeat", {}).get("boundsPeriod", {})
    end = parse_fhir_time(bounds["end"]) if bounds.get("end") else None
    return (end is None or occurred.interval.start <= end.end) and within_window(
        exposure.interval, occurred.interval,
        maximum=_days("renal-medication-risk", "exposureToMonitoringDays", "maximum") + timedelta(days=1),
    )


def _medication_exposure(resource: dict) -> ClinicalInterval | None:
    occurrence = occurrence_from_resource(resource)
    if occurrence is None:
        return None
    bounds = resource.get("dosageInstruction", [{}])[0].get("timing", {}).get("repeat", {}).get("boundsPeriod", {})
    start = parse_fhir_time(bounds["start"]) if bounds.get("start") else occurrence.interval
    if bounds.get("end"):
        end = parse_fhir_time(bounds["end"])
    elif resource.get("status") == "active":
        end = parse_fhir_time("9999-12-31")
    else:
        end = occurrence.interval
    return ClinicalInterval(start.start, end.end, "period")


def evaluate(name: str, resources: list[dict], *, require_three_source_context: bool = False) -> bool:
    # Backward-compatible token fixtures remain useful, while generated v2 graphs
    # always take the status/temporal path below.
    if any(resource.get("resourceType") is None for resource in resources):
        codes = set(_codings(resources))
        if name == "renal-medication-risk":
            return (("http://snomed.info/sct", "431857002") in codes and any(float(resource.get("valueQuantity", {}).get("value", -999)) > 5.1 for resource in resources if resource.get("resourceType") == "Observation") and ("http://www.nlm.nih.gov/research/umls/rxnorm", "9997") in codes)
        if name == "bleeding-risk":
            return (("http://www.nlm.nih.gov/research/umls/rxnorm", "11289") in codes and any(float(resource.get("valueQuantity", {}).get("value", 999)) < 12 for resource in resources if resource.get("resourceType") == "Observation") and ("http://snomed.info/sct", "13200003") in codes)
        if name == "allergy-treatment-risk":
            return {("http://snomed.info/sct", "91936005"), ("http://loinc.org", "630-4"), ("http://www.nlm.nih.gov/research/umls/rxnorm", "723")}.issubset(codes)
    if name == "renal-medication-risk":
        renal = any(_active_condition(resource, "431857002") for resource in resources)
        medications = _active_medications(resources, "9997")
        high_results = [
            resource for resource in _observations(resources, "6298-4")
            if (_observation_value(resource, "mmol/L") or -999) > 5.1
        ]
        ordered_timing = any(
            _exposure_precedes(med, obs, "effectiveDateTime", interval_safe=require_three_source_context)
            for med in medications for obs in high_results
        )
        return renal and bool(medications) and bool(high_results) and ordered_timing
    if name == "bleeding-risk":
        medications = _active_medications(resources, "11289")
        bleeds = [resource for resource in resources if _active_condition(resource, "74474003") or _active_condition(resource, "13200003")]
        low_results = [
            resource for resource in _observations(resources, "718-7")
            if (_observation_value(resource, "g/dL") or 999) < 12.0
        ]
        coherent = any(abs(((_date(obs.get("effectiveDateTime")) or date.min) - (_date(bleed.get("recordedDate")) or date.max)).days) <= 14 for bleed in bleeds for obs in low_results)
        exposed = any(
            _exposure_precedes(med, obs, "effectiveDateTime", interval_safe=require_three_source_context)
            and (
                not require_three_source_context
                or within_window(
                    occurrence_from_resource(med).interval,
                    occurrence_from_resource(obs).interval,
                    maximum=_days("bleeding-risk", "exposureToBleedingDays", "maximum") + timedelta(days=1),
                )
            )
            for med in medications for obs in low_results
        )
        return bool(medications) and bool(bleeds) and bool(low_results) and coherent and exposed
    if name == "allergy-treatment-risk":
        allergies = [
            resource for resource in resources
            if resource.get("resourceType") == "AllergyIntolerance"
            and resource.get("criticality") == "high"
            and _has_code(resource.get("code"), "http://snomed.info/sct", "91936005")
            and _has_code(resource.get("clinicalStatus"), "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical", "active")
            and _has_code(resource.get("verificationStatus"), "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification", "confirmed")
            and any(reaction.get("severity") == "severe" for reaction in resource.get("reaction", []))
        ]
        medications = _active_medications(resources, "723")
        cultures = _observations(resources, "630-4")
        coherent = any(
            (
                allergy_occurrence is not None and medication_occurrence is not None
                and starts_before(allergy_occurrence.interval, medication_occurrence.interval)
                and within_window(
                    allergy_occurrence.interval, medication_occurrence.interval,
                    maximum=_days("allergy-treatment-risk", "allergyBeforeTreatmentDays", "maximum") + timedelta(days=1),
                )
            ) if require_three_source_context else (
                (_date(allergy.get("onsetDateTime")) or date.max) <= (_date(med.get("authoredOn")) or date.min)
            )
            for allergy in allergies for med in medications
            for allergy_occurrence in [occurrence_from_resource(allergy)]
            for medication_occurrence in [occurrence_from_resource(med)]
        )
        return bool(allergies) and bool(medications) and coherent and (
            bool(cultures) if require_three_source_context else True
        )
    raise ValueError(f"unknown cross-source predicate: {name}")


def _dose(resource: dict) -> float | None:
    from .medication_safety import normalized_dose_mg
    return normalized_dose_mg(resource)


def evaluate_variant(
    name: str, variant: str, resources: list[dict], *, require_three_source_context: bool = False,
) -> bool:
    """Evaluate the family truth plus the observable facts unique to a variant."""
    if not evaluate(name, resources, require_three_source_context=require_three_source_context):
        return False
    evidence = _variant_evidence(resources)
    if variant == "chronic-renal-trend":
        values = sorted(
            ((_date(row.get("effectiveDateTime")) or date.min, float(row["valueQuantity"]["value"])) for row in renal_trend_results(evidence)),
            key=lambda row: row[0],
        )
        return len(values) >= 2 and values[-1][1] > values[0][1] and 0 < (values[-1][0] - values[0][0]).days <= 90
    if variant == "acute-renal-precipitant":
        acute = [row for row in evidence if _active_condition(row, "14669001")]
        high = [row for row in _observations(evidence, "6298-4") if (_observation_value(row, "mmol/L") or -999) > 5.1]
        return any(
            left is not None and right is not None
            and starts_before(left.interval, right.interval)
            and within_window(left.interval, right.interval, maximum=_days("renal-medication-risk", "precipitantToChangeDays", "maximum"))
            for condition in acute for observation in high
            for left in [occurrence_from_resource(condition)]
            for right in [occurrence_from_resource(observation)]
        )
    if variant == "medication-dose-escalation":
        rows = [row for row in evidence if row.get("resourceType") == "MedicationRequest" and _has_code(row.get("medicationCodeableConcept"), "http://www.nlm.nih.gov/research/umls/rxnorm", "9997")]
        if not require_three_source_context:
            doses = [dose for row in rows if (dose := _dose(row)) is not None]
            return len(rows) >= 2 and max(doses, default=0) > min(doses, default=0)
        ordered = sorted(
            ((occurrence.interval, dose) for row in rows
             if (occurrence := occurrence_from_resource(row)) is not None
             and (dose := _dose(row)) is not None),
            key=lambda item: item[0].start,
        )
        return any(strictly_before(left[0], right[0]) and right[1] > left[1] for left, right in zip(ordered, ordered[1:]))
    if variant == "overt-bleeding":
        return any(_active_condition(row, "74474003") for row in resources)
    if variant == "falling-hemoglobin":
        values = sorted(
            ((_date(row.get("effectiveDateTime")) or date.min, float(row["valueQuantity"]["value"])) for row in _observations(evidence, "718-7")),
            key=lambda row: row[0],
        )
        return any(
            later[1] < earlier[1]
            and TIMING_POLICY["bleeding-risk"]["hemoglobinTrendDays"]["minimum"]
            <= (later[0] - earlier[0]).days
            <= TIMING_POLICY["bleeding-risk"]["hemoglobinTrendDays"]["maximum"]
            for position, earlier in enumerate(values)
            for later in values[position + 1:]
        )
    if variant == "interacting-exposure":
        anticoagulants = _active_medications(evidence, "11289")
        interacting = _active_medications(evidence, "1191")
        return any(
            left is not None and right is not None and active_during(left, right)
            for one in anticoagulants for two in interacting
            for left in [_medication_exposure(one)] for right in [_medication_exposure(two)]
        )
    if variant == "prescribing-after-allergy":
        return bool(_active_medications(resources, "723"))
    if variant == "administration-after-allergy":
        if not require_three_source_context:
            return any(row.get("resourceType") == "MedicationStatement" and row.get("status") == "active" and _has_code(row.get("medicationCodeableConcept"), "http://www.nlm.nih.gov/research/umls/rxnorm", "723") and row.get("effectivePeriod", {}).get("start") for row in evidence)
        administrations = [row for row in evidence if row.get("resourceType") == "MedicationAdministration" and row.get("status") == "completed" and _has_code(row.get("medicationCodeableConcept"), "http://www.nlm.nih.gov/research/umls/rxnorm", "723")]
        allergies = [row for row in evidence if row.get("resourceType") == "AllergyIntolerance" and _has_code(row.get("code"), "http://snomed.info/sct", "91936005") and row.get("criticality") == "high"]
        return any(left is not None and right is not None and strictly_before(left.interval, right.interval)
                   and within_window(left.interval, right.interval, maximum=_days("allergy-treatment-risk", "allergyBeforeTreatmentDays", "maximum") + timedelta(days=1))
                   for allergy in allergies for administration in administrations
                   for left in [occurrence_from_resource(allergy)] for right in [occurrence_from_resource(administration)])
    if variant == "culture-informed-treatment":
        cultures = _observations(evidence, "630-4")
        specimens = [
            row for row in evidence
            if row.get("resourceType") == "Specimen" and row.get("status") == "available"
        ]
        orders = [
            row for row in evidence
            if row.get("resourceType") == "ServiceRequest"
            and row.get("status") == "completed"
            and _has_code(row.get("code"), "http://loinc.org", "630-4")
        ]
        treatments = _active_medications(evidence, "723")
        return any(
            ordered is not None and collected is not None and resulted is not None and treated is not None
            and strictly_before(ordered.interval, collected.interval)
            and starts_before(collected.interval, resulted.interval)
            and strictly_before(resulted.interval, treated.interval)
            and within_window(collected.interval, resulted.interval, maximum=_days("culture-informed-treatment", "collectionToResultDays", "maximum"))
            and within_window(resulted.interval, treated.interval, maximum=_days("culture-informed-treatment", "resultToTreatmentDays", "maximum"))
            for order in orders for specimen in specimens for culture in cultures for treatment in treatments
            for ordered in [occurrence_from_resource(order)]
            for collected in [occurrence_from_resource(specimen)]
            for resulted in [occurrence_from_resource(culture)]
            for treated in [occurrence_from_resource(treatment)]
        )
    raise ValueError(f"unknown cross-source variant: {name}/{variant}")


def subset_truth_table(name: str, resources_by_source: dict[str, list[dict]]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for size in (1, 2, 3):
        for subset in combinations(SOURCE_ORDER, size):
            key = "+".join(subset)
            result[key] = evaluate(name, [resource for source in subset for resource in resources_by_source[source]])
    return result


def subset_truth_table_variant(
    name: str, variant: str, resources_by_source: dict[str, list[dict]], *,
    require_three_source_context: bool = False,
) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for size in (1, 2, 3):
        for subset in combinations(SOURCE_ORDER, size):
            key = "+".join(subset)
            result[key] = evaluate_variant(
                name, variant, [resource for source in subset for resource in resources_by_source[source]],
                require_three_source_context=require_three_source_context,
            )
    return result
