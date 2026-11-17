from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, datetime
from typing import Any, Callable, Iterable

from .catalogs import load_catalog
from .observable import is_supplemental_capture


EVENT_FIELDS = {
    "authoredOn", "recordedDate", "effectiveDateTime", "issued", "occurrenceDateTime",
    "performedDateTime", "onsetDateTime", "dateAsserted", "receivedTime", "start", "end",
}
ACTUAL_EVENT_FIELDS = EVENT_FIELDS - {"onsetDateTime", "dateAsserted"}


def _walk(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _codings(value: Any) -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        if isinstance(value.get("system"), str) and isinstance(value.get("code"), str):
            yield value["system"], value["code"]
        for child in value.values():
            yield from _codings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _codings(child)


def _dates(resource: dict[str, Any]) -> list[date]:
    result = []
    for key, value in _walk(resource):
        if key in EVENT_FIELDS and isinstance(value, str) and len(value) >= 10:
            try:
                result.append(date.fromisoformat(value[:10]))
            except ValueError:
                continue
    return result


def _actual_event_dates(resource: dict[str, Any]) -> list[date]:
    result = []
    for key, value in _walk(resource):
        if key in ACTUAL_EVENT_FIELDS and isinstance(value, str) and len(value) >= 10:
            try:
                result.append(date.fromisoformat(value[:10]))
            except ValueError:
                continue
    return result


def _has_code(resources: list[dict[str, Any]], code: str, resource_type: str | None = None) -> bool:
    return any(
        (resource_type is None or resource["resourceType"] == resource_type)
        and any(found == code for _system, found in _codings(resource))
        for resource in resources
    )


def _count(resources: list[dict[str, Any]], resource_type: str) -> int:
    return sum(resource["resourceType"] == resource_type for resource in resources)


def _medication_complete(resources: list[dict[str, Any]]) -> bool:
    medications = [resource for resource in resources if resource["resourceType"] == "MedicationRequest"]
    return bool(medications) and all(
        any(
            dosage.get("timing", {}).get("repeat", {}).get("frequency")
            and dosage.get("route", {}).get("coding")
            and dosage.get("doseAndRate", [{}])[0].get("doseQuantity", {}).get("value") is not None
            for dosage in resource.get("dosageInstruction", [])
        )
        for resource in medications
    )


def _observable_secondary_trait(resources: list[dict[str, Any]]) -> str:
    """Infer secondary clinical evidence without crosswalk or generator tags."""
    if _has_code(resources, "237602007", "Condition"):
        return "metabolic-risk"
    if _has_code(resources, "195967001", "Condition"):
        return "respiratory-history"
    hypertension = any(
        row["resourceType"] == "Condition"
        and _has_code([row.get("code", {})], "38341003")
        and _has_code([row.get("clinicalStatus", {})], "active")
        for row in resources
    )
    lisinopril = any(
        row["resourceType"] == "MedicationRequest"
        and row.get("status") == "active"
        and _has_code([row.get("medicationCodeableConcept", {})], "29046")
        and _medication_complete([row])
        for row in resources
    )
    if hypertension and lisinopril:
        return "medication-exposure"
    if any(
        row["resourceType"] == "MedicationStatement"
        and row.get("status") == "completed"
        and _has_code([row.get("medicationCodeableConcept", {})], "723")
        and any(
            dosage.get("timing", {}).get("repeat", {}).get("frequency")
            and dosage.get("route", {}).get("coding")
            and dosage.get("doseAndRate", [{}])[0].get("doseQuantity", {}).get("value") is not None
            for dosage in row.get("dosage", [])
        )
        for row in resources
    ):
        return "medication-exposure"
    return "none"


def _span(resources: list[dict[str, Any]]) -> int:
    values = sorted({event for resource in resources for event in _actual_event_dates(resource)})
    return (values[-1] - values[0]).days if len(values) >= 2 else 0


def _after_code(resources: list[dict[str, Any]], later_type: str, earlier_type: str, earlier_code: str | None = None) -> bool:
    earlier = [resource for resource in resources if resource["resourceType"] == earlier_type and (not earlier_code or _has_code([resource], earlier_code))]
    later = [resource for resource in resources if resource["resourceType"] == later_type]
    return any(max(_dates(right), default=date.min) > min(_dates(left), default=date.max) for left in earlier for right in later)


Predicate = Callable[[list[dict[str, Any]], dict[str, Any]], bool]


PREDICATES: dict[str, Predicate] = {
    "preventive-or-acute-encounter": lambda rows, patient: _count(rows, "Encounter") >= 1,
    "resolved-minor-condition": lambda rows, patient: any(
        resource["resourceType"] == "Condition"
        and any(code == "resolved" for _system, code in _codings(resource.get("clinicalStatus")))
        and bool(resource.get("abatementDateTime"))
        for resource in rows
    ),
    "no-unexplained-severe-chronic": lambda rows, patient: not any(
        _has_code([resource], code, "Condition") for resource in rows for code in ("431857002", "363346000")
    ),
    "active-chronic-diagnosis": lambda rows, patient: _count(rows, "Condition") >= 1,
    "serial-monitoring": lambda rows, patient: _count(rows, "Observation") >= 2,
    "chronic-medication": lambda rows, patient: _medication_complete(rows),
    "one-year-span": lambda rows, patient: _span(rows) >= 365,
    "guardian": lambda rows, patient: _count(rows, "RelatedPerson") >= 1,
    "growth-or-vital": lambda rows, patient: _count(rows, "Observation") >= 1,
    "multiple-immunizations": lambda rows, patient: _count(rows, "Immunization") >= 2,
    "no-prebirth-event": lambda rows, patient: all(
        event >= date.fromisoformat(patient["birthDate"])
        for resource in rows for event in _dates(resource)
        if resource["resourceType"] not in {"Patient", "RelatedPerson"}
    ),
    "medication-request-and-statement": lambda rows, patient: _count(rows, "MedicationRequest") >= 1 and _count(rows, "MedicationStatement") >= 1,
    "complete-dosage": lambda rows, patient: _medication_complete(rows),
    "medication-lifecycle": lambda rows, patient: any(
        resource["resourceType"] == "MedicationStatement" and resource.get("status") in {"active", "stopped", "completed"}
        and bool(resource.get("derivedFrom")) for resource in rows
    ),
    "risk-aware-assessment": lambda rows, patient: patient.get("gender") == "female" and _count(rows, "Encounter") >= 1,
    "mammography-age-valid": lambda rows, patient: not _has_code(rows, "71651007", "Procedure") or 50 <= patient["age"] <= 74,
    "surgical-referral": lambda rows, patient: _has_code(rows, "80146002", "ServiceRequest"),
    "surgical-procedure": lambda rows, patient: _has_code(rows, "80146002", "Procedure"),
    "postoperative-follow-up": lambda rows, patient: _has_code(rows, patient.get("_postoperativeFollowupCode", "185389009"), "Encounter") and _after_code(rows, "Encounter", "Procedure", "80146002"),
    "oncology-diagnosis": lambda rows, patient: _has_code(rows, "363346000", "Condition"),
    "oncology-confirmation": lambda rows, patient: _has_code(rows, "86273004", "Procedure"),
    "oncology-treatment": lambda rows, patient: _has_code(rows, "367336001", "Procedure"),
    "oncology-monitoring": lambda rows, patient: _count(rows, "Observation") >= 2,
    "oncology-surveillance": lambda rows, patient: _has_code(rows, "363346000", "Encounter") and _after_code(rows, "Encounter", "Procedure", "367336001"),
    "named-complex-path": lambda rows, patient: any(_has_code(rows, code, "Condition") for code in ("55464009", "193093009", "709044004", "75934005")),
    "multi-specialty": lambda rows, patient: len({resource["_source"] for resource in rows if resource["resourceType"] == "Encounter"}) >= 2,
    "complex-follow-up": lambda rows, patient: _count(rows, "Observation") >= 2 and _medication_complete(rows) and _span(rows) >= 365,
}


GRAMMARS: dict[str, dict[str, list[str]]] = {
    "generally-healthy": {
        "required": ["preventive-or-acute-encounter", "resolved-minor-condition"],
        "forbidden": ["no-unexplained-severe-chronic"],
    },
    "chronic-care": {
        "required": ["active-chronic-diagnosis", "serial-monitoring", "chronic-medication"],
        "ordered": ["one-year-span"],
    },
    "pediatric": {
        "required": ["guardian", "growth-or-vital", "multiple-immunizations"],
        "forbidden": ["no-prebirth-event"],
    },
    "medication-intensive": {
        "required": ["medication-request-and-statement", "complete-dosage", "serial-monitoring"],
        "lifecycle": ["medication-lifecycle", "one-year-span"],
    },
    "womens-health": {
        "required": ["risk-aware-assessment"],
        "forbidden": ["mammography-age-valid"],
    },
    "perioperative-surgical": {
        "required": ["surgical-referral", "surgical-procedure", "postoperative-follow-up"],
        "ordered": ["postoperative-follow-up"],
    },
    "oncology": {
        "required": ["oncology-diagnosis", "oncology-confirmation", "oncology-treatment", "oncology-monitoring", "oncology-surveillance"],
        "ordered": ["oncology-surveillance", "one-year-span"],
    },
    "rare-complex": {
        "required": ["named-complex-path", "multi-specialty", "complex-follow-up"],
        "ordered": ["one-year-span"],
    },
}

# These historical v2 predicates encode one concrete pathway (for example an
# appendectomy or biopsy) as though it were the whole archetype. The v4 typed
# clauses validate the selected, observable branch with code, status, temporal,
# and source checks. Keep every legacy evaluation visible and require the named
# typed successor to pass when the historical predicate does not apply.
LEGACY_TYPED_SUCCESSORS = {
    "resolved-minor-condition", "serial-monitoring", "chronic-medication",
    "multiple-immunizations",
    "medication-request-and-statement", "medication-lifecycle",
    "surgical-referral", "surgical-procedure", "postoperative-follow-up",
    "oncology-diagnosis", "oncology-confirmation", "oncology-treatment",
    "oncology-monitoring", "oncology-surveillance", "complex-follow-up",
}


_PREDICATE_RESOURCE_TYPES = {
    "preventive-or-acute-encounter": ["Encounter"],
    "resolved-minor-condition": ["Condition"],
    "no-unexplained-severe-chronic": ["Condition"],
    "active-chronic-diagnosis": ["Condition"],
    "serial-monitoring": ["Observation"],
    "chronic-medication": ["MedicationRequest"],
    "one-year-span": ["clinical-event"],
    "guardian": ["RelatedPerson"],
    "growth-or-vital": ["Observation"],
    "multiple-immunizations": ["Immunization"],
    "no-prebirth-event": ["clinical-event"],
    "medication-request-and-statement": ["MedicationRequest", "MedicationStatement"],
    "complete-dosage": ["MedicationRequest"],
    "medication-lifecycle": ["MedicationRequest", "MedicationStatement"],
    "risk-aware-assessment": ["Patient", "Encounter"],
    "mammography-age-valid": ["Patient", "Procedure"],
    "surgical-referral": ["ServiceRequest"],
    "surgical-procedure": ["Procedure"],
    "postoperative-follow-up": ["Procedure", "Encounter"],
    "oncology-diagnosis": ["Condition"],
    "oncology-confirmation": ["Procedure"],
    "oncology-treatment": ["Procedure"],
    "oncology-monitoring": ["Observation"],
    "oncology-surveillance": ["Procedure", "Encounter"],
    "named-complex-path": ["Condition"],
    "multi-specialty": ["Encounter"],
    "complex-follow-up": ["Observation", "MedicationRequest", "clinical-event"],
}

_PREDICATE_CODE_FAMILIES = {
    "resolved-minor-condition": ["SNOMED-clinical-finding", "condition-clinical=resolved"],
    "no-unexplained-severe-chronic": ["SNOMED severe-chronic exclusion set"],
    "chronic-medication": ["RxNorm ingredient/product", "SNOMED route"],
    "multiple-immunizations": ["IL-Core vaccine binding"],
    "medication-request-and-statement": ["RxNorm ingredient/product"],
    "complete-dosage": ["UCUM dose", "SNOMED route"],
    "medication-lifecycle": ["FHIR medication status transition"],
    "mammography-age-valid": ["SNOMED 71651007 mammography"],
    "surgical-referral": ["SNOMED 80146002 appendectomy"],
    "surgical-procedure": ["SNOMED 80146002 appendectomy"],
    "postoperative-follow-up": ["SNOMED 185389009 follow-up"],
    "oncology-diagnosis": ["SNOMED malignant-disease family"],
    "oncology-confirmation": ["SNOMED 86273004 biopsy"],
    "oncology-treatment": ["SNOMED 367336001 chemotherapy"],
    "oncology-surveillance": ["SNOMED malignant-disease follow-up"],
    "named-complex-path": ["reviewed named rare/complex SNOMED set"],
}

_PREDICATE_TEMPORAL_RELATIONS = {
    "one-year-span": "latest actual event occurs at least 365 days after earliest actual event",
    "no-prebirth-event": "every actual event is on/after Patient.birthDate",
    "mammography-age-valid": "mammography absent or Patient age is 50..74 at the evaluation anchor",
    "postoperative-follow-up": "follow-up Encounter occurs after the surgical Procedure",
    "oncology-surveillance": "surveillance Encounter occurs after completed treatment",
    "complex-follow-up": "serial monitoring and complete medication evidence span at least 365 days",
}


def _clause_contracts() -> dict[str, dict[str, Any]]:
    categories: dict[str, set[str]] = defaultdict(set)
    archetypes: dict[str, set[str]] = defaultdict(set)
    for archetype, grammar in GRAMMARS.items():
        for category, names in grammar.items():
            for name in names:
                categories[name].add(category)
                archetypes[name].add(archetype)
    return {
        name: {
            "predicate": name,
            "clauseTypes": sorted(categories[name]),
            "archetypes": sorted(archetypes[name]),
            "eventResourceTypes": _PREDICATE_RESOURCE_TYPES[name],
            "allowedProfiles": [
                f"declared pinned profile for {resource_type}"
                for resource_type in _PREDICATE_RESOURCE_TYPES[name]
            ],
            "allowedCodeFamilies": _PREDICATE_CODE_FAMILIES.get(
                name, ["catalogued semantic family appropriate to the event role"],
            ),
            "minimumCardinality": 0 if "forbidden" in categories[name] else 1,
            "maximumCardinality": None,
            "sourceOwnership": "any source that plausibly captures the declared event role",
            "temporalRelation": _PREDICATE_TEMPORAL_RELATIONS.get(
                name, "presence, lifecycle, status, or code-family relation",
            ),
            "lifecycleTransition": (
                "explicit status/dosage transition required"
                if name == "medication-lifecycle" else "not-applicable"
            ),
            "branchProbability": {
                "kind": "deterministic-balanced-pathway-allocation",
                "minimumCanonicalCases": 3 if "rare-complex" in archetypes[name] else 5,
                "maximumArchetypeShare": .35,
            },
            "forbiddenCombinations": (
                ["adult screening without age/risk support"] if name == "mammography-age-valid"
                else ["event before birth"] if name == "no-prebirth-event"
                else ["active unexplained severe chronic condition"] if name == "no-unexplained-severe-chronic"
                else []
            ),
            "evaluationInput": "tag-stripped observable FHIR graph",
        }
        for name in sorted(PREDICATES)
    }


CLAUSE_CONTRACTS = _clause_contracts()


def _clinical_signature(resources: list[dict[str, Any]]) -> str:
    rows = []
    for resource in resources:
        if resource["resourceType"] in {"Patient", "RelatedPerson"}:
            continue
        rows.append((
            resource["resourceType"],
            tuple(sorted(set(_codings({key: resource.get(key) for key in ("code", "reasonCode", "medicationCodeableConcept", "vaccineCode", "category")})))),
            str(resource.get("status", "")),
            tuple(sorted(reference.get("reference", "").split(":")[-1].split("/")[0] for _key, reference in _walk(resource) if isinstance(reference, dict) and "reference" in reference)),
        ))
    raw = json.dumps(sorted(rows), sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(raw.encode()).hexdigest()


def build_grammar_audit(
    bundles: dict[str, dict[str, Any]], crosswalk: dict[str, Any], spec: dict[str, Any],
) -> dict[str, Any]:
    cohort_system = spec["identity"]["cohortIdentifierSystem"]
    url_to_cohort: dict[tuple[str, str], str] = {}
    patients: dict[str, dict[str, Any]] = {}
    for source, bundle in bundles.items():
        for entry in bundle["entry"]:
            resource = entry["resource"]
            if resource["resourceType"] != "Patient":
                continue
            cohort = next((row.get("value") for row in resource.get("identifier", []) if row.get("system") == cohort_system), None)
            if cohort:
                url_to_cohort[(source, entry["fullUrl"])] = cohort
                patients.setdefault(cohort, resource)
    by_cohort: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source, bundle in bundles.items():
        for entry in bundle["entry"]:
            resource = entry["resource"]
            cohort = url_to_cohort.get((source, entry["fullUrl"]))
            if not cohort:
                references = [value.get("reference") for _key, value in _walk(resource) if isinstance(value, dict) and value.get("reference")]
                cohort = next((url_to_cohort.get((source, reference)) for reference in references if url_to_cohort.get((source, reference))), None)
            if not cohort:
                cohort = next((
                    value.get("value") for _key, value in _walk(resource)
                    if isinstance(value, dict) and value.get("system") == cohort_system and value.get("value") in patients
                ), None)
            if cohort:
                by_cohort[cohort].append({
                    **resource, "_source": source, "_observableSource": source,
                    "_fullUrl": entry["fullUrl"],
                })
    archetype_by_cohort = {row["cohortId"]: row["primaryArchetype"] for row in crosswalk["patients"]}
    variant_by_cohort = {row["cohortId"]: row.get("pathwayVariant") for row in crosswalk["patients"]}
    secondary_by_cohort = {row["cohortId"]: row.get("secondaryTrait") for row in crosswalk["patients"]}
    failures = []
    clause_counts: Counter[str] = Counter()
    signatures: dict[str, Counter[str]] = defaultdict(Counter)
    spans: dict[str, list[int]] = defaultdict(list)
    summaries: dict[str, Counter[str]] = defaultdict(Counter)
    declared_variants: dict[str, Counter[str]] = defaultdict(Counter)
    declared_variant_signatures: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    secondary_combinations: dict[str, Counter[str]] = defaultdict(Counter)
    reconciliations: Counter[str] = Counter()
    typed_clauses = None
    if spec.get("remediation"):
        from .typed_grammar import compile_grammar
        typed_clauses = compile_grammar(spec["remediation"]["pathwayTemplateCatalog"])
    realism_catalog = load_catalog(spec["clinicalRealism"]["catalog"]) if spec.get("clinicalRealism") else {}
    immunization_decision = realism_catalog.get("pathwayReleaseDecisions", {}).get(
        "pediatric-immunization", {}
    )
    blocked_predicates = {
        "multiple-immunizations"
    } if immunization_decision.get("generationDefault") == "excluded" else set()
    blocked_clauses: Counter[str] = Counter()
    anchor = date.fromisoformat(spec["timelineAnchor"][:10])
    for cohort, archetype in archetype_by_cohort.items():
        resources = by_cohort[cohort]
        patient = patients[cohort]
        born = date.fromisoformat(patient["birthDate"])
        context = {
            **patient,
            "age": anchor.year - born.year - ((anchor.month, anchor.day) < (born.month, born.day)),
            "_postoperativeFollowupCode": "185389009" if spec.get("remediation") else "183755002",
        }
        signature = _clinical_signature(resources)
        signatures[archetype][signature] += 1
        variant = variant_by_cohort.get(cohort) or "undeclared"
        declared_variants[archetype][variant] += 1
        declared_variant_signatures[archetype][variant].add(signature)
        declared_secondary = secondary_by_cohort.get(cohort) or "none"
        observable_secondary = _observable_secondary_trait(resources)
        secondary_combinations[archetype][declared_secondary if declared_secondary == observable_secondary else "none"] += 1
        if declared_secondary != "none" and declared_secondary != observable_secondary:
            failures.append({
                "cohortId": cohort, "archetype": archetype, "category": "secondary-trait",
                "clause": f"declared-observable-mismatch:{declared_secondary}:{observable_secondary}",
            })
        spans[archetype].append(_span(resources))
        summaries[archetype].update(resource["resourceType"] for resource in resources if resource["resourceType"] != "Patient")
        successor_passed = None
        for category, names in GRAMMARS[archetype].items():
            for name in names:
                if name in blocked_predicates:
                    clause_counts[f"{archetype}:{category}:{name}:release-blocked"] += 1
                    blocked_clauses[f"{archetype}:{category}:{name}"] += 1
                    continue
                passed = PREDICATES[name](resources, context)
                if not passed and name in LEGACY_TYPED_SUCCESSORS and typed_clauses is not None:
                    if successor_passed is None:
                        from .typed_grammar import evaluate_variant
                        successor_passed = all(
                            row["passed"] for row in evaluate_variant(
                                typed_clauses[archetype][variant],
                                [resource for resource in resources if not is_supplemental_capture(resource)],
                                context,
                            )
                        )
                    if successor_passed:
                        reconciliations[f"{archetype}/{variant}/{name} -> typed:{archetype}.{variant}"] += 1
                        passed = True
                clause_counts[f"{archetype}:{category}:{name}:pass" if passed else f"{archetype}:{category}:{name}:fail"] += 1
                if not passed:
                    failures.append({"cohortId": cohort, "archetype": archetype, "category": category, "clause": name})
    archetypes = {}
    for archetype, counts in sorted(signatures.items()):
        patient_count = sum(counts.values())
        archetypes[archetype] = {
            "patientCount": patient_count,
            "materialVariantCount": len(counts),
            "topVariantShare": round(max(counts.values(), default=0) / max(1, patient_count), 6),
            "eventSpanDays": {"minimum": min(spans[archetype]), "maximum": max(spans[archetype])},
            "resourceTypeCounts": dict(sorted(summaries[archetype].items())),
            "declaredVariantCounts": dict(sorted(declared_variants[archetype].items())),
            "declaredVariantStructuralSignatureCounts": {
                variant: len(values)
                for variant, values in sorted(declared_variant_signatures[archetype].items())
            },
            "secondaryTraitCounts": dict(sorted(secondary_combinations[archetype].items())),
        }
    if spec.get("clinicalRealism") and spec["populationSize"] >= 1000:
        catalog = load_catalog(spec["clinicalRealism"]["catalog"])
        for archetype, expected_variants in catalog["pathwayVariants"].items():
            counts = declared_variants[archetype]
            if set(counts) != set(expected_variants):
                failures.append({
                    "cohortId": None, "archetype": archetype, "category": "variant",
                    "clause": "declared-variant-coverage",
                })
            minimum = 3 if archetype == "rare-complex" else 5
            total = sum(counts.values())
            for variant in expected_variants:
                if counts[variant] < minimum or counts[variant] / max(1, total) > .35:
                    failures.append({
                        "cohortId": None, "archetype": archetype, "category": "variant",
                        "clause": f"variant-bounds:{variant}",
                    })
                if not declared_variant_signatures[archetype][variant]:
                    failures.append({
                        "cohortId": None, "archetype": archetype, "category": "variant",
                        "clause": f"variant-has-no-structural-signature:{variant}",
                    })
            if archetype in {"chronic-care", "medication-intensive", "oncology", "rare-complex"} and len(secondary_combinations[archetype]) < 3:
                failures.append({
                    "cohortId": None, "archetype": archetype, "category": "secondary-trait",
                    "clause": "fewer-than-three-secondary-trait-combinations",
                })
            if archetype in {"chronic-care", "medication-intensive", "oncology", "rare-complex"}:
                non_none = total - secondary_combinations[archetype]["none"]
                if non_none / max(1, total) < .20:
                    failures.append({
                        "cohortId": None, "archetype": archetype, "category": "secondary-trait",
                        "clause": "secondary-trait-richness-below-20-percent",
                    })
    return {
        "schemaVersion": "1.0.0",
        "tagStrippedEvaluation": True,
        "registeredPredicates": sorted(PREDICATES),
        "clauseContracts": CLAUSE_CONTRACTS,
        "grammars": GRAMMARS,
        "clauseCounts": dict(sorted(clause_counts.items())),
        "typedSuccessorReconciliations": dict(sorted(reconciliations.items())),
        "failureCount": len(failures),
        "failures": failures,
        "blockedClauses": dict(sorted(blocked_clauses.items())),
        "releaseEligible": not blocked_clauses,
        "releaseBlockers": ([immunization_decision] if blocked_clauses else []),
        "archetypes": archetypes,
        "status": "pass" if not failures else "fail",
    }
