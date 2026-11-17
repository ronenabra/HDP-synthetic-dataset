from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import copy
from datetime import date, datetime, timedelta, timezone
from typing import Any

from .catalogs import load_catalog
from .care_pathway import INJURY_BRANCH_CONCEPTS
from .condition_coding import semantic_condition_codings
from .clinical_branch_predicates import branch_proof, material_branch_proof, branch_coverage_audit
from .observable import codings, is_supplemental_capture, patient_graphs, references, strip_generator_metadata
from .temporal import (
    active_during, contains, ends_after, monotonic_trend, occurrence_from_resource,
    overlaps, starts_before, strictly_after, strictly_before, within_window,
)
from .utils import sha256_json


ROLE_TYPES = {
    "encounter": ("Encounter",), "emergency-encounter": ("Encounter",),
    "inpatient-encounter": ("Encounter",), "follow-up": ("Encounter",),
    "condition": ("Condition",), "assessment": ("Observation", "RiskAssessment", "Condition"),
    "order": ("ServiceRequest",), "referral": ("ServiceRequest",),
    "collection": ("Specimen",), "result": ("Observation", "DiagnosticReport"),
    "growth-measurement": ("Observation",), "prescription": ("MedicationRequest",),
    "dispense": ("MedicationDispense",), "medication-review": ("MedicationStatement",),
    "medication-administration": ("MedicationAdministration",),
    "procedure": ("Procedure",), "oncology-treatment": ("Procedure", "MedicationAdministration"),
    "immunization": ("Immunization",), "discharge": ("Encounter", "DocumentReference"),
}

ROLE_STATUSES = {
    "encounter": ("finished",), "emergency-encounter": ("finished",),
    "inpatient-encounter": ("finished",), "follow-up": ("finished",),
    "condition": ("active", "resolved"), "assessment": ("final",),
    "order": ("completed",), "referral": ("active", "completed"),
    "collection": ("available",), "result": ("final",),
    "growth-measurement": ("final",), "prescription": ("active", "completed", "stopped"),
    "dispense": ("completed",), "medication-review": ("active", "stopped"),
    "medication-administration": ("completed",), "procedure": ("completed",),
    "oncology-treatment": ("completed",), "immunization": ("completed",),
    "discharge": ("current",),
}

SUPPORTED_PREDICATES = {
    "distinct-event-count", "same-analyte-repeats", "trend-direction",
    "active-resolved-status", "medication-exposure", "dose-status-transition",
    "linked-events", "age-sex-risk-eligibility", "temporal-relation",
    "source-participation", "forbidden-combination",
}

MUTATION_CLASSES = (
    "remove-required-event", "compatible-control-code", "incompatible-code",
    "reverse-or-break-window", "lifecycle-status", "broken-reference",
    "insert-forbidden-event", "strip-source-capture", "cosmetic-only",
)


@dataclass(frozen=True)
class Clause:
    id: str
    archetype: str
    variant: str
    category: str
    predicate: str
    clinical_intent: str
    severity: str
    event_role: str
    resource_types: tuple[str, ...]
    code_system: str
    code: str
    family: str
    minimum_distinct_events: int
    maximum_distinct_events: int | None
    allowed_statuses: tuple[str, ...]
    forbidden_statuses: tuple[str, ...]
    relation_to: str | None
    temporal_relation: str | None
    minimum_days: float | None
    maximum_days: float | None
    discriminating: bool
    rationale: str
    source: str
    profiles: tuple[str, ...]
    value_semantics: str
    subject_reference_required: bool
    source_participation_required: bool
    positive_fixture_id: str
    mutation_classes: tuple[str, ...]
    nominal_offset_days: int = 0
    alternate_codes: tuple[str, ...] = ()
    alternate_codings: tuple[tuple[str, str], ...] = ()


def compile_grammar(path: str) -> dict[str, dict[str, list[Clause]]]:
    catalog = load_catalog(path)
    allowed_root = {
        "catalogVersion", "status", "systemDefaults", "historicalEncounterReasons", "archetypes",
    }
    unknown = set(catalog) - allowed_root
    if unknown:
        raise ValueError(f"unused pathway/grammar catalog fields: {sorted(unknown)}")
    history_reasons = catalog.get("historicalEncounterReasons")
    if not isinstance(history_reasons, list) or len(history_reasons) < 5:
        raise ValueError("historicalEncounterReasons must contain at least five reviewed alternatives")
    seen_history_codes: set[tuple[str, str]] = set()
    for reason in history_reasons:
        expected_reason_fields = {"system", "code", "display", "family"}
        if set(reason) != expected_reason_fields or any(
            not isinstance(reason.get(field), str) or not reason[field].strip()
            for field in expected_reason_fields
        ):
            raise ValueError("historical encounter reason has incomplete or unused fields")
        key = (reason["system"], reason["code"])
        if key in seen_history_codes:
            raise ValueError(f"duplicate historical encounter reason: {key}")
        seen_history_codes.add(key)
    result: dict[str, dict[str, list[Clause]]] = {}
    seen_variants = set()
    seen_semantics: dict[tuple[str, str], str] = {}
    for archetype, variants in catalog["archetypes"].items():
        if not variants:
            raise ValueError(f"archetype {archetype} has no variants")
        result[archetype] = {}
        for variant in variants:
            expected_fields = {"id", "historyDays", "eventSpacingDays", "secondaryBranches", "events"}
            if set(variant) != expected_fields:
                raise ValueError(
                    f"variant {archetype}/{variant.get('id')} fields mismatch: "
                    f"unknown={sorted(set(variant)-expected_fields)}, missing={sorted(expected_fields-set(variant))}"
                )
            key = (archetype, variant["id"])
            if key in seen_variants:
                raise ValueError(f"duplicate variant: {archetype}/{variant['id']}")
            seen_variants.add(key)
            if len(variant["events"]) < 3 or len(variant["secondaryBranches"]) < 3:
                raise ValueError(f"variant {archetype}/{variant['id']} lacks material events/branches")
            if not isinstance(variant["historyDays"], int) or variant["historyDays"] < 0:
                raise ValueError(f"variant {archetype}/{variant['id']} has invalid historyDays")
            spacing_values = variant["eventSpacingDays"]
            if len(spacing_values) != len(variant["events"]) or any(
                not isinstance(value, int) or value < 0 for value in spacing_values
            ) or spacing_values != sorted(spacing_values):
                raise ValueError(f"variant {archetype}/{variant['id']} has invalid event spacing")
            if len(variant["secondaryBranches"]) != len(set(variant["secondaryBranches"])):
                raise ValueError(f"variant {archetype}/{variant['id']} has duplicate secondary branches")
            clauses = []
            spacing = variant["eventSpacingDays"]
            for position, event in enumerate(variant["events"]):
                allowed_event = {
                    "role", "family", "system", "code", "display", "setting",
                    "status", "lifecycle", "capability", "relationship", "domain", "shared", "workflowAction", "timingAfterPreviousHours",
                }
                unknown_event = set(event) - allowed_event
                if unknown_event:
                    raise ValueError(f"unused event fields {archetype}/{variant['id']}: {sorted(unknown_event)}")
                action = event.get("workflowAction")
                if action is not None:
                    # Workflow actions describe a review of a preceding ingredient,
                    # never a medication concept. Compile to observable ingredient,
                    # lifecycle and temporal clauses, not a narrative-code shortcut.
                    action_status = {"430193006": "active", "182840001": "stopped"}
                    prior_medication = next((row for row in reversed(variant["events"][:position])
                                             if row["role"] in {"prescription", "dispense", "medication-review"}), None)
                    if (action not in action_status or event["role"] != "medication-review"
                            or event.get("status") != action_status[action]
                            or prior_medication is None
                            or (event["system"], event["code"]) != (prior_medication["system"], prior_medication["code"])
                            or (action == "182840001" and event.get("lifecycle") != "stopped")):
                        raise ValueError(f"invalid executable medication workflow action: {archetype}/{variant['id']}")
                if event["role"] not in ROLE_TYPES:
                    raise ValueError(f"unknown event role {event['role']}")
                required_strings = {"role", "family", "system", "code", "display", "setting"}
                if any(not isinstance(event.get(field), str) or not event[field].strip() for field in required_strings):
                    raise ValueError(f"event selector is incomplete for {archetype}/{variant['id']}")
                semantic = (event["system"], event["code"])
                # One code may have several pathway roles/families, but its
                # canonical display cannot conflict within the reviewed catalog.
                prior_display = seen_semantics.setdefault(semantic, event["display"])
                if prior_display != event["display"]:
                    raise ValueError(f"conflicting display for {semantic}: {prior_display!r} != {event['display']!r}")
                spacing_delta = spacing[position] - spacing[position - 1] if clauses else None
                spacing_tolerance = max(2, round(spacing_delta * .25)) if spacing_delta is not None else None
                timing = event.get("timingAfterPreviousHours")
                if timing is not None:
                    if (archetype != "perioperative-surgical" or variant["id"] != "procedure-with-medication"
                            or position != 2 or event["role"] != "medication-administration"
                            or not isinstance(timing, dict)
                            or any(type(v) is not int for v in timing.values())
                            or timing != {"minimum": 2, "maximum": 2}
                            or (event["system"], event["code"]) != ("http://www.nlm.nih.gov/research/umls/rxnorm", "161")
                            or variant["events"][position - 1]["role"] != "procedure"):
                        raise ValueError("unsupported specific postoperative timing contract")
                allowed_statuses = ROLE_STATUSES[event["role"]]
                if event["role"] == "condition" and event.get("status") in {"active", "resolved"}:
                    allowed_statuses = (event["status"],)
                if event["role"] == "medication-review" and event.get("status") in {"active", "stopped"}:
                    allowed_statuses = (event["status"],)
                clauses.append(Clause(
                    id=f"{archetype}.{variant['id']}.{position + 1}.{event['family']}",
                    archetype=archetype, variant=variant["id"],
                    category="required" if position == 0 else "discriminating",
                    predicate="distinct-event-count", clinical_intent=event["family"], severity="error",
                    event_role=event["role"], resource_types=ROLE_TYPES[event["role"]],
                    code_system=event["system"], code=event["code"], family=event["family"],
                    minimum_distinct_events=1, maximum_distinct_events=None,
                    allowed_statuses=allowed_statuses,
                    forbidden_statuses=("entered-in-error",),
                    relation_to=clauses[-1].id if clauses else None,
                    temporal_relation="strictly-after" if clauses else None,
                    minimum_days=timing["minimum"] / 24 if timing else max(0, spacing_delta - spacing_tolerance) if spacing_delta is not None else None,
                    maximum_days=timing["maximum"] / 24 if timing else spacing_delta + spacing_tolerance if spacing_delta is not None else None,
                    discriminating=position > 0,
                    rationale="Observable event required by the reviewed material pathway template.",
                    source=path,
                    profiles=tuple(
                        f"http://hl7.org/fhir/StructureDefinition/{resource_type}"
                        for resource_type in ROLE_TYPES[event["role"]]
                    ),
                    value_semantics=(
                        "same-ingredient-review-" + action if action else
                        "reviewed-quantitative-or-coded-result" if event["role"] in {"assessment", "result", "growth-measurement"}
                        else "reviewed-coded-clinical-event"
                    ),
                    subject_reference_required=True,
                    source_participation_required=True,
                    positive_fixture_id=f"fixture:{archetype}:{variant['id']}:{position + 1}",
                    mutation_classes=MUTATION_CLASSES, nominal_offset_days=spacing[position],
                    alternate_codes=(
                        tuple(sorted({concept[0] for concept in INJURY_BRANCH_CONCEPTS.values()} - {event["code"]}))
                        if variant["id"] in {"minor-injury", "ordinary-injury"} and position < 2 else
                        ("174041007",) if variant["id"] == "emergency-appendectomy" and position == 1 else
                        ("52175",) if archetype == "medication-intensive" and variant["id"] == "dose-change" and position == 1 else ()
                    ),
                    alternate_codings=(
                        (("http://loinc.org", "82675-0"),)
                        if variant["id"] == "age-eligible-screening" and position in {1, 2} else ()
                    ),
                ))
            result[archetype][variant["id"]] = clauses
    vectors = {}
    for archetype, variants in result.items():
        vectors[archetype] = {}
        for variant, clauses in variants.items():
            vector = tuple((row.event_role, row.code_system, row.code) for row in clauses)
            if vector in vectors[archetype].values():
                raise ValueError(f"non-unique variant predicate vector in {archetype}: {variant}")
            vectors[archetype][variant] = vector
    return result


def _status(resource: dict[str, Any]) -> set[str]:
    values = {str(resource.get("status", ""))}
    values.update(code for system, code in codings(resource.get("clinicalStatus")) if system.endswith("condition-clinical"))
    return values - {""}


def _grammar_occurrence(resource):
    # A medication review is asserted at dateAsserted; effectivePeriod is the
    # exposure it reviews and may start before the prescription or span a hold.
    if resource.get("resourceType") == "MedicationStatement" and resource.get("dateAsserted"):
        return occurrence_from_resource({**resource, "effectivePeriod": None, "effectiveDateTime": resource["dateAsserted"]})
    return occurrence_from_resource(resource)


def evaluate_clause(clause: Clause, resources: list[dict[str, Any]], prior: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    matches = []
    forbidden_matches = []
    for resource in resources:
        if resource.get("resourceType") not in clause.resource_types:
            continue
        allowed_codings = {(clause.code_system, code) for code in (clause.code, *clause.alternate_codes)}
        allowed_codings.update(clause.alternate_codings)
        if not allowed_codings.intersection(semantic_condition_codings(resource)):
            continue
        statuses = _status(resource)
        if statuses & set(clause.forbidden_statuses):
            forbidden_matches.append(resource)
            continue
        if clause.allowed_statuses and not statuses & set(clause.allowed_statuses):
            continue
        subject_references = {
            value for value in references({
                "subject": resource.get("subject"), "patient": resource.get("patient"),
            })
        }
        if clause.subject_reference_required and not subject_references:
            continue
        if clause.source_participation_required and not resource.get("_observableSource"):
            continue
        matches.append(resource)
    passed = len(matches) >= clause.minimum_distinct_events and (
        clause.maximum_distinct_events is None or len(matches) <= clause.maximum_distinct_events
    ) and not forbidden_matches
    temporal = None
    if passed and clause.temporal_relation and prior:
        current_intervals = [_grammar_occurrence(row) for row in matches]
        prior_intervals = [_grammar_occurrence(row) for row in prior]
        current_intervals = [row for row in current_intervals if row]
        prior_intervals = [row for row in prior_intervals if row]
        temporal = any(
            strictly_before(left.interval, right.interval)
            and within_window(
                left.interval, right.interval,
                minimum=timedelta(days=clause.minimum_days or 0),
                maximum=timedelta(days=clause.maximum_days) if clause.maximum_days is not None else None,
            )
            for left in prior_intervals for right in current_intervals
        )
        passed = passed and temporal
    return {
        "clauseId": clause.id, "passed": passed,
        "observedCardinality": len(matches),
        "expected": {"minimum": clause.minimum_distinct_events, "maximum": clause.maximum_distinct_events},
        "temporalRelationPassed": temporal,
        "forbiddenEvidenceCount": len(forbidden_matches),
        "evidenceIds": sorted(row.get("_evidenceId") or "unidentified" for row in matches),
    }


def evaluate_supported_predicate(
    predicate: str, resources: list[dict[str, Any]], *,
    system: str | None = None, code: str | None = None,
    minimum: int = 1, statuses: tuple[str, ...] = (),
    direction: str = "stable", tolerance: float = 0.0,
    relation: str = "strictly-before", maximum_days: int | None = None,
    patient: dict[str, Any] | None = None, event_date: date | None = None,
    minimum_age_days: int = 0, maximum_age_days: int | None = None,
    permitted_sexes: tuple[str, ...] = (), second_system: str | None = None,
    second_code: str | None = None,
) -> bool:
    """Evaluate every predicate type promised by the v4 grammar contract.

    This registry is intentionally independent of crosswalk labels and generator
    tags so unit/mutation fixtures and future catalog clauses share semantics.
    """
    if predicate not in SUPPORTED_PREDICATES:
        raise ValueError(f"unknown typed predicate: {predicate}")

    def selected(selected_system: str | None, selected_code: str | None) -> list[dict[str, Any]]:
        return [
            resource for resource in resources
            if (selected_system is None or selected_code is None)
            or (selected_system, selected_code) in semantic_condition_codings(resource)
        ]

    rows = selected(system, code)
    if predicate == "distinct-event-count":
        return len({row.get("_evidenceId") or sha256_json(row) for row in rows}) >= minimum
    if predicate == "same-analyte-repeats":
        dates = {
            occurrence.interval.start.date() for row in rows
            if (occurrence := occurrence_from_resource(row)) is not None
        }
        return len(dates) >= max(2, minimum)
    if predicate == "trend-direction":
        ordered = sorted(
            (
                occurrence.interval.start,
                float(row.get("valueQuantity", {}).get("value")),
            )
            for row in rows
            if (occurrence := occurrence_from_resource(row)) is not None
            and isinstance(row.get("valueQuantity", {}).get("value"), (int, float))
        )
        return monotonic_trend([value for _when, value in ordered], direction, tolerance)
    if predicate == "active-resolved-status":
        return any(_status(row) & set(statuses) for row in rows)
    if predicate == "medication-exposure":
        medication_types = {
            "MedicationRequest", "MedicationDispense", "MedicationStatement", "MedicationAdministration",
        }
        return any(
            row.get("resourceType") in medication_types and occurrence_from_resource(row)
            and any(system.endswith("/rxnorm") for system, _code in codings(row.get("medicationCodeableConcept")))
            and bool(row.get("dosageInstruction") or row.get("dosage"))
            for row in rows
        )
    if predicate == "dose-status-transition":
        transitions = []
        for row in rows:
            occurrence = occurrence_from_resource(row)
            dose = next((
                item.get("doseQuantity", {}).get("value")
                for instruction in row.get("dosageInstruction", [])
                for item in instruction.get("doseAndRate", [])
                if isinstance(item.get("doseQuantity", {}).get("value"), (int, float))
            ), None)
            if occurrence and dose is not None:
                transitions.append((occurrence.interval.start, float(dose), tuple(sorted(_status(row))), tuple(sorted(codings(row.get("medicationCodeableConcept"))))))
        transitions.sort()
        return len(transitions) >= 2 and any(
            earlier[3] == later[3] and bool(earlier[3]) and (earlier[1] != later[1] or earlier[2] != later[2])
            for earlier, later in zip(transitions, transitions[1:])
        )
    if predicate == "linked-events":
        targets = selected(second_system, second_code)
        target_ids = {
            value for row in targets
            for value in (row.get("_fullUrl"), row.get("id"), row.get("_evidenceId")) if value
        }
        return bool(target_ids) and any(set(references(row)) & target_ids for row in rows)
    if predicate == "age-sex-risk-eligibility":
        if not patient or not patient.get("birthDate") or not event_date:
            return False
        age_days = (event_date - date.fromisoformat(patient["birthDate"])).days
        return (
            age_days >= minimum_age_days
            and (maximum_age_days is None or age_days <= maximum_age_days)
            and (not permitted_sexes or patient.get("gender") in permitted_sexes)
        )
    if predicate == "temporal-relation":
        left = [occurrence_from_resource(row) for row in rows]
        right = [occurrence_from_resource(row) for row in selected(second_system, second_code)]
        left = [row.interval for row in left if row]
        right = [row.interval for row in right if row]
        relations = {
            "strictly-before": strictly_before, "strictly-after": strictly_after,
            "starts-before": starts_before, "ends-after": ends_after,
            "contains": contains, "overlaps": overlaps, "active-during": active_during,
        }
        if relation == "within-window":
            return any(within_window(one, two, maximum=timedelta(days=maximum_days) if maximum_days is not None else None) for one in left for two in right)
        if relation not in relations:
            raise ValueError(f"unknown temporal relation: {relation}")
        return any(relations[relation](one, two) for one in left for two in right)
    if predicate == "source-participation":
        return len({row.get("_observableSource") for row in rows if row.get("_observableSource")}) >= minimum
    if predicate == "forbidden-combination":
        return not (rows and selected(second_system, second_code))
    raise AssertionError(f"unhandled typed predicate: {predicate}")


def evaluate_variant(
    clauses: list[Clause], resources: list[dict[str, Any]], patient: dict[str, Any] | None = None,
    timeline_anchor: str | None = None,
) -> list[dict[str, Any]]:
    if clauses and clauses[0].archetype == "pediatric" and patient and timeline_anchor:
        born = datetime.fromisoformat(patient['birthDate'] + 'T00:00:00+00:00')
        horizon = datetime.fromisoformat(timeline_anchor.replace('Z', '+00:00')) - timedelta(microseconds=1)
        available = max(0, (horizon - born - timedelta(days=7)).days)
        maximum = max(c.nominal_offset_days for c in clauses)
        if maximum > available:
            offsets = [round(c.nominal_offset_days * available / maximum) for c in clauses]
            adjusted = []
            for i, clause in enumerate(clauses):
                if i:
                    delta = offsets[i] - offsets[i-1]
                    tolerance = max(2, round(delta * .25))
                    clause = replace(clause, minimum_days=max(0, delta-tolerance), maximum_days=delta+tolerance)
                adjusted.append(clause)
            clauses = adjusted
    results = []
    previous_matches: list[dict[str, Any]] = []
    for clause in clauses:
        result = evaluate_clause(clause, resources, previous_matches)
        results.append(result)
        previous_matches = [
            resource for resource in resources
            if resource.get("resourceType") in clause.resource_types
            and any((clause.code_system, code) in semantic_condition_codings(resource)
                    for code in (clause.code, *clause.alternate_codes))
        ]
    return results


def _observable_graph(resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the black-box FHIR evidence boundary used by release grammar."""
    output = []
    for resource in resources:
        source = resource.get("_source")
        clean = strip_generator_metadata(resource)
        clean["_observableSource"] = source or "unknown-source"
        clean["_evidenceId"] = sha256_json(clean)
        output.append(clean)
    return output


def _fixture_resource(clause: Clause, when: datetime) -> dict[str, Any]:
    resource_type = clause.resource_types[0]
    timestamp = when.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    coding = {"system": clause.code_system, "code": clause.code, "display": clause.family}
    resource: dict[str, Any] = {
        "resourceType": resource_type,
        "meta": {"profile": list(clause.profiles)},
        "_observableSource": "fixture-source",
        "_evidenceId": clause.positive_fixture_id,
    }
    if resource_type == "Immunization":
        resource.update({
            "status": clause.allowed_statuses[0], "vaccineCode": {"coding": [coding]},
            "patient": {"reference": "Patient/fixture"}, "occurrenceDateTime": timestamp,
            "primarySource": True,
        })
        return resource
    resource["subject"] = {"reference": "Patient/fixture"}
    if resource_type == "Encounter":
        resource.update({
            "status": clause.allowed_statuses[0], "reasonCode": [{"coding": [coding]}],
            "period": {"start": timestamp, "end": timestamp},
        })
    elif resource_type == "Condition":
        resource.update({
            "code": {"coding": [coding]},
            "clinicalStatus": {"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                "code": clause.allowed_statuses[0],
            }]},
            "onsetDateTime": timestamp,
        })
    elif resource_type == "Observation":
        resource.update({
            "status": clause.allowed_statuses[0], "code": {"coding": [coding]},
            "effectiveDateTime": timestamp,
            "valueQuantity": {"value": 1.0, "system": "http://unitsofmeasure.org", "code": "1"},
        })
    elif resource_type == "ServiceRequest":
        resource.update({
            "status": clause.allowed_statuses[0], "intent": "order",
            "code": {"coding": [coding]}, "authoredOn": timestamp,
        })
    elif resource_type == "Specimen":
        resource.update({
            "status": clause.allowed_statuses[0], "type": {"coding": [coding]},
            "collection": {"collectedDateTime": timestamp},
        })
    elif resource_type in {"MedicationRequest", "MedicationDispense", "MedicationStatement", "MedicationAdministration"}:
        resource.update({
            "status": clause.allowed_statuses[0],
            "medicationCodeableConcept": {"coding": [coding]},
        })
        if resource_type == "MedicationRequest":
            resource.update({"intent": "order", "authoredOn": timestamp})
        elif resource_type == "MedicationDispense":
            resource["whenHandedOver"] = timestamp
        else:
            resource["effectiveDateTime"] = timestamp
    elif resource_type == "Procedure":
        resource.update({
            "status": clause.allowed_statuses[0], "code": {"coding": [coding]},
            "performedDateTime": timestamp,
        })
    elif resource_type == "DocumentReference":
        resource.update({
            "status": clause.allowed_statuses[0], "type": {"coding": [coding]},
            "date": timestamp,
        })
    else:
        raise ValueError(f"no positive fixture builder for {resource_type}")
    return resource


def _set_fixture_status(resource: dict[str, Any], status: str) -> None:
    if resource.get("resourceType") == "Condition":
        resource["clinicalStatus"] = {"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": status,
        }]}
    else:
        resource["status"] = status


def _remove_subject_reference(resource: dict[str, Any]) -> None:
    resource.pop("subject", None)
    resource.pop("patient", None)


def replace_code(value: Any, system: str, code: str) -> None:
    if isinstance(value, dict):
        if value.get("system") == system and value.get("code") == code:
            value["system"] = "urn:hdp-test-bundle:mutation"
            value["code"] = "removed-decisive-code"
            value.pop("display", None)
        for child in value.values():
            replace_code(child, system, code)
    elif isinstance(value, list):
        for child in value:
            replace_code(child, system, code)


def _clause_mutation_evidence(
    clause: Clause, prior_clause: Clause | None, compatible_control: tuple[str, str] | None,
) -> dict[str, Any]:
    base = datetime(2024, 1, 1, 10, tzinfo=timezone.utc)
    # A reviewed exact sub-day relation must also constrain its positive fixture.
    gap_days = (clause.minimum_days if clause.minimum_days is not None
                and clause.minimum_days == clause.maximum_days
                else max(1, clause.minimum_days or 1))
    prior = [_fixture_resource(prior_clause, base)] if prior_clause else None
    current_time = base + timedelta(days=gap_days) if prior_clause else base
    current = _fixture_resource(clause, current_time)
    positive = evaluate_clause(clause, [current], prior)

    def result(resources: list[dict[str, Any]], prior_rows: list[dict[str, Any]] | None = prior) -> bool:
        return evaluate_clause(clause, resources, prior_rows)["passed"]

    removed_failed = not result([])
    compatible = copy.deepcopy(current)
    compatible_system, compatible_code = compatible_control or (clause.code_system, f"{clause.code}-compatible-control")
    replace_code(compatible, clause.code_system, clause.code)
    # replace_code deliberately uses an incompatible namespace; rewrite it to a
    # reviewed sibling selector for the compatible-control mutation.
    def set_control(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("system") == "urn:hdp-test-bundle:mutation" and value.get("code") == "removed-decisive-code":
                value.update({"system": compatible_system, "code": compatible_code})
            for child in value.values():
                set_control(child)
        elif isinstance(value, list):
            for child in value:
                set_control(child)
    set_control(compatible)
    compatible_failed = not result([compatible])
    incompatible = copy.deepcopy(current)
    replace_code(incompatible, clause.code_system, clause.code)
    incompatible_failed = not result([incompatible])

    if prior_clause:
        reversed_current = _fixture_resource(clause, base)
        reversed_prior = [_fixture_resource(prior_clause, base + timedelta(days=gap_days + 1))]
        temporal_failed = not result([reversed_current], reversed_prior)
    else:
        temporal_failed = True
    lifecycle = copy.deepcopy(current)
    _set_fixture_status(lifecycle, "entered-in-error")
    lifecycle_failed = not result([lifecycle])
    broken_reference = copy.deepcopy(current)
    _remove_subject_reference(broken_reference)
    reference_failed = not result([broken_reference])
    forbidden = copy.deepcopy(current)
    _set_fixture_status(forbidden, "entered-in-error")
    forbidden["_evidenceId"] = f"{clause.positive_fixture_id}:forbidden"
    forbidden_failed = not result([current, forbidden])
    no_source = copy.deepcopy(current)
    no_source.pop("_observableSource", None)
    source_failed = not result([no_source])
    cosmetic = copy.deepcopy(current)
    cosmetic["id"] = "changed-id"
    cosmetic["note"] = [{"text": "Cosmetic note"}]
    cosmetic.setdefault("meta", {}).setdefault("tag", []).append({
        "system": "urn:hdp-test-bundle:synthetic-generator-tag", "code": "changed",
    })
    for node in (cosmetic,):
        if isinstance(node.get("valueQuantity", {}).get("value"), (int, float)):
            node["valueQuantity"]["value"] += 0.001
    cosmetic_passed = result([cosmetic])
    outcomes = {
        "remove-required-event": removed_failed,
        "compatible-control-code": compatible_failed,
        "incompatible-code": incompatible_failed,
        "reverse-or-break-window": temporal_failed,
        "lifecycle-status": lifecycle_failed,
        "broken-reference": reference_failed,
        "insert-forbidden-event": forbidden_failed,
        "strip-source-capture": source_failed,
        "cosmetic-only": cosmetic_passed,
    }
    return {
        "clauseId": clause.id, "positiveFixtureId": clause.positive_fixture_id,
        "positiveFixturePassed": positive["passed"], "mutationResults": outcomes,
        "status": "pass" if positive["passed"] and all(outcomes.values()) else "fail",
    }


def build_typed_grammar_audit(
    bundles: dict[str, dict[str, Any]], crosswalk: dict[str, Any], spec: dict[str, Any], catalog_path: str,
) -> dict[str, Any]:
    grammar = compile_grammar(catalog_path)
    raw_graphs, patients = patient_graphs(bundles, spec)
    # Supplemental source-local capture episodes have their own complete
    # mapping and branch-graph audit. Variant inference concerns the reviewed
    # typed pathway and must not borrow events from a separate episode.
    pathway_graphs = {
        cohort: [resource for resource in resources if not is_supplemental_capture(resource)]
        for cohort, resources in raw_graphs.items()
    }
    graphs = {cohort: _observable_graph(resources) for cohort, resources in pathway_graphs.items()}
    failures = []
    patient_results = []
    excluded = []
    for row in crosswalk["patients"]:
        if not row.get("ordinaryPathwayEligible", True):
            excluded.append({"cohortId": row["cohortId"], "reason": row.get("evaluationClass", "non-ordinary")})
            continue
        cohort = row["cohortId"]
        archetype = row["primaryArchetype"]
        variant = row["pathwayVariant"]
        results = evaluate_variant(grammar[archetype][variant], graphs.get(cohort, []), patients.get(cohort), spec["timelineAnchor"])
        failed = [item for item in results if not item["passed"]]
        if variant in {"age-eligible-screening", "emergency-appendectomy", "dispense-follow-up", "diabetes-stable"} \
                or variant == "reproductive-follow-up" and row.get("secondaryBranch") == "nonpregnant-follow-up":
            branch = row["secondaryBranch"]
            if not branch_proof(branch, pathway_graphs.get(cohort, []), patients[cohort]["birthDate"]):
                failed.append({
                    "id": f"{archetype}.{variant}.{branch}.structural-proof",
                    "passed": False, "observedCardinality": 0,
                    "expected": {"independentFhirBranchProof": branch},
                    "temporalRelationPassed": False, "evidenceIds": [],
                })
        if variant in {"minor-injury", "ordinary-injury"}:
            branch = row["secondaryBranch"]
            expected_code = INJURY_BRANCH_CONCEPTS[branch][0]
            for resource_type in ("Encounter", "Condition"):
                observed = [resource for resource in graphs.get(cohort, [])
                            if resource.get("resourceType") == resource_type
                            and ("http://snomed.info/sct", expected_code) in semantic_condition_codings(resource)]
                if not observed:
                    failed.append({
                        "id": f"{archetype}.{variant}.{branch}.{resource_type}",
                        "passed": False, "observedCardinality": 0,
                        "expected": {"secondaryBranchCode": expected_code, "resourceType": resource_type},
                        "temporalRelationPassed": None, "evidenceIds": [],
                    })
        branch = row.get("secondaryBranch")
        if branch and material_branch_proof(branch, pathway_graphs.get(cohort, []), patients[cohort]["birthDate"]) is False:
            failed.append({"id": f"{archetype}.{variant}.{branch}.observable-content", "passed": False,
                           "observedCardinality": 0, "expected": {"observableBranch": branch},
                           "temporalRelationPassed": False, "evidenceIds": []})
        passing_variants = [
            candidate for candidate, candidate_clauses in grammar[archetype].items()
            if all(item["passed"] for item in evaluate_variant(candidate_clauses, graphs.get(cohort, []), patients.get(cohort), spec["timelineAnchor"]))
        ]
        if passing_variants != [variant]:
            failures.append({
                "cohortId": cohort, "archetype": archetype, "variant": variant,
                "clauseId": "variant-inference", "passed": False,
                "observedCardinality": len(passing_variants),
                "expected": {"exactVariant": variant}, "temporalRelationPassed": None,
                "evidenceIds": passing_variants,
            })
        patient_results.append({
            "cohortId": cohort, "archetype": archetype, "variant": variant,
            "inferredVariants": passing_variants, "clauses": results,
        })
        failures.extend({"cohortId": cohort, "archetype": archetype, "variant": variant, **item} for item in failed)
    all_clauses = [
        clause for variants in grammar.values() for clauses in variants.values() for clause in clauses
    ]
    mutation_rows = []
    for archetype, variants in grammar.items():
        for variant, clauses in variants.items():
            for position, clause in enumerate(clauses):
                control = next((
                    (candidate.code_system, candidate.code) for candidate in all_clauses
                    if candidate.id != clause.id and candidate.event_role == clause.event_role
                    and (candidate.code_system, candidate.code) != (clause.code_system, clause.code)
                ), None)
                row = _clause_mutation_evidence(
                    clause, clauses[position - 1] if position else None, control,
                )
                mutation_rows.append(row)
                if row["status"] != "pass":
                    failures.append({
                        "cohortId": "mutation-fixture", "archetype": archetype,
                        "variant": variant, "clauseId": clause.id, "passed": False,
                        "observedCardinality": 0,
                        "expected": {"mutationClasses": list(MUTATION_CLASSES)},
                        "temporalRelationPassed": None, "evidenceIds": [],
                        "mutationResults": row["mutationResults"],
                    })
    uncovered = sum(row["status"] != "pass" for row in mutation_rows)
    return {
        "schemaVersion": "4.1.0", "branchCoverage": branch_coverage_audit(bundles, crosswalk, spec), "evaluationBoundary": {
            "name": "tag-stripped-observable-fhir",
            "removed": [
                "synthetic tags", "hidden truth", "answer keys", "notes", "narrative",
                "resource ids", "identifiers", "displays as evidence", "meta.source",
            ],
            "sourceParticipation": "derived from source Bundle membership, not a generator tag",
        },
        "predicateRegistry": sorted(SUPPORTED_PREDICATES),
        "clauseDefinitions": [
            asdict(clause) for clause in all_clauses
        ],
        "patientCount": len(patient_results), "failureCount": len(failures),
        "excludedCount": len(excluded), "exclusions": excluded,
        "mutationCount": len(mutation_rows) * len(MUTATION_CLASSES),
        "mutationClauseCount": len(mutation_rows), "uncoveredMutationCount": uncovered,
        "mutations": mutation_rows,
        "failures": failures, "patients": patient_results,
        "status": "pass" if not failures else "fail",
    }
