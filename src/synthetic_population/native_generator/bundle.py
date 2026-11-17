from __future__ import annotations

import copy
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .demographics import demographic_truth, patient_resource
from .archetypes import archetype_rule, distribution, primary_archetype, source_event_time
from .buckets import coverage_role
from .catalogs import input_path, load_catalog as load_versioned_catalog
from .geography import choose_locality, load_catalog
from .ids import cohort_id, deterministic_uuid, full_url, strong_identifier
from .metadata import meta
from .predicates import evaluate, evaluate_variant, subset_truth_table, subset_truth_table_variant
from .temporal import occurrence_from_resource, parse_fhir_time
from .problems import additional_fixture_conclusions, conditional_fixture_conclusions, SCENARIO_VARIANTS, allocation, scenario_name
from .pathways import care_pathway, utilizes_source
from .profiles import ENCOUNTER_PROFILES, PROFILES
from .sensitivity import classify as classify_sensitivity, fact_for_patient, owning_source, requires_source
from .clinical_data import quantitative_result, repeated_value
from .terminology import iter_codings
from .condition_coding import select_condition_coding
from .utils import read_json, sha256_json
from .care_pathway import build_patient_pathway
from .nested_allocation import nested, block_position
from .capture import capture_ledger_rows, project_capture
from .fhir_projection import project_pathway
from .lifecycle import assign_deaths
from .medication_safety import dosage_for, regimen_catalog, compatible_ingredient, ingredient


def _walk_replace(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _walk_replace(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_walk_replace(item, replacements) for item in value]
    if isinstance(value, str):
        return replacements.get(value, value)
    return value


def _order_transaction_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Put frequently referenced fullUrls first without changing any entry.

    This reduces their positions in a linear Bundle.entry reference scan.
    Official full-envelope runtime still requires direct measurement.
    """
    known = {entry["fullUrl"] for entry in entries}
    references: Counter[str] = Counter()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "reference" and isinstance(child, str) and child in known:
                    references[child] += 1
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for entry in entries:
        visit(entry["resource"])
    return sorted(entries, key=lambda entry: -references[entry["fullUrl"]])


def _reference_identifier(spec: dict, organization: str, resource_type: str, value: str) -> dict:
    identifier = strong_identifier(spec, organization, resource_type, value)
    identifier.pop("type", None)
    return identifier


def _master_resources(spec: dict, organization: dict) -> list[tuple[str, dict]]:
    org_id = organization["id"]
    version = spec["datasetVersion"]
    org_url = full_url(version, org_id, "master", "Organization")
    practitioner_url = full_url(version, org_id, "master", "Practitioner")
    role_url = full_url(version, org_id, "master", "PractitionerRole")
    org_uuid = org_url.removeprefix("urn:uuid:")
    practitioner_uuid = practitioner_url.removeprefix("urn:uuid:")
    role_uuid = role_url.removeprefix("urn:uuid:")
    org = {
        "resourceType": "Organization", "meta": meta(spec, org_id, PROFILES["Organization"], "organization-master", archetype="organization-master"),
        "identifier": [strong_identifier(spec, org_id, "Organization", org_uuid)], "active": True,
        "type": [{"coding": [{
            "system": "http://fhir.health.gov.il/cs/institution-type-moh",
            "code": "73" if org_id == "hmo" else "1",
            "display": "קופות חולים" if org_id == "hmo" else "אשפוז כללי",
        }]}],
        "name": organization["name"], "text": {"status": "generated", "div": f"<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic organization {organization['name']}.</div>"},
        "partOf": {"identifier": {"system": "urn:hdp-test-bundle:synthetic:parent-organization", "value": "national-test-network"}},
        "address": [{"use": "work", "type": "both", "line": ["Synthetic Health Campus 1"], "city": "Jerusalem", "country": "IL"}],
    }
    practitioner = {
        "resourceType": "Practitioner", "meta": meta(spec, org_id, PROFILES["Practitioner"], "practitioner-master", archetype="organization-master"),
        "identifier": [
            strong_identifier(spec, org_id, "Practitioner", practitioner_uuid),
            {"system": "http://fhir.health.gov.il/identifier/encrypted-id-primary-moh", "value": f"HTEST-{deterministic_uuid(version, org_id, 'fake-practitioner-token')}"},
        ],
        "active": True, "name": [{"family": "Synthetic", "given": ["Clinician"]}],
        "text": {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic clinician.</div>"},
    }
    role = {
        "resourceType": "PractitionerRole", "meta": meta(spec, org_id, PROFILES["PractitionerRole"], "practitioner-master", archetype="organization-master"),
        "identifier": [strong_identifier(spec, org_id, "PractitionerRole", role_uuid)], "active": True,
        "practitioner": {"reference": practitioner_url}, "organization": {"reference": org_url},
        "code": [{"coding": [{"system": "http://snomed.info/sct", "code": "309343006", "display": "Physician"}]}],
        "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "408443003", "display": "General medical practice"}]}],
        "text": {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic clinician role.</div>"},
    }
    return [(org_url, org), (practitioner_url, practitioner), (role_url, role)]


def _problem_for_resource(problems: list[dict], organization: str, resource_type: str) -> tuple[list[str], list[str]]:
    selected = []
    for position, problem in enumerate(problems):
        if problem["class"] == "cross-source-clinical":
            targets = {
                "renal-medication-risk": {"hmo": "Condition", "hospital-a": "Observation", "hospital-b": "MedicationRequest"},
                "bleeding-risk": {"hmo": "MedicationRequest", "hospital-a": "Observation", "hospital-b": "Condition"},
                "allergy-treatment-risk": {"hmo": "AllergyIntolerance", "hospital-a": ("Observation", "ServiceRequest"), "hospital-b": "MedicationRequest"},
            }
            target = targets[problem["scenario"]][organization]
            if resource_type == target or (isinstance(target, tuple) and resource_type in target):
                selected.append(problem)
        elif problem["class"] == "data-quality":
            org_order = ("hmo", "hospital-a", "hospital-b")
            chosen_type = {
                "conflicting-diagnosis": "Condition",
                "discordant-lab": "Observation",
                "timestamp-order": "Observation",
                "semantically-wrong-code": "Condition",
                "unit-mismatch": "Observation",
                "stale-demographics": "Patient",
                "probable-duplicate": "Patient",
            }[problem["kind"]]
            chosen_index = (int(problem["problemId"].split("-")[-1]) + position) % 3
            chosen_org = org_order[chosen_index]
            cross = next((item for item in problems if item["class"] == "cross-source-clinical"), None)
            if cross:
                reserved = {
                    "renal-medication-risk": {"hmo": {"Condition"}, "hospital-a": {"Observation"}, "hospital-b": {"MedicationRequest"}},
                    "bleeding-risk": {"hmo": {"MedicationRequest"}, "hospital-a": {"Observation"}, "hospital-b": {"Condition"}},
                    "allergy-treatment-risk": {"hmo": {"AllergyIntolerance"}, "hospital-a": {"Observation", "ServiceRequest"}, "hospital-b": {"MedicationRequest"}},
                }[cross["scenario"]]
                for _ in range(3):
                    if chosen_type not in reserved.get(chosen_org, set()):
                        break
                    chosen_index = (chosen_index + 1) % 3
                    chosen_org = org_order[chosen_index]
            if organization == chosen_org and resource_type == chosen_type:
                selected.append(problem)
    return [item["problemId"] for item in selected], [item["kind"] for item in selected]


def _problem_variant(problems: list[dict], organization: str, resource_type: str, kind: str) -> str | None:
    ids, _kinds = _problem_for_resource(problems, organization, resource_type)
    return next((row.get("variant") for row in problems if row["problemId"] in ids and row["kind"] == kind), None)


def _sensitivity(spec: dict, index: int) -> tuple[str | None, list[str], bool]:
    if spec.get("schemaVersion") == "2.0.0":
        return None, [], False
    rules = spec["sensitivity"]
    if index % rules["veryRestrictedNoPatientAccessEvery"] == 0:
        return "V", ["SDV"], True
    if index % rules["veryRestrictedHivEvery"] == 0:
        return "V", ["HIV"], False
    if index % rules["restrictedMentalHealthEvery"] == 0:
        return "R", ["MH"], False
    return None, [], False


def _source_episode_base(spec: dict, source: str, index: int, birth_date: str | None = None) -> datetime:
    """Actual supplemental episode start, shared with scenario scheduling."""
    base = source_event_time(spec, source, index)
    anchor = datetime.fromisoformat(spec["timelineAnchor"].replace("Z", "+00:00"))
    if birth_date:
        born = datetime.fromisoformat(birth_date + "T08:00:00+00:00")
        earliest = born + timedelta(days=7)
        if base < earliest:
            base = min(earliest + timedelta(days={"hmo": 0, "hospital-a": 30, "hospital-b": 60}[source]),
                       anchor - timedelta(days=2))
            base = max(base, born + timedelta(days=1))
    return base


def _clinical_resources(spec: dict, organization: dict, index: int, patient_url: str, problems: list[dict], birth_date: str | None = None,
                        *, full_capture: bool = False, gender: str | None = None,
                        condition_coding_selection: dict[str, str] | None = None,
                        fixture_context: bool = False, renal_stage4: bool = False) -> list[tuple[str, dict]]:
    org_id = organization["id"]
    version = spec["datasetVersion"]
    org_url = full_url(version, org_id, "master", "Organization")
    practitioner_url = full_url(version, org_id, "master", "Practitioner")
    encounter_url = full_url(version, org_id, cohort_id(index), "Encounter", "primary")
    condition_url = full_url(version, org_id, cohort_id(index), "Condition", "primary")
    observation_url = full_url(version, org_id, cohort_id(index), "Observation", "primary")
    specimen_url = full_url(version, org_id, cohort_id(index), "Specimen", "primary")
    request_url = full_url(version, org_id, cohort_id(index), "ServiceRequest", "lab")
    archetype = primary_archetype(spec, index) if spec.get("schemaVersion") == "2.0.0" else "strict-patient-clinical"
    trajectory_followup_days = 90
    if spec.get("schemaVersion") == "2.0.0":
        base = _source_episode_base(spec, org_id, index, birth_date)
        if spec.get("clinicalRealism") and not problems and not fixture_context and archetype in {
            "chronic-care", "medication-intensive", "oncology", "rare-complex",
        }:
            # Longitudinal gates use actual events, not diagnosis onset or
            # metadata. Place the initial episode more than one year before the
            # deterministic follow-up while retaining birth-date safety.
            trajectory_followup_days = 400 + (index * 17 + len(org_id) * 11) % 181
            shifted = base - timedelta(days=trajectory_followup_days)
            if birth_date:
                shifted = max(
                    shifted,
                    datetime.fromisoformat(birth_date + "T08:00:00+00:00") + timedelta(days=30),
                )
                trajectory_followup_days = max(90, (base - shifted).days)
            base = shifted
        source_model = load_versioned_catalog(spec["sourceModels"]["catalog"])["sources"][org_id]
    else:
        anchor = datetime.fromisoformat(spec["timelineAnchor"].replace("Z", "+00:00"))
        history_days = 6 * 365
        day_offset = history_days if index == 1 else ((index * 7919) % history_days)
        base = anchor - timedelta(days=day_offset, hours=16 - ((index + len(org_id)) % 8))
        source_model = None
    if (renal_stage4 and birth_date and source_model
            and source_model["observation"]["code"] == "2160-0"):
        born = datetime.fromisoformat(birth_date + "T00:00:00+00:00")
        try:
            adult = born.replace(year=born.year + 18)
        except ValueError:
            adult = born.replace(year=born.year + 18, month=3, day=1)
        # Shift the entire supplemental episode together; its specimen/order
        # chronology must follow the same adult floor as the renal result.
        base = max(base, adult)
    if spec.get("schemaVersion") == "2.0.0":
        anchor = datetime.fromisoformat(spec["timelineAnchor"].replace("Z", "+00:00"))
        if base >= anchor:
            # Bound only an episode that is still at the snapshot after its
            # longitudinal/adult scheduling. Valid historical episodes stay intact.
            base = anchor - timedelta(days=4)
            if birth_date and base < datetime.fromisoformat(birth_date + "T08:00:00+00:00") + timedelta(days=1):
                raise ValueError("supplemental episode cannot fit between birth and snapshot")
            if (renal_stage4 and birth_date and source_model
                    and source_model["observation"]["code"] == "2160-0" and base < adult):
                raise ValueError("adult renal episode cannot fit before snapshot")
    pids, pkinds = _problem_for_resource(problems, org_id, "Encounter")
    confidentiality, sensitivity, nopat = _sensitivity(spec, index)
    pathway = care_pathway(
        spec, index, archetype,
        force_all_sources=any(problem["class"] == "cross-source-clinical" for problem in problems),
        birth_date=birth_date,
    ) if spec.get("schemaVersion") == "2.0.0" else None
    encounter_policy = pathway["encounters"][org_id] if pathway else {
        "class": "AMB" if org_id == "hmo" else "IMP", "role": "legacy", "durationHours": 2,
    }
    if pathway and not full_capture and org_id != "hmo" and "captureVariant" in encounter_policy:
        # Keep the clinical encounter class consistent with the emitted
        # source-local workflow, including short admissions and emergency care.
        variant = encounter_policy["captureVariant"]
        encounter_policy = dict(encounter_policy)
        if "emergency" in variant:
            encounter_policy["class"] = "EMER"
            encounter_policy["role"] = "emergency"
        elif variant in {"inpatient-procedure-workup", "short-admission"}:
            encounter_policy["class"] = "IMP"
            encounter_policy["role"] = "inpatient"
        elif variant in {"specialty-referral", "imaging-referral", "discharge-follow-up", "transfer-summary"}:
            encounter_policy["class"] = "AMB"
            encounter_policy["role"] = "specialty-ambulatory"
    encounter_class = encounter_policy["class"]
    encounter_type = "community-hmo-ambulatory-hdp" if org_id == "hmo" else (
        "hospitalization-hdp" if encounter_class == "IMP" else "supplemental-encounter-hdp"
    )
    encounter_profile = (
        ENCOUNTER_PROFILES[org_id]
        if org_id == "hmo" or encounter_class == "IMP"
        else "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-encounter-supplemental"
    )
    reason = encounter_policy.get("reason", {
        "system": "http://snomed.info/sct",
        "code": "185349003" if org_id == "hmo" else "161891005" if org_id == "hospital-a" else "386661006",
        "display": "Encounter for check up" if org_id == "hmo" else "Backache" if org_id == "hospital-a" else "Fever",
    })
    encounter = {
        "resourceType": "Encounter",
        "meta": meta(spec, org_id, encounter_profile, "encounter", confidentiality=confidentiality,
                     sensitivity=sensitivity, nopat=nopat, problem_ids=pids, problem_kinds=pkinds, archetype=archetype),
        "identifier": [strong_identifier(spec, org_id, "Encounter", encounter_url.removeprefix("urn:uuid:"))],
        "status": "finished",
        "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": encounter_class},
        "type": [{"coding": [{"system": "http://fhir.health.gov.il/cs/il-core-encounter-type", "code": encounter_type}]}],
        "subject": {"reference": patient_url},
        "participant": [{"individual": {"reference": practitioner_url}}],
        "serviceProvider": {"reference": org_url},
        "period": {"start": base.isoformat().replace("+00:00", "Z"), "end": (base + timedelta(hours=encounter_policy["durationHours"])).isoformat().replace("+00:00", "Z")},
        "reasonCode": [{"coding": [reason]}],
        "reasonReference": [{"type": "Condition", "identifier": _reference_identifier(spec, org_id, "Condition", condition_url.removeprefix("urn:uuid:"))}],
        "diagnosis": [{"condition": {"type": "Condition", "identifier": _reference_identifier(spec, org_id, "Condition", condition_url.removeprefix("urn:uuid:"))}, "use": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/diagnosis-role", "code": "AD", "display": "Admission diagnosis"}]}, "rank": 1}],
        "serviceType": {"coding": [{"system": "http://snomed.info/sct", "code": "700232004", "display": "General medical service"}]},
        "text": {"status": "generated", "div": f"<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic {org_id} encounter.</div>"},
    }
    pids, pkinds = _problem_for_resource(problems, org_id, "Condition")
    condition = {
        "resourceType": "Condition",
        "meta": meta(spec, org_id, PROFILES["Condition"], "diagnosis", confidentiality=confidentiality,
                     sensitivity=sensitivity, nopat=nopat, problem_ids=pids, problem_kinds=pkinds, archetype=archetype),
        "identifier": [strong_identifier(spec, org_id, "Condition", condition_url.removeprefix("urn:uuid:"))],
        "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
        "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed"}]},
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-category", "code": "problem-list-item"}]}],
        "code": {"coding": [source_model["condition"] if source_model else {"system": "http://snomed.info/sct", "code": "38341003", "display": "Hypertensive disorder"}]},
        "subject": {"reference": patient_url, "identifier": {"system": spec["identity"]["cohortIdentifierSystem"], "value": cohort_id(index)}}, "encounter": {"reference": encounter_url},
        "recordedDate": base.date().isoformat(), "recorder": {"reference": practitioner_url},
        "text": {"status": "generated", "div": f"<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic {org_id} diagnosis.</div>"},
    }
    if spec.get("schemaVersion") == "2.0.0" and archetype in {"generally-healthy", "pediatric"}:
        code = ("195662009", "Acute upper respiratory infection") if archetype == "pediatric" else ("34014006", "Viral disease")
        condition["code"] = {"coding": [{"system": "http://snomed.info/sct", "code": code[0], "display": code[1]}]}
        condition["clinicalStatus"]["coding"][0]["code"] = "resolved"
        condition["abatementDateTime"] = (base + timedelta(days=7)).date().isoformat()
    elif spec.get("schemaVersion") == "2.0.0" and archetype == "oncology":
        condition["code"] = {"coding": [{"system": "http://snomed.info/sct", "code": "363346000", "display": "Malignant neoplastic disease"}]}
    elif spec.get("schemaVersion") == "2.0.0" and archetype == "perioperative-surgical" and org_id == "hospital-a":
        condition["code"] = {"coding": [{"system": "http://snomed.info/sct", "code": "74400008", "display": "Appendicitis"}]}
    if spec.get("schemaVersion") == "2.0.0":
        span_days = archetype_rule(spec, archetype)["minimumSpanDays"] + (index * 53 + len(org_id) * 17) % 241
        onset = base - timedelta(days=span_days)
        if birth_date:
            birth = datetime.fromisoformat(birth_date + "T08:00:00+00:00")
            # A minor infant episode is documented at base, so its onset must
            # remain inside the actual life/episode interval.
            onset = max(onset, min(base, birth + timedelta(days=30)))
        condition["onsetDateTime"] = onset.isoformat().replace("+00:00", "Z")
        if archetype in {"generally-healthy", "pediatric"}:
            end = base + timedelta(days=7)
            anchor = datetime.fromisoformat(spec["timelineAnchor"].replace("Z", "+00:00"))
            if end > anchor:
                condition["clinicalStatus"]["coding"][0]["code"] = "active"
                condition.pop("abatementDateTime", None)
    if spec.get("remediation") and pathway:
        template = next(row for row in load_versioned_catalog(spec["remediation"]["pathwayTemplateCatalog"])["archetypes"][archetype] if row["id"] == pathway["pathwayVariant"])
        diagnosis = next((row for row in template["events"] if row["role"] == "condition"), None)
        if diagnosis:
            condition["code"] = {"coding": [{key: diagnosis[key] for key in ("system", "code", "display")}]}
    condition_problem_ids = list(pids)
    condition_problem_kinds = list(pkinds)
    pids, pkinds = _problem_for_resource(problems, org_id, "Observation")
    if source_model:
        distribution_catalog = load_versioned_catalog(spec["clinicalDistributions"]["catalog"])["guardrails"]
        observation_model = source_model["observation"]
        guard = distribution_catalog[observation_model["code"]]
        value = round(guard["ordinaryMin"] + ((index * 7919 + len(org_id) * 97) % 1000) / 999 * (guard["ordinaryMax"] - guard["ordinaryMin"]), 2)
        trajectory_class = "legacy-single-result"
        if spec.get("clinicalRealism"):
            modeled = quantitative_result(
                spec, index, org_id, archetype, observation_model["code"],
                guard["referenceLow"], guard["referenceHigh"], guard["abnormalLow"], guard["abnormalHigh"],
                birth_date, gender,
                secondary_trait=pathway["secondaryTrait"] if pathway else "none",
                pathway_variant=pathway["pathwayVariant"] if pathway else "",
                occurrence_date=base.isoformat(),
            )
            value = modeled["value"]
            interpretation_code = modeled["interpretation"]
            trajectory_class = modeled["trajectoryClass"]
        else:
            abnormal = (index * 43 + {"hmo": 7, "hospital-a": 19, "hospital-b": 31}[org_id]) % 100 < 8
            if abnormal:
                prefer_high = archetype in {"chronic-care", "medication-intensive"} or index % 2 == 0
                value = guard["abnormalHigh"] if prefer_high else guard["abnormalLow"]
            interpretation_code = "H" if value > guard["referenceHigh"] else ("L" if value < guard["referenceLow"] else "N")
    else:
        observation_model = {"system": "http://loinc.org", "code": "6298-4", "display": "Potassium [Moles/volume] in Blood", "unit": "mmol/L", "low": 3.5, "high": 5.1}
        value = 4.5 + (index % 30) / 10
        interpretation_code = "N"
    cross_problem = next((problem for problem in problems if problem.get("scenario")), None)
    scenario = cross_problem["scenario"] if cross_problem else "none"
    variant_index = (
        SCENARIO_VARIANTS[scenario].index(cross_problem["variant"])
        if cross_problem and spec.get("clinicalRealism")
        else (((int(cross_problem["problemId"].split("-")[-1]) - 1) // 3) % 3 if cross_problem else 0)
    )
    observation_base = base
    issued = observation_base + timedelta(hours=2, minutes=source_model["documentationLagMinutes"] if source_model else 60)
    if "timestamp-order" in pkinds:
        timestamp_variant = _problem_variant(problems, org_id, "Observation", "timestamp-order")
        if timestamp_variant == "future-result":
            observation_base = datetime.fromisoformat(spec["timelineAnchor"].replace("Z", "+00:00")) + timedelta(days=30 + index % 30)
        elif timestamp_variant == "pre-birth-event" and birth_date:
            observation_base = datetime.fromisoformat(birth_date + "T08:00:00+00:00") - timedelta(days=7)
        issued = observation_base + timedelta(hours=3) if timestamp_variant in {"future-result", "pre-birth-event"} else observation_base - timedelta(hours=1 + index % 5)
    observation = {
        "resourceType": "Observation",
        "meta": meta(spec, org_id, PROFILES["Observation"], "laboratory-result", confidentiality=confidentiality,
                     sensitivity=sensitivity, nopat=nopat, problem_ids=pids, problem_kinds=pkinds, archetype=archetype),
        "identifier": [strong_identifier(spec, org_id, "Observation", observation_url.removeprefix("urn:uuid:"))],
        "status": "final",
        "category": [{"coding": [{"system": "http://snomed.info/sct", "code": "108252007", "display": "Laboratory procedure"}]}],
        "code": {"coding": [{key: observation_model[key] for key in ("system", "code", "display")}]},
        "subject": {"reference": patient_url, "identifier": {"system": spec["identity"]["cohortIdentifierSystem"], "value": cohort_id(index)}}, "encounter": {"reference": encounter_url},
        "effectiveDateTime": (observation_base + timedelta(hours=2, minutes=30)).isoformat().replace("+00:00", "Z"),
        "issued": issued.isoformat().replace("+00:00", "Z"), "performer": [{"reference": org_url}], "specimen": {"reference": specimen_url},
        "basedOn": [{"type": "ServiceRequest", "identifier": _reference_identifier(spec, org_id, "ServiceRequest", request_url.removeprefix("urn:uuid:"))}],
        "valueQuantity": {"value": value, "unit": observation_model["unit"], "system": "http://unitsofmeasure.org", "code": observation_model["unit"]},
        "referenceRange": [{"low": {"value": observation_model["low"], "unit": observation_model["unit"], "system": "http://unitsofmeasure.org", "code": observation_model["unit"]}, "high": {"value": observation_model["high"], "unit": observation_model["unit"], "system": "http://unitsofmeasure.org", "code": observation_model["unit"]}}],
        "interpretation": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": interpretation_code, "display": {"N": "Normal", "H": "High", "L": "Low"}[interpretation_code]}]}],
        "text": {"status": "generated", "div": f"<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic {observation_model['display']} result: {value} {observation_model['unit']}.</div>"},
    }
    if spec.get("clinicalRealism"):
        observation["meta"]["tag"].append({"system": "urn:hdp-test-bundle:synthetic-trajectory-class", "code": trajectory_class})
    if problems and problems[0]["class"] == "cross-source-clinical" and org_id == "hospital-a":
        if scenario == "renal-medication-risk":
            observation["code"] = {"coding": [{"system": "http://loinc.org", "code": "6298-4", "display": "Potassium [Moles/volume] in Blood"}]}
            observation["valueQuantity"] = {"value": 5.7, "unit": "mmol/L", "system": "http://unitsofmeasure.org", "code": "mmol/L"}
            observation["referenceRange"] = [{"low": {"value": 3.5, "unit": "mmol/L", "system": "http://unitsofmeasure.org", "code": "mmol/L"}, "high": {"value": 5.1, "unit": "mmol/L", "system": "http://unitsofmeasure.org", "code": "mmol/L"}}]
            observation["interpretation"] = [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "H", "display": "High"}]}]
        elif scenario == "bleeding-risk":
            observation["code"] = {"coding": [{"system": "http://loinc.org", "code": "718-7", "display": "Hemoglobin [Mass/volume] in Blood"}]}
            observation["valueQuantity"] = {"value": 11.2, "unit": "g/dL", "system": "http://unitsofmeasure.org", "code": "g/dL"}
            observation["referenceRange"] = [{"low": {"value": 12.0, "unit": "g/dL", "system": "http://unitsofmeasure.org", "code": "g/dL"}, "high": {"value": 16.0, "unit": "g/dL", "system": "http://unitsofmeasure.org", "code": "g/dL"}}]
            observation["interpretation"] = [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "L", "display": "Low"}]}]
        elif scenario == "allergy-treatment-risk":
            observation["code"] = {"coding": [{"system": "http://loinc.org", "code": "630-4", "display": "Bacteria identified in Specimen by Culture"}]}
            observation.pop("valueQuantity", None)
            observation.pop("referenceRange", None)
            observation["valueCodeableConcept"] = {"coding": [{"system": "http://snomed.info/sct", "code": "112283007", "display": "Escherichia coli"}]}
            observation["referenceRange"] = [{
                "low": {"value": 0, "unit": "no growth", "system": "http://unitsofmeasure.org", "code": "1"},
                "high": {"value": 0, "unit": "no growth", "system": "http://unitsofmeasure.org", "code": "1"},
                "text": "No bacterial growth expected in this synthetic specimen context."
            }]
        if variant_index == 0:
            observation["method"] = {"coding": [{"system": "http://snomed.info/sct", "code": "396550006", "display": "Blood test (procedure)"}]}
        elif variant_index == 1:
            observation["note"] = [{"text": "Synthetic repeat/trend context reviewed across source documentation."}]
            observation["bodySite"] = {"coding": [{"system": "http://snomed.info/sct", "code": "368208006", "display": "Left arm"}]}
    if problems and problems[0]["class"] == "cross-source-clinical":
        condition["clinicalStatus"]["coding"][0]["code"] = "active"
        condition.pop("abatementDateTime", None)
        if scenario == "renal-medication-risk" and org_id == "hmo":
            condition["code"] = {"coding": [{"system": "http://snomed.info/sct", "code": "431857002", "display": "Chronic kidney disease stage 4"}]}
        elif scenario == "bleeding-risk" and org_id == "hospital-b":
            bleeding_code = ("74474003", "Gastrointestinal hemorrhage") if spec.get("schemaVersion") == "2.0.0" else ("13200003", "Peptic ulcer")
            condition["code"] = {"coding": [{"system": "http://snomed.info/sct", "code": bleeding_code[0], "display": bleeding_code[1]}]}
            condition["recordedDate"] = ((source_event_time(spec, "hospital-a", index) - timedelta(days=2)).date().isoformat() if spec.get("schemaVersion") == "2.0.0" else base.date().isoformat())
    if "semantically-wrong-code" in condition_problem_kinds:
        wrong_variant = _problem_variant(problems, org_id, "Condition", "semantically-wrong-code")
        wrong = {
            "specimen-as-diagnosis": ("119297000", "Blood specimen"),
            "body-site-as-diagnosis": ("368208006", "Left arm"),
            "procedure-as-diagnosis": ("396550006", "Blood test (procedure)"),
        }.get(wrong_variant, ("119297000", "Blood specimen"))
        condition["code"] = {"coding": [{"system": "http://snomed.info/sct", "code": wrong[0], "display": wrong[1]}]}
    if "discordant-lab" in pkinds:
        discordant_variant = _problem_variant(problems, org_id, "Observation", "discordant-lab")
        delta = {"same-specimen-disagreement": .8, "near-time-method-disagreement": 1.1, "unit-normalized-disagreement": 1.4}.get(discordant_variant, .8)
        observation["valueQuantity"]["value"] = round(float(observation["valueQuantity"]["value"]) + delta, 2)
        observation["interpretation"] = [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "H"}]}]
    if "unit-mismatch" in pkinds:
        unit_variant = _problem_variant(problems, org_id, "Observation", "unit-mismatch")
        if unit_variant == "scale-mismatch":
            observation["valueQuantity"]["value"] = round(float(observation["valueQuantity"]["value"]) * 10, 2)
            # The coded display remains plausible; comparison with the source
            # representation below exposes the missing unit conversion.
        elif unit_variant == "wrong-unit-family":
            observation["valueQuantity"].update({"unit": "kg", "code": "kg"})
        else:
            observation["valueQuantity"]["unit"] = "mmol/L" if observation["valueQuantity"].get("code") == "mg/dL" else "mg/dL"
    specimen = {
        "resourceType": "Specimen",
        "meta": meta(spec, org_id, "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-specimen-lab", "laboratory-result",
                     confidentiality=confidentiality, sensitivity=sensitivity, nopat=nopat),
        "identifier": [strong_identifier(spec, org_id, "Specimen", specimen_url.removeprefix("urn:uuid:"))],
        "status": "available",
        "type": {"coding": [{"system": "http://snomed.info/sct", "code": "119297000", "display": "Blood specimen"}]},
        "subject": {"reference": patient_url},
        "receivedTime": (base + timedelta(hours=2, minutes=15)).isoformat().replace("+00:00", "Z"),
        "request": [{"reference": request_url, "type": "ServiceRequest"}],
        "collection": {"collectedDateTime": (base + timedelta(hours=2)).isoformat().replace("+00:00", "Z"), "bodySite": {"coding": [{"system": "http://snomed.info/sct", "code": "368208006", "display": "Left arm"}]}},
        "text": {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic laboratory specimen.</div>"},
    }
    request_pids, request_pkinds = _problem_for_resource(problems, org_id, "ServiceRequest")
    service_request = {
        "resourceType": "ServiceRequest",
        "meta": meta(spec, org_id, "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-service-request", "referral",
                     confidentiality=confidentiality,
                     problem_ids=request_pids, problem_kinds=request_pkinds),
        "identifier": [strong_identifier(spec, org_id, "ServiceRequest", request_url.removeprefix("urn:uuid:"))],
        "status": "completed", "intent": "order", "priority": "routine",
        "category": [{"coding": [{"system": "http://snomed.info/sct", "code": "108252007", "display": "Laboratory procedure"}]}],
        "code": {"coding": [{key: observation_model[key] for key in ("system", "code", "display")}]},
        "subject": {"reference": patient_url},
        "encounter": {"type": "Encounter", "identifier": _reference_identifier(spec, org_id, "Encounter", encounter_url.removeprefix("urn:uuid:"))},
        "authoredOn": (base + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        "occurrenceDateTime": (base + timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
        "requester": {"reference": practitioner_url}, "performer": [{"reference": org_url}],
        "reasonCode": [{"coding": [{"system": "http://snomed.info/sct", "code": "38341003", "display": "Hypertensive disorder"}]}],
        "reasonReference": [{"reference": condition_url}],
        "supportingInfo": [{"reference": condition_url}],
        "note": [{"text": "Synthetic routine laboratory order."}],
        "text": {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic routine potassium laboratory order.</div>"},
    }
    if problems and problems[0]["class"] == "cross-source-clinical":
        service_request["code"] = copy.deepcopy(observation["code"])
    if request_pids and scenario == "allergy-treatment-risk":
        service_request["code"] = {"coding": [{"system": "http://loinc.org", "code": "630-4", "display": "Bacteria identified in Specimen by Culture"}]}
        service_request["note"] = [{"text": "Synthetic culture order for the reviewed cross-source scenario."}]
        service_request["text"]["div"] = "<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic culture order.</div>"
    if condition_coding_selection is not None:
        if condition_problem_ids:
            raise ValueError("S18-LC Condition coding selection requires an ordinary source record")
        condition = select_condition_coding(condition, condition_coding_selection, spec["metadata"])
    resources = [(encounter_url, encounter), (condition_url, condition), (request_url, service_request), (specimen_url, specimen), (observation_url, observation)]
    if pathway and "captureVariant" in encounter_policy:
        for _url, resource in resources:
            resource["meta"]["tag"].extend([
                {"system": "urn:hdp-test-bundle:synthetic-pathway-variant", "code": pathway["pathwayVariant"]},
                {"system": "urn:hdp-test-bundle:synthetic-capture-variant", "code": encounter_policy["captureVariant"]},
            ])
        if not full_capture:
            capture = set(encounter_policy["captureResourceTypes"])
            if "Observation" in capture and "Specimen" not in capture:
                observation["specimen"] = {
                    "type": "Specimen",
                    "identifier": _reference_identifier(spec, org_id, "Specimen", specimen_url.removeprefix("urn:uuid:")),
                }
            if "Observation" in capture and "ServiceRequest" not in capture:
                observation["basedOn"] = [{
                    "type": "ServiceRequest",
                    "identifier": _reference_identifier(spec, org_id, "ServiceRequest", request_url.removeprefix("urn:uuid:")),
                }]
            if "ServiceRequest" in capture and encounter_policy["captureVariant"] in {"cancelled-order", "cancelled-diagnostic"}:
                service_request["status"] = "revoked"
            resources = [row for row in resources if row[1]["resourceType"] in capture]
        if spec.get("clinicalRealism") and not any(row["class"] == "cross-source-clinical" for row in problems) and archetype in {
            "chronic-care", "medication-intensive", "oncology", "rare-complex",
        } \
                and any(row[1]["resourceType"] == "Observation" for row in resources):
            repeat_url = full_url(version, org_id, cohort_id(index), "Observation", "trajectory-follow-up")
            repeat = copy.deepcopy(observation)
            repeat["meta"]["tag"] = [
                tag for tag in repeat["meta"].get("tag", [])
                if tag.get("system") not in {
                    spec["metadata"]["problemMarkerSystem"], spec["metadata"]["problemIdSystem"],
                    spec["metadata"]["problemKindSystem"],
                }
            ]
            repeat["code"] = {"coding": [{key: observation_model[key] for key in ("system", "code", "display")}]}
            repeat["valueQuantity"] = {
                "value": value, "unit": observation_model["unit"],
                "system": "http://unitsofmeasure.org", "code": observation_model["unit"],
            }
            repeat["identifier"] = [strong_identifier(spec, org_id, "Observation", repeat_url.removeprefix("urn:uuid:"))]
            repeat_time = min(base + timedelta(days=trajectory_followup_days, hours=2, minutes=30), datetime.fromisoformat(spec["timelineAnchor"].replace("Z", "+00:00")) - timedelta(days=1))
            repeat["effectiveDateTime"] = repeat_time.isoformat().replace("+00:00", "Z")
            repeat["issued"] = (repeat_time + timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
            repeat["specimen"] = {"type": "Specimen", "identifier": {
                "system": f"urn:hdp-test-bundle:synthetic:{org_id}:specimen", "value": repeat_url + "-sample"}}
            repeat["basedOn"] = [{"type": "ServiceRequest", "identifier": {
                "system": f"urn:hdp-test-bundle:synthetic:{org_id}:servicerequest", "value": repeat_url + "-order"}}]
            repeat["valueQuantity"]["value"] = min(
                guard["abnormalHigh"],
                max(guard["abnormalLow"], repeated_value(value, interpretation_code, index, trajectory_class)),
            )
            repeat_value = float(repeat["valueQuantity"]["value"])
            repeat_interpretation = "H" if repeat_value > observation_model["high"] else ("L" if repeat_value < observation_model["low"] else "N")
            repeat["interpretation"] = [{"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                "code": repeat_interpretation,
                "display": {"N": "Normal", "H": "High", "L": "Low"}[repeat_interpretation],
            }]}]
            resources.append((repeat_url, repeat))
    if "conflicting-diagnosis" in condition_problem_kinds:
        # Preserve the clean archetype Condition. The intentional problem is a
        # separate reversible overlay and must not erase a required resolved
        # minor condition or make the clean story fail its grammar.
        conflict_one_url = full_url(version, org_id, cohort_id(index), "Condition", "conflicting-type-1")
        conflict_url = full_url(version, org_id, cohort_id(index), "Condition", "conflicting-type-2")
        conflict_one = copy.deepcopy(condition)
        problem_tag_systems = {
            spec["metadata"]["problemMarkerSystem"],
            spec["metadata"]["problemIdSystem"],
            spec["metadata"]["problemKindSystem"],
        }
        condition["meta"]["tag"] = [
            tag for tag in condition["meta"].get("tag", [])
            if tag.get("system") not in problem_tag_systems
        ]
        conflict_one["identifier"] = [strong_identifier(spec, org_id, "Condition", conflict_one_url.removeprefix("urn:uuid:"))]
        conflict_one["clinicalStatus"]["coding"][0]["code"] = "active"
        conflict_one.pop("abatementDateTime", None)
        conflict_one["code"] = {"coding": [{"system": "http://snomed.info/sct", "code": "46635009", "display": "Type 1 diabetes mellitus"}]}
        conflict = copy.deepcopy(conflict_one)
        conflict["identifier"] = [strong_identifier(spec, org_id, "Condition", conflict_url.removeprefix("urn:uuid:"))]
        conflict["code"] = {"coding": [{"system": "http://snomed.info/sct", "code": "44054006", "display": "Type 2 diabetes mellitus"}]}
        conflict["recordedDate"] = (base + timedelta(days=1)).date().isoformat()
        conflict_variant = _problem_variant(problems, org_id, "Condition", "conflicting-diagnosis")
        if spec.get("remediation") and conflict_variant == "code-family-conflict" and not any(row.get("code") in {"46635009", "44054006"} for row in condition["code"]["coding"]):
            raise ValueError("code-family conflict requires an existing diabetes context")
        if conflict_variant in {"status-lifecycle-conflict", "onset-provenance-conflict"}:
            assertion_code = condition["code"]
            if conflict_variant == "status-lifecycle-conflict" and any(
                row.get("code") in {"34014006", "195662009", "74400008", "25374005"}
                for row in assertion_code.get("coding", [])
            ):
                # Keep the clean acute story and a separate chronic assertion
                # pair whose intentional contradiction survives acute snapshots.
                assertion_code = conflict_one["code"]
            conflict_one["code"] = copy.deepcopy(assertion_code)
            conflict["code"] = copy.deepcopy(assertion_code)
            assertion = {"system": "urn:hdp-test-bundle:source-diagnosis-assertion", "value": conflict_one_url.removeprefix("urn:uuid:")}
            conflict_one["identifier"].append(assertion)
            conflict["identifier"].append(copy.deepcopy(assertion))
            conflict["recordedDate"] = conflict_one["recordedDate"]
        if conflict_variant == "status-lifecycle-conflict":
            conflict["clinicalStatus"]["coding"][0]["code"] = "resolved"
            onset = datetime.fromisoformat(conflict["onsetDateTime"].replace("Z", "+00:00"))
            end = max(base - timedelta(days=1), onset + timedelta(days=1))
            anchor = datetime.fromisoformat(spec["timelineAnchor"].replace("Z", "+00:00"))
            if end > anchor:
                raise ValueError("status-conflict resolved interval cannot fit the snapshot")
            conflict["abatementDateTime"] = end.isoformat().replace("+00:00", "Z")
        elif conflict_variant == "onset-provenance-conflict":
            earliest = (
                datetime.fromisoformat(birth_date + "T08:00:00+00:00") + timedelta(days=1)
                if birth_date else datetime.min.replace(tzinfo=base.tzinfo)
            )
            conflict_one["onsetDateTime"] = max(
                earliest, base - timedelta(days=365),
            ).isoformat().replace("+00:00", "Z")
            conflict["onsetDateTime"] = max(
                earliest + timedelta(days=1), base - timedelta(days=120),
            ).isoformat().replace("+00:00", "Z")
        resources.append((conflict_one_url, conflict_one))
        resources.append((conflict_url, conflict))
    if "discordant-lab" in pkinds:
        paired_url = full_url(version, org_id, cohort_id(index), "Observation", "discordant-confirmation")
        paired = copy.deepcopy(observation)
        paired["identifier"] = [strong_identifier(spec, org_id, "Observation", paired_url.removeprefix("urn:uuid:"))]
        discordant_variant = _problem_variant(problems, org_id, "Observation", "discordant-lab")
        paired["valueQuantity"]["value"] = round(float(paired["valueQuantity"]["value"]) - {"same-specimen-disagreement": 1.0, "near-time-method-disagreement": 1.3, "unit-normalized-disagreement": 1.6}.get(discordant_variant, 1.0), 2)
        if discordant_variant == "near-time-method-disagreement":
            paired["method"] = {"coding": [{"system": "http://snomed.info/sct", "code": "396550006", "display": "Blood test (procedure)"}]}
            paired["effectiveDateTime"] = (base + timedelta(hours=5)).isoformat().replace("+00:00", "Z")
            paired["issued"] = (base + timedelta(hours=6)).isoformat().replace("+00:00", "Z")
        elif discordant_variant == "unit-normalized-disagreement":
            loinc = next(
                coding.get("code") for coding in paired.get("code", {}).get("coding", [])
                if coding.get("system") == "http://loinc.org"
            )
            alternate = {
                "6298-4": ("meq/L", 1.0),
                "718-7": ("g/L", 10.0),
                "2160-0": ("umol/L", 88.4),
            }.get(loinc)
            if alternate:
                unit, multiplier = alternate
                paired["valueQuantity"]["value"] = round(
                    float(paired["valueQuantity"]["value"]) * multiplier, 2
                )
                paired["valueQuantity"].update({"unit": unit, "code": unit})
        else:
            paired["issued"] = (issued + timedelta(minutes=20)).isoformat().replace("+00:00", "Z")
        paired["text"]["div"] = "<div xmlns=\"http://www.w3.org/1999/xhtml\">Laboratory result.</div>"
        resources.append((paired_url, paired))
    if "unit-mismatch" in pkinds and _problem_variant(problems, org_id, "Observation", "unit-mismatch") == "scale-mismatch":
        comparator_url = full_url(version, org_id, cohort_id(index), "Observation", "source-unit-representation")
        comparator = copy.deepcopy(observation)
        loinc = observation_model["code"]
        alternate, multiplier = {"6298-4": ("meq/L", 1.0), "718-7": ("g/L", 10.0), "2160-0": ("umol/L", 88.4)}[loinc]
        comparator["identifier"] = [strong_identifier(spec, org_id, "Observation", comparator_url.removeprefix("urn:uuid:"))]
        comparator["valueQuantity"].update(value=round(value * multiplier, 2), unit=alternate, code=alternate)
        observation["derivedFrom"] = [{"reference": comparator_url}]
        resources.append((comparator_url, comparator))
    if problems and problems[0]["class"] == "cross-source-clinical":
        target_type = {"renal-medication-risk": {"hospital-b": "MedicationRequest"}, "bleeding-risk": {"hmo": "MedicationRequest"}, "allergy-treatment-risk": {"hmo": "AllergyIntolerance", "hospital-b": "MedicationRequest"}}.get(scenario, {}).get(org_id)
        if target_type == "MedicationRequest":
            medication_url = full_url(version, org_id, cohort_id(index), "MedicationRequest", scenario)
            pids, pkinds = _problem_for_resource(problems, org_id, "MedicationRequest")
            medication_codes = {
                "renal-medication-risk": ("9997", "Spironolactone"),
                "bleeding-risk": ("11289", "Warfarin"),
                "allergy-treatment-risk": ("723", "Amoxicillin"),
            }
            code, display = medication_codes[scenario]
            medication = {
                "resourceType": "MedicationRequest", "meta": meta(spec, org_id, "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-medication-request", "medication",
                    confidentiality=confidentiality, sensitivity=sensitivity, nopat=nopat, problem_ids=pids, problem_kinds=pkinds),
                "identifier": [strong_identifier(spec, org_id, "MedicationRequest", medication_url.removeprefix("urn:uuid:"))],
                "status": "active", "intent": "order",
                "category": [{"coding": [{"system": "http://fhir.health.gov.il/cs/il-core-medication-request-category", "code": "community-hmo" if org_id == "hmo" else "inpatient", "display": "Community Care/HMO" if org_id == "hmo" else "Inpatient"}]}],
                "medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": code, "display": display}], "text": {"renal-medication-risk": "Spironolactone 25 mg oral tablet", "bleeding-risk": "Warfarin 5 mg oral tablet", "allergy-treatment-risk": "Amoxicillin 500 mg oral capsule"}[scenario]},
                "subject": {"reference": patient_url}, "encounter": {"reference": encounter_url},
                "authoredOn": (
                    (source_event_time(spec, "hospital-a", index) - timedelta(days=90)).date().isoformat()
                    if scenario == "renal-medication-risk" and (
                        spec.get("schemaVersion") == "2.0.0" or spec.get("remediation")
                    )
                    else (base - timedelta(days=90)).date().isoformat()
                    if scenario == "renal-medication-risk"
                    else base.date().isoformat()
                ), "requester": {"reference": practitioner_url},
                "dosageInstruction": [{
                    "text": {"renal-medication-risk": "25 mg once daily", "bleeding-risk": "5 mg once daily with monitoring", "allergy-treatment-risk": "500 mg three times daily for seven days"}[scenario],
                    "timing": {"repeat": {"frequency": 3 if scenario == "allergy-treatment-risk" else 1, "period": 1, "periodUnit": "d"}},
                    "route": {"coding": [{"system": "http://snomed.info/sct", "code": "26643006", "display": "Oral route"}]},
                    "method": {"coding": [{"system": "http://snomed.info/sct", "code": "738995006", "display": "Swallow (administration method)"}]},
                    "doseAndRate": [{"doseQuantity": {"value": {"renal-medication-risk": 25, "bleeding-risk": 5, "allergy-treatment-risk": 500}[scenario], "unit": "mg", "system": "http://unitsofmeasure.org", "code": "mg"}}],
                }],
                "courseOfTherapyType": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/medicationrequest-course-of-therapy", "code": "acute" if scenario == "allergy-treatment-risk" else "continuous", "display": "Short course (acute) therapy" if scenario == "allergy-treatment-risk" else "Continuous long term therapy"}]},
                "extension": [{"url": "http://fhir.health.gov.il/StructureDefinition/ext-narcotic-medication", "valueBoolean": False}],
                "text": {"status": "generated", "div": f"<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic active {display} order.</div>"},
            }
            if variant_index == 1:
                medication["dispenseRequest"] = {"quantity": {"value": 30 if scenario != "allergy-treatment-risk" else 21, "unit": "tablet"}}
            elif variant_index == 2:
                medication["note"] = [{"text": "Synthetic source documents medication reconciliation and exposure overlap."}]
            resources.append((medication_url, medication))
        elif target_type == "AllergyIntolerance":
            allergy_url = full_url(version, org_id, cohort_id(index), "AllergyIntolerance", scenario)
            pids, pkinds = _problem_for_resource(problems, org_id, "AllergyIntolerance")
            allergy_onset = base - timedelta(days=365)
            if birth_date:
                allergy_onset = max(allergy_onset, datetime.fromisoformat(birth_date + "T08:00:00+00:00") + timedelta(days=30))
            allergy = {
                "resourceType": "AllergyIntolerance", "meta": meta(spec, org_id, "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-allergy-intolerance", "allergy-record",
                    confidentiality=confidentiality, sensitivity=sensitivity, nopat=nopat, problem_ids=pids, problem_kinds=pkinds),
                "identifier": [strong_identifier(spec, org_id, "AllergyIntolerance", allergy_url.removeprefix("urn:uuid:"))],
                "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical", "code": "active"}]},
                "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification", "code": "confirmed"}]},
                "type": "allergy", "category": ["medication"], "criticality": "high",
                "code": {"coding": [{"system": "http://snomed.info/sct", "code": "91936005", "display": "Allergy to penicillin"}]},
                "patient": {"reference": patient_url}, "encounter": {"reference": encounter_url},
                "recordedDate": base.date().isoformat(), "asserter": {"reference": patient_url}, "recorder": {"reference": practitioner_url},
                "onsetDateTime": allergy_onset.date().isoformat(),
                "reaction": [{"substance": {"coding": [{"system": "http://snomed.info/sct", "code": "91936005", "display": "Allergy to penicillin"}]}, "manifestation": [{"coding": [{"system": "http://snomed.info/sct", "code": "39579001", "display": "Anaphylaxis"}]}], "severity": "severe"}],
                "text": {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic confirmed severe penicillin allergy.</div>"},
            }
            resources.append((allergy_url, allergy))
    return resources


def _sensitive_fact_resources(spec: dict, organization: dict, index: int, patient_url: str, birth_date: str) -> list[tuple[str, dict]]:
    if spec.get("schemaVersion") != "2.0.0":
        return []
    fact = fact_for_patient(spec, index)
    owner = owning_source(spec, fact) if fact else None
    if not fact or organization["id"] != owner:
        return []
    org_id = organization["id"]
    event_time = source_event_time(spec, org_id, index)
    born = datetime.fromisoformat(birth_date + "T08:00:00+00:00")
    earliest = born + timedelta(days=30)
    if fact in {"adult-hiv-positive", "adult-hiv-negative"}:
        try:
            earliest = born.replace(year=born.year + 18)
        except ValueError:  # February 29 reaches adulthood on February 28 here.
            earliest = born.replace(year=born.year + 18, day=28)
    event_date = max(event_time, earliest).date().isoformat()
    url = full_url(spec["datasetVersion"], org_id, cohort_id(index), "ClinicalFact", fact)
    if fact in {"adult-hiv-positive", "minor-hiv-positive", "adult-hiv-negative"}:
        role = "hiv-result"
        confidentiality, sensitivity, nopat = classify_sensitivity(spec, fact, role, birth_date, event_date)
        profile = PROFILES["Observation"]
        resource = {
            "resourceType": "Observation",
            "meta": meta(spec, org_id, profile, "laboratory-result", confidentiality=confidentiality, sensitivity=sensitivity, nopat=nopat, archetype=primary_archetype(spec, index)),
            "identifier": [strong_identifier(spec, org_id, "Observation", url.removeprefix("urn:uuid:"))],
            "status": "final",
            "category": [{"coding": [{"system": "http://snomed.info/sct", "code": "108252007", "display": "Laboratory procedure"}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": "68961-2", "display": "HIV 1 Ab [Presence] in Serum or Plasma by Immunoblot"}]},
            "subject": {"reference": patient_url, "identifier": {"system": spec["identity"]["cohortIdentifierSystem"], "value": cohort_id(index)}},
            "effectiveDateTime": event_date,
            "issued": event_date + "T12:00:00Z",
            "performer": [{"reference": full_url(spec["datasetVersion"], org_id, "master", "Organization")}],
            "specimen": {"identifier": {"system": "urn:hdp-test-bundle:synthetic:specimen", "value": f"hiv-specimen-{index:04d}"}},
            "basedOn": [{"identifier": {"system": "urn:hdp-test-bundle:synthetic:order", "value": f"hiv-{index:04d}"}}],
            "valueCodeableConcept": {"coding": [{"system": "http://snomed.info/sct", "code": "260373001" if fact.endswith("positive") else "260385009", "display": "Detected" if fact.endswith("positive") else "Negative"}]},
            "interpretation": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "POS" if fact.endswith("positive") else "NEG"}]}],
            "referenceRange": [{
                "low": {"value": 0, "unit": "qualitative index", "system": "http://unitsofmeasure.org", "code": "1"},
                "high": {"value": 0, "unit": "qualitative index", "system": "http://unitsofmeasure.org", "code": "1"},
                "text": "Expected non-reactive synthetic screening context; qualitative result is authoritative.",
            }],
            "text": {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic HIV laboratory fact for confidentiality-policy testing.</div>"},
        }
    else:
        role = {"mental-health-diagnosis": "mental-health-condition", "protected-domestic-violence-indicator": "violence-flag", "ordinary-injury": "ordinary-injury"}[fact]
        confidentiality, sensitivity, nopat = classify_sensitivity(spec, fact, role, birth_date, event_date)
        code = {
            "mental-health-diagnosis": ("35489007", "Depressive disorder"),
            "protected-domestic-violence-indicator": ("707087005", "At risk of domestic abuse"),
            "ordinary-injury": ("125666000", "Burn"),
        }[fact]
        resource = {
            "resourceType": "Condition",
            "meta": meta(spec, org_id, PROFILES["Condition"], "diagnosis", confidentiality=confidentiality, sensitivity=sensitivity, nopat=nopat, archetype=primary_archetype(spec, index)),
            "identifier": [strong_identifier(spec, org_id, "Condition", url.removeprefix("urn:uuid:"))],
            "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
            "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed"}]},
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-category", "code": "problem-list-item"}]}],
            "code": {"coding": [{"system": "http://snomed.info/sct", "code": code[0], "display": code[1]}]},
            "subject": {"reference": patient_url, "identifier": {"system": spec["identity"]["cohortIdentifierSystem"], "value": cohort_id(index)}},
            "recordedDate": event_date,
            "recorder": {"reference": full_url(spec["datasetVersion"], org_id, "master", "Practitioner")},
            "text": {"status": "generated", "div": f"<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic {fact} fact.</div>"},
        }
    resource["meta"]["tag"].extend([
        {"system": "urn:hdp-test-bundle:synthetic-sensitivity-fact", "code": fact},
        {"system": "urn:hdp-test-bundle:synthetic-fact-role", "code": role},
    ])
    resources = [(url, resource)]
    if confidentiality:
        support_url = full_url(spec["datasetVersion"], org_id, cohort_id(index), "ClinicalFactSupport", fact)
        support = copy.deepcopy(resource)
        support["identifier"] = [strong_identifier(spec, org_id, support["resourceType"], support_url.removeprefix("urn:uuid:"))]
        support["text"] = {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic related sensitive-fact graph support.</div>"}
        if resource["resourceType"] == "Observation":
            resource["hasMember"] = [{"reference": support_url}]
            support["derivedFrom"] = [{"reference": url}]
        else:
            resource["evidence"] = [{"detail": [{"reference": support_url}]}]
            support["evidence"] = [{"detail": [{"reference": url}]}]
        resources.append((support_url, support))
    return resources


def _secondary_trait_resources(
    spec: dict, organization: dict, index: int, patient_url: str,
    birth_date: str | None = None,
) -> list[tuple[str, dict]]:
    """Materialize the declared secondary trait as ordinary observable FHIR.

    The crosswalk may describe latent truth, but R29 does not permit that label to
    count as clinical richness.  These HMO-owned rows are deliberately outside
    the minimum primary-archetype grammar and are evaluated from code/status and
    medication details after generator tags have been ignored.
    """
    if not spec.get("clinicalRealism") or organization["id"] != "hmo":
        return []
    archetype = primary_archetype(spec, index)
    pathway = care_pathway(spec, index, archetype)
    trait = pathway["secondaryTrait"]
    if trait == "none":
        return []
    if trait == "medication-exposure" and spec.get("remediation"):
        # The v4 typed pathway already emits the hypertension condition and
        # active lisinopril authorization. Do not append the old unrelated
        # short-course amoxicillin statement to v4 problem/control overlays.
        return []
    org_id = organization["id"]
    version = spec["datasetVersion"]
    base = source_event_time(spec, org_id, index)
    practitioner_url = full_url(version, org_id, "master", "Practitioner")
    if trait in {"metabolic-risk", "respiratory-history"}:
        code, display = {
            "metabolic-risk": ("237602007", "Metabolic syndrome"),
            "respiratory-history": ("195967001", "Asthma"),
        }[trait]
        url = full_url(version, org_id, cohort_id(index), "Condition", f"secondary-{trait}")
        onset = base - timedelta(days=730 + index % 365)
        recorded = base - timedelta(days=120 + index % 90)
        if birth_date:
            born = datetime.fromisoformat(birth_date + "T08:00:00+00:00")
            onset = max(onset, born + timedelta(days=30))
            recorded = max(recorded, onset)
        resource = {
            "resourceType": "Condition",
            "meta": meta(spec, org_id, PROFILES["Condition"], "diagnosis", archetype=archetype),
            "identifier": [strong_identifier(spec, org_id, "Condition", url.removeprefix("urn:uuid:"))],
            "clinicalStatus": {"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active",
            }]},
            "verificationStatus": {"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed",
            }]},
            "category": [{"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-category", "code": "problem-list-item",
            }]}],
            "code": {"coding": [{"system": "http://snomed.info/sct", "code": code, "display": display}]},
            "subject": {
                "reference": patient_url,
                "identifier": {"system": spec["identity"]["cohortIdentifierSystem"], "value": cohort_id(index)},
            },
            "onsetDateTime": onset.date().isoformat(),
            "recordedDate": recorded.date().isoformat(),
            "recorder": {"reference": practitioner_url},
            "text": {"status": "generated", "div": f"<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic clinically compatible {display.lower()} secondary condition.</div>"},
        }
        return [(url, resource)]
    if trait == "medication-exposure":
        url = full_url(version, org_id, cohort_id(index), "MedicationStatement", "secondary-medication-exposure")
        start_instant = base - timedelta(days=180 + index % 60)
        if birth_date:
            born = datetime.fromisoformat(birth_date + "T08:00:00+00:00")
            start_instant = max(start_instant, born + timedelta(days=30))
        start = start_instant.date()
        end = start + timedelta(days=7)
        resource = {
            "resourceType": "MedicationStatement",
            "meta": meta(
                spec, org_id,
                "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-medication-statement",
                "medication", archetype=archetype,
            ),
            "identifier": [strong_identifier(spec, org_id, "MedicationStatement", url.removeprefix("urn:uuid:"))],
            "status": "completed",
            "medicationCodeableConcept": {"coding": [{
                "system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "723", "display": "Amoxicillin",
            }], "text": "Amoxicillin 500 mg oral capsule"},
            "subject": {"reference": patient_url},
            "effectivePeriod": {"start": start.isoformat(), "end": end.isoformat()},
            "dateAsserted": (end + timedelta(days=1)).isoformat(),
            "informationSource": {"reference": practitioner_url},
            "dosage": [{
                "text": "500 mg three times daily for seven days",
                "timing": {"repeat": {"frequency": 3, "period": 1, "periodUnit": "d"}},
                "route": {"coding": [{
                    "system": "http://snomed.info/sct", "code": "26643006", "display": "Oral route",
                }]},
                "method": {"coding": [{
                    "system": "http://snomed.info/sct", "code": "738995006", "display": "Swallow (administration method)",
                }]},
                "doseAndRate": [{"doseQuantity": {
                    "value": 500, "unit": "mg", "system": "http://unitsofmeasure.org", "code": "mg",
                }}],
            }],
            "extension": [{
                "url": "http://fhir.health.gov.il/StructureDefinition/ext-medication-course-of-therapy-type",
                "valueCodeableConcept": {"coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/medicationrequest-course-of-therapy",
                    "code": "acute", "display": "Short course (acute) therapy",
                }]},
            }],
            "text": {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic completed ordinary antibiotic exposure outside the primary archetype grammar.</div>"},
        }
        return [(url, resource)]
    raise ValueError(f"unsupported secondary trait: {trait}")


def _archetype_resources(spec: dict, organization: dict, index: int, patient_url: str, birth_date: str | None = None, existing_medications: set[str] | None = None) -> list[tuple[str, dict]]:
    if spec.get("schemaVersion") != "2.0.0":
        return []
    archetype = primary_archetype(spec, index)
    pathway_variant = (
        care_pathway(spec, index, archetype).get("pathwayVariant")
        if spec.get("clinicalRealism") else None
    )
    org_id = organization["id"]
    base = source_event_time(spec, org_id, index)
    version = spec["datasetVersion"]
    maintenance_code = "29046" if compatible_ingredient("29046", existing_medications or set()) else "17767"
    maintenance_display = "Lisinopril" if maintenance_code == "29046" else "Amlodipine"
    def select_maintenance(resource):
        resource["medicationCodeableConcept"] = {"coding": [{
            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
            "code": maintenance_code, "display": maintenance_display}]}
        resource["dosageInstruction"] = [dosage_for(regimen_catalog(spec)[maintenance_code], start=resource["authoredOn"])]
    if archetype == "chronic-care" and org_id == "hmo":
        url = full_url(version, org_id, cohort_id(index), "MedicationRequest", "chronic-maintenance")
        resource = {
            "resourceType": "MedicationRequest",
            "meta": meta(spec, org_id, "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-medication-request", "medication", archetype=archetype),
            "identifier": [strong_identifier(spec, org_id, "MedicationRequest", url.removeprefix("urn:uuid:"))],
            "status": "active", "intent": "order",
            "category": [{"coding": [{"system": "http://fhir.health.gov.il/cs/il-core-medication-request-category", "code": "community-hmo", "display": "Community Care/HMO"}]}],
            "medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "29046", "display": "Lisinopril"}]},
            "subject": {"reference": patient_url}, "authoredOn": (base + timedelta(days=90 + index % 90)).date().isoformat(),
            "requester": {"reference": full_url(version, org_id, "master", "Practitioner")},
            "dosageInstruction": [{"text": "One 10 mg tablet once daily", "timing": {"repeat": {"frequency": 1, "period": 1, "periodUnit": "d"}}, "route": {"coding": [{"system": "http://snomed.info/sct", "code": "26643006", "display": "Oral route"}]}, "doseAndRate": [{"doseQuantity": {"value": 10, "unit": "mg", "system": "http://unitsofmeasure.org", "code": "mg"}}]}],
            "courseOfTherapyType": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/medicationrequest-course-of-therapy", "code": "continuous", "display": "Continuous long term therapy"}]},
            "extension": [{"url": "http://fhir.health.gov.il/StructureDefinition/ext-narcotic-medication", "valueBoolean": False}],
            "text": {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic chronic-care maintenance prescription and follow-up.</div>"},
        }
        select_maintenance(resource)
        return [(url, resource)]
    if archetype == "pediatric" and org_id == "hmo" and birth_date:
        born = datetime.fromisoformat(birth_date + "T08:00:00+00:00")
        anchor = datetime.fromisoformat(spec["timelineAnchor"].replace("Z", "+00:00"))
        vaccine_events = [(
            "routine-history", born + timedelta(days=60 + (index % 5) * 90),
            {"system": "http://snomed.info/sct", "code": "346313005", "display": "Allergen extract vaccines"},
        )]
        if spec.get("clinicalRealism"):
            catalog = load_versioned_catalog(spec["clinicalRealism"]["catalog"])
            immunization_decision = catalog.get("pathwayReleaseDecisions", {}).get(
                "pediatric-immunization", {}
            )
            anchor_date = anchor.date()
            age = anchor_date.year - born.date().year - ((anchor_date.month, anchor_date.day) < (born.date().month, born.date().day))
            age_band = "infant" if age < 1 else "child" if age < 13 else "adolescent"
            concepts = {row["ageBand"]: {key: row[key] for key in ("system", "code", "display")} for row in catalog["vaccineConcepts"]}
            schedule = {
                "infant": (("birth-dose", 1, concepts["infant"]), ("infant-primary", 60, concepts["adolescent"])),
                "child": (("birth-dose", 1, concepts["infant"]), ("child-routine", 365, concepts["child"])),
                "adolescent": (("child-routine", 365, concepts["child"]), ("adolescent-booster", max(365, age * 365 - 180), concepts["adolescent"])),
            }
            vaccine_events = [(role, born + timedelta(days=offset), concept) for role, offset, concept in schedule[age_band]]
            if immunization_decision.get("generationDefault") == "excluded":
                vaccine_events = []
        resources = []
        for dose_number, (role, planned, vaccine) in enumerate(vaccine_events, start=1):
            occurrence = max(born + timedelta(days=1), min(anchor - timedelta(days=dose_number), planned))
            url = full_url(version, org_id, cohort_id(index), "Immunization", role)
            resource = {
                "resourceType": "Immunization",
                "meta": meta(spec, org_id, "http://fhir.health.gov.il/StructureDefinition/il-core-immunization", "immunization", archetype=archetype),
                "identifier": [strong_identifier(spec, org_id, "Immunization", url.removeprefix("urn:uuid:"))],
                "status": "completed",
                "vaccineCode": {"coding": [vaccine]},
                "patient": {"reference": patient_url}, "occurrenceDateTime": occurrence.date().isoformat(),
                "primarySource": True, "lotNumber": f"SYN-{index:06d}-{dose_number}",
                "site": {"coding": [{"system": "http://snomed.info/sct", "code": "368208006", "display": "Left arm"}]},
                "route": {"coding": [{"system": "http://snomed.info/sct", "code": "78421000", "display": "Intramuscular route (qualifier value)"}]},
                "doseQuantity": {"value": 0.5, "unit": "mL", "system": "http://unitsofmeasure.org", "code": "mL"},
                "performer": [{"actor": {"reference": full_url(version, org_id, "master", "Practitioner")}}],
                "protocolApplied": [{"series": "Synthetic routine pediatric series", "doseNumberPositiveInt": dose_number, "seriesDosesPositiveInt": len(vaccine_events)}],
                "text": {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic age-appropriate pediatric immunization history.</div>"},
            }
            resources.append((url, resource))
        if spec.get("clinicalRealism"):
            vital_url = full_url(version, org_id, cohort_id(index), "Observation", "pediatric-vital-signs")
            vital_time = max(born + timedelta(days=30), min(anchor - timedelta(days=1), base))
            distribution_guards = load_versioned_catalog(
                spec["clinicalDistributions"]["catalog"]
            )["guardrails"]
            bp_models = {}
            for loinc in ("8480-6", "8462-4"):
                guard = distribution_guards[loinc]
                bp_models[loinc] = quantitative_result(
                    spec, index, org_id, archetype, loinc,
                    guard["referenceLow"], guard["referenceHigh"],
                    guard["abnormalLow"], guard["abnormalHigh"],
                    birth_date, None, pathway_variant=pathway_variant or "", occurrence_date=vital_time.isoformat(),
                )
            systolic = bp_models["8480-6"]["value"]
            diastolic = bp_models["8462-4"]["value"]
            vital = {
                "resourceType": "Observation",
                "meta": meta(spec, org_id, "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-bp", "measurement", archetype=archetype),
                "identifier": [strong_identifier(spec, org_id, "Observation", vital_url.removeprefix("urn:uuid:"))],
                "status": "final",
                "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
                "code": {"coding": [{"system": "http://loinc.org", "code": "85354-9", "display": "Blood pressure panel with all children optional"}]},
                "subject": {"reference": patient_url},
                "effectiveDateTime": vital_time.isoformat().replace("+00:00", "Z"),
                "performer": [{"reference": full_url(version, org_id, "master", "Practitioner")}],
                "component": [
                    {
                        "code": {"coding": [{"system": "http://loinc.org", "code": "8480-6", "display": "Systolic blood pressure"}]},
                        "valueQuantity": {"value": systolic, "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"},
                        "referenceRange": [{"low": {"value": 90, "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"}, "high": {"value": 140, "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"}}],
                        "interpretation": [{"coding": [{
                            "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                            "code": bp_models["8480-6"]["interpretation"],
                            "display": {"N": "Normal", "H": "High", "L": "Low"}[bp_models["8480-6"]["interpretation"]],
                        }]}],
                    },
                    {
                        "code": {"coding": [{"system": "http://loinc.org", "code": "8462-4", "display": "Diastolic blood pressure"}]},
                        "valueQuantity": {"value": diastolic, "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"},
                        "referenceRange": [{"low": {"value": 60, "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"}, "high": {"value": 90, "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"}}],
                        "interpretation": [{"coding": [{
                            "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                            "code": bp_models["8462-4"]["interpretation"],
                            "display": {"N": "Normal", "H": "High", "L": "Low"}[bp_models["8462-4"]["interpretation"]],
                        }]}],
                    },
                ],
                "text": {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic age-appropriate pediatric vital-sign assessment.</div>"},
            }
            resources.append((vital_url, vital))
        return resources
    if archetype == "medication-intensive" and org_id == "hmo":
        url = full_url(version, org_id, cohort_id(index), "MedicationRequest", "long-term")
        resource = {
            "resourceType": "MedicationRequest",
            "meta": meta(spec, org_id, "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-medication-request", "medication", archetype=archetype),
            "identifier": [strong_identifier(spec, org_id, "MedicationRequest", url.removeprefix("urn:uuid:"))],
            "status": "active", "intent": "order",
            "category": [{"coding": [{"system": "http://fhir.health.gov.il/cs/il-core-medication-request-category", "code": "community-hmo", "display": "Community Care/HMO"}]}],
            "medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "161", "display": "Acetaminophen"}]},
            "subject": {"reference": patient_url}, "authoredOn": base.date().isoformat(),
            "requester": {"reference": full_url(version, org_id, "master", "Practitioner")},
            "dosageInstruction": [{"text": "One tablet twice daily", "timing": {"repeat": {"frequency": 2, "period": 1, "periodUnit": "d"}}, "route": {"coding": [{"system": "http://snomed.info/sct", "code": "26643006", "display": "Oral route"}]}, "doseAndRate": [{"doseQuantity": {"value": 1, "unit": "tablet", "system": "http://unitsofmeasure.org", "code": "1"}}]}],
            "courseOfTherapyType": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/medicationrequest-course-of-therapy", "code": "continuous", "display": "Continuous long term therapy"}]},
            "extension": [{"url": "http://fhir.health.gov.il/StructureDefinition/ext-narcotic-medication", "valueBoolean": False}],
            "text": {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic longitudinal community prescription.</div>"},
        }
        resource["dosageInstruction"] = [dosage_for({**regimen_catalog(spec)["161"], "frequencyPerDay": 2}, start=resource["authoredOn"])]
        resource["dispenseRequest"] = {"quantity": {"value": 30, "unit": "tablet", "system": "http://snomed.info/sct", "code": "428673006"}, "numberOfRepeatsAllowed": 0}
        statement_url = full_url(version, org_id, cohort_id(index), "MedicationStatement", "follow-up")
        statement = {
            "resourceType": "MedicationStatement",
            "meta": meta(spec, org_id, "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-medication-statement", "medication", archetype=archetype),
            "identifier": [strong_identifier(spec, org_id, "MedicationStatement", statement_url.removeprefix("urn:uuid:"))],
            "status": "active",
            "medicationCodeableConcept": copy.deepcopy(resource["medicationCodeableConcept"]),
            "subject": {"reference": patient_url},
            "effectivePeriod": {"start": (base + timedelta(days=14)).date().isoformat()},
            "dateAsserted": (base + timedelta(days=30)).date().isoformat(),
            "derivedFrom": [{"reference": url}],
            "category": {"coding": [{"system": "http://fhir.health.gov.il/cs/il-core-medication-request-category", "code": "community-other", "display": "Community Care/Long Term Care/Home"}]},
            "dosage": [{
                "text": "Synthetic follow-up confirms one tablet twice daily",
                "timing": {"repeat": {"frequency": 2, "period": 1, "periodUnit": "d"}},
                "route": {"coding": [{"system": "http://snomed.info/sct", "code": "26643006", "display": "Oral route"}]},
                "doseAndRate": [{"doseQuantity": {"value": 1, "unit": "tablet", "system": "http://unitsofmeasure.org", "code": "1"}}],
            }],
            "extension": [{"url": "http://fhir.health.gov.il/StructureDefinition/ext-medication-course-of-therapy-type", "valueCodeableConcept": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/medicationrequest-course-of-therapy", "code": "continuous", "display": "Continuous long term therapy"}]}}],
            "text": {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic medication follow-up statement.</div>"},
        }
        statement["dosage"] = copy.deepcopy(resource["dosageInstruction"])
        if spec.get("clinicalRealism") and pathway_variant == "temporary-stop":
            statement["status"] = "stopped"
            statement["effectivePeriod"]["end"] = (base + timedelta(days=75)).date().isoformat()
            statement["dateAsserted"] = (base + timedelta(days=76)).date().isoformat()
        elif spec.get("clinicalRealism") and pathway_variant == "dose-change":
            statement["dosage"][0]["text"] = "Synthetic follow-up confirms 500 mg once daily"
            statement["dosage"][0]["timing"]["repeat"]["frequency"] = 1
        rows = [(url, resource), (statement_url, statement)]
        if spec.get("clinicalRealism") and pathway_variant == "dispense-follow-up":
            dispense_url = full_url(version, org_id, cohort_id(index), "MedicationDispense", "follow-up")
            dispense = {
                "resourceType": "MedicationDispense",
                "meta": meta(
                    spec, org_id,
                    "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-medication-dispense",
                    "medication", archetype=archetype,
                ),
                "identifier": [strong_identifier(spec, org_id, "MedicationDispense", dispense_url.removeprefix("urn:uuid:"))],
                "status": "completed",
                "medicationCodeableConcept": copy.deepcopy(resource["medicationCodeableConcept"]),
                "subject": {"reference": patient_url},
                "performer": [{"actor": {"reference": full_url(version, org_id, "master", "Practitioner")}}],
                "authorizingPrescription": [{"reference": url}],
                "type": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "FFC", "display": "First Fill - Complete"}]},
                "quantity": {"value": 30, "unit": "tablet", "system": "http://snomed.info/sct", "code": "428673006"},
                "daysSupply": {"value": 15, "unit": "day", "system": "http://unitsofmeasure.org", "code": "d"},
                "whenHandedOver": (base + timedelta(days=7)).isoformat().replace("+00:00", "Z"),
                "dosageInstruction": [copy.deepcopy(resource["dosageInstruction"][0])],
                "text": {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic medication dispense follow-up.</div>"},
            }
            rows.append((dispense_url, dispense))
        if spec.get("clinicalRealism") and pathway_variant == "polypharmacy-review":
            second_url = full_url(version, org_id, cohort_id(index), "MedicationRequest", "concurrent")
            second = copy.deepcopy(resource)
            second["identifier"] = [strong_identifier(spec, org_id, "MedicationRequest", second_url.removeprefix("urn:uuid:"))]
            second["medicationCodeableConcept"] = {
                "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "29046", "display": "Lisinopril"}]
            }
            second["dosageInstruction"][0]["text"] = "One 10 mg tablet once daily as a concurrent reviewed medication"
            second["dosageInstruction"][0]["timing"]["repeat"]["frequency"] = 1
            second["dosageInstruction"][0]["doseAndRate"][0]["doseQuantity"].update({"value": 10, "unit": "mg", "code": "mg"})
            second["text"]["div"] = "<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic concurrent medication reviewed during reconciliation.</div>"
            select_maintenance(second)
            rows.append((second_url, second))
        return rows
    if archetype == "rare-complex" and org_id == "hmo":
        url = full_url(version, org_id, cohort_id(index), "MedicationRequest", "complex-care-maintenance")
        resource = {
            "resourceType": "MedicationRequest",
            "meta": meta(spec, org_id, "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-medication-request", "medication", archetype=archetype),
            "identifier": [strong_identifier(spec, org_id, "MedicationRequest", url.removeprefix("urn:uuid:"))],
            "status": "active", "intent": "order",
            "category": [{"coding": [{"system": "http://fhir.health.gov.il/cs/il-core-medication-request-category", "code": "community-hmo", "display": "Community Care/HMO"}]}],
            "medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "29046", "display": "Lisinopril"}], "text": "Lisinopril 10 mg oral tablet"},
            "subject": {"reference": patient_url}, "authoredOn": base.date().isoformat(),
            "requester": {"reference": full_url(version, org_id, "master", "Practitioner")},
            "dosageInstruction": [{"text": "10 mg once daily as part of synthetic multi-specialty maintenance care", "timing": {"repeat": {"frequency": 1, "period": 1, "periodUnit": "d"}}, "route": {"coding": [{"system": "http://snomed.info/sct", "code": "26643006", "display": "Oral route"}]}, "doseAndRate": [{"doseQuantity": {"value": 10, "unit": "mg", "system": "http://unitsofmeasure.org", "code": "mg"}}]}],
            "courseOfTherapyType": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/medicationrequest-course-of-therapy", "code": "continuous", "display": "Continuous long term therapy"}]},
            "extension": [{"url": "http://fhir.health.gov.il/StructureDefinition/ext-narcotic-medication", "valueBoolean": False}],
            "note": [{"text": "Synthetic rare/complex pathway medication reconciled with specialty follow-up."}],
            "text": {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic rare/complex maintenance medication.</div>"},
        }
        select_maintenance(resource)
        subpaths = (
            ("multi-system-inflammatory", "55464009", "Systemic lupus erythematosus"),
            ("neuromuscular-support", "193093009", "Motor neuron disease"),
            ("complex-renal", "709044004", "Chronic kidney disease"),
            ("rare-metabolic", "75934005", "Metabolic disease"),
        )
        subpath, condition_code, condition_display = (
            next(row for row in subpaths if row[0] == pathway_variant)
            if pathway_variant else subpaths[index % len(subpaths)]
        )
        condition_url = full_url(version, org_id, cohort_id(index), "Condition", subpath)
        condition = {
            "resourceType": "Condition",
            "meta": meta(spec, org_id, PROFILES["Condition"], "diagnosis", archetype=archetype),
            "identifier": [strong_identifier(spec, org_id, "Condition", condition_url.removeprefix("urn:uuid:"))],
            "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
            "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed"}]},
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-category", "code": "problem-list-item"}]}],
            "code": {"coding": [{"system": "http://snomed.info/sct", "code": condition_code, "display": condition_display}]},
            "subject": {"reference": patient_url, "identifier": {"system": spec["identity"]["cohortIdentifierSystem"], "value": cohort_id(index)}},
            "onsetDateTime": (base - timedelta(days=730 + index % 365)).isoformat().replace("+00:00", "Z"),
            "recordedDate": (base - timedelta(days=700 + index % 300)).date().isoformat(),
            "recorder": {"reference": full_url(version, org_id, "master", "Practitioner")},
            "text": {"status": "generated", "div": f"<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic named {subpath} pathway diagnosis.</div>"},
        }
        return [(condition_url, condition), (url, resource)]
    procedure_codes = {
        "womens-health": ("71651007", "Mammography"),
        "perioperative-surgical": ("80146002", "Appendectomy"),
        "oncology": ("367336001", "Chemotherapy"),
    }
    screening_eligible = True
    if archetype == "womens-health":
        if not birth_date:
            screening_eligible = False
        else:
            anchor_date = datetime.fromisoformat(spec["timelineAnchor"].replace("Z", "+00:00")).date()
            born = datetime.fromisoformat(birth_date + "T00:00:00+00:00").date()
            age = anchor_date.year - born.year - ((anchor_date.month, anchor_date.day) < (born.month, born.day))
            screening_eligible = 50 <= age <= 74 and pathway_variant in {
                "age-eligible-screening", "breast-symptom",
            }
    if archetype in procedure_codes and org_id == "hospital-a" and screening_eligible:
        url = full_url(version, org_id, cohort_id(index), "Procedure", archetype)
        code, display = procedure_codes[archetype]
        resource = {
            "resourceType": "Procedure",
            "meta": meta(spec, org_id, "http://fhir.health.gov.il/StructureDefinition/il-core-procedure", "procedure", archetype=archetype),
            "identifier": [strong_identifier(spec, org_id, "Procedure", url.removeprefix("urn:uuid:"))],
            "status": "completed", "category": {"coding": [{"system": "http://snomed.info/sct", "code": "387713003", "display": "Surgical procedure"}]},
            "code": {"coding": [{"system": "http://snomed.info/sct", "code": code, "display": display}]},
            "subject": {"reference": patient_url},
            "encounter": {"reference": full_url(version, org_id, cohort_id(index), "Encounter", "primary")},
            "performedDateTime": (base + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
            "performer": [{"actor": {"reference": full_url(version, org_id, "master", "Practitioner")}}],
            "text": {"status": "generated", "div": f"<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic {display} event.</div>"},
        }
        if archetype == "perioperative-surgical":
            request_url = full_url(version, org_id, cohort_id(index), "ServiceRequest", "surgical-plan")
            encounter_url = full_url(version, org_id, cohort_id(index), "Encounter", "primary")
            condition_url = full_url(version, org_id, cohort_id(index), "Condition", "primary")
            practitioner_url = full_url(version, org_id, "master", "Practitioner")
            organization_url = full_url(version, org_id, "master", "Organization")
            request = {
                "resourceType": "ServiceRequest",
                "meta": meta(spec, org_id, "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-service-request", "referral", archetype=archetype),
                "identifier": [strong_identifier(spec, org_id, "ServiceRequest", request_url.removeprefix("urn:uuid:"))],
                "status": "completed", "intent": "order", "priority": "routine",
                "category": [{"coding": [{"system": "http://snomed.info/sct", "code": "387713003", "display": "Surgical procedure"}]}],
                "code": {"coding": [{"system": "http://snomed.info/sct", "code": code, "display": display}]},
                "subject": {"reference": patient_url},
                "encounter": {"type": "Encounter", "identifier": _reference_identifier(spec, org_id, "Encounter", encounter_url.removeprefix("urn:uuid:"))},
                "authoredOn": (base - timedelta(days=7)).date().isoformat(),
                "occurrenceDateTime": base.date().isoformat(),
                "requester": {"reference": practitioner_url}, "performer": [{"reference": organization_url}],
                "reasonCode": [{"coding": [{"system": "http://snomed.info/sct", "code": "74400008", "display": "Appendicitis"}]}],
                "reasonReference": [{"reference": condition_url}], "supportingInfo": [{"reference": condition_url}],
                "note": [{"text": "Synthetic reviewed pre-operative surgical order."}],
                "text": {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic pre-operative appendectomy order.</div>"},
            }
            resource["basedOn"] = [{"reference": request_url}]
            resource["followUp"] = [{"text": "Synthetic post-operative review planned within 14 days."}]
            followup_url = full_url(version, org_id, cohort_id(index), "Encounter", "post-operative-follow-up")
            followup_start = base + timedelta(days=10 + index % 5)
            followup_code = "185389009" if spec.get("remediation") else "183755002"
            followup_display = "Follow-up visit" if spec.get("remediation") else "Postoperative follow-up visit"
            followup = {
                "resourceType": "Encounter",
                "meta": meta(spec, org_id, "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-encounter-supplemental", "encounter", archetype=archetype),
                "identifier": [strong_identifier(spec, org_id, "Encounter", followup_url.removeprefix("urn:uuid:"))],
                "status": "finished",
                "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB"},
                "type": [{"coding": [{"system": "http://fhir.health.gov.il/cs/il-core-encounter-type", "code": "supplemental-encounter-hdp"}]}],
                "subject": {"reference": patient_url},
                "participant": [{"individual": {"reference": practitioner_url}}],
                "serviceProvider": {"reference": organization_url},
                "period": {"start": followup_start.isoformat().replace("+00:00", "Z"), "end": (followup_start + timedelta(hours=1)).isoformat().replace("+00:00", "Z")},
                "reasonCode": [{"coding": [{"system": "http://snomed.info/sct", "code": followup_code, "display": followup_display}]}],
                "serviceType": {"coding": [{"system": "http://snomed.info/sct", "code": "700232004", "display": "General medical service"}]},
                "text": {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic actual postoperative follow-up encounter.</div>"},
            }
            return [(request_url, request), (url, resource), (followup_url, followup)]
        if archetype == "oncology":
            biopsy_url = full_url(version, org_id, cohort_id(index), "Procedure", "diagnostic-confirmation")
            biopsy = copy.deepcopy(resource)
            biopsy["identifier"] = [strong_identifier(spec, org_id, "Procedure", biopsy_url.removeprefix("urn:uuid:"))]
            biopsy["code"] = {"coding": [{"system": "http://snomed.info/sct", "code": "86273004", "display": "Biopsy"}]}
            biopsy["performedDateTime"] = (base - timedelta(days=30 + index % 20)).isoformat().replace("+00:00", "Z")
            biopsy["text"]["div"] = "<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic diagnostic biopsy confirming the oncology pathway.</div>"
            surveillance_url = full_url(version, org_id, cohort_id(index), "Encounter", "oncology-surveillance")
            surveillance_start = base + timedelta(days=90 + index % 30)
            surveillance = {
                "resourceType": "Encounter",
                "meta": meta(spec, org_id, "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-encounter-supplemental", "encounter", archetype=archetype),
                "identifier": [strong_identifier(spec, org_id, "Encounter", surveillance_url.removeprefix("urn:uuid:"))],
                "status": "finished",
                "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB"},
                "type": [{"coding": [{"system": "http://fhir.health.gov.il/cs/il-core-encounter-type", "code": "supplemental-encounter-hdp"}]}],
                "subject": {"reference": patient_url},
                "participant": [{"individual": {"reference": full_url(version, org_id, "master", "Practitioner")}}],
                "serviceProvider": {"reference": full_url(version, org_id, "master", "Organization")},
                "period": {"start": surveillance_start.isoformat().replace("+00:00", "Z"), "end": (surveillance_start + timedelta(hours=2)).isoformat().replace("+00:00", "Z")},
                "reasonCode": [{"coding": [{"system": "http://snomed.info/sct", "code": "363346000", "display": "Malignant neoplastic disease"}]}],
                "serviceType": {"coding": [{"system": "http://snomed.info/sct", "code": "700232004", "display": "General medical service"}]},
                "text": {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic oncology surveillance follow-up encounter.</div>"},
            }
            return [(biopsy_url, biopsy), (url, resource), (surveillance_url, surveillance)]
        return [(url, resource)]
    return []


def _pediatric_guardian(spec: dict, organization: str, index: int, patient_url: str) -> tuple[str, dict] | None:
    is_pediatric = primary_archetype(spec, index) == "pediatric" if spec.get("schemaVersion") == "2.0.0" else index % 40 == 0
    if not is_pediatric:
        return None
    url = full_url(spec["datasetVersion"], organization, cohort_id(index), "RelatedPerson", "guardian")
    resource = {
        "resourceType": "RelatedPerson",
        "meta": meta(spec, organization, "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-related-person", "patient-demographics"),
        "identifier": [strong_identifier(spec, organization, "RelatedPerson", url.removeprefix("urn:uuid:"))],
        "active": True,
        "patient": {"type": "Patient", "identifier": {"system": spec["identity"]["cohortIdentifierSystem"], "value": cohort_id(index)}},
        "relationship": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-RoleCode", "code": "GUARD", "display": "Guardian"}]}],
        "name": [{"family": "SyntheticGuardian", "given": [f"Guardian{index:04d}"], "use": "official"}],
        "telecom": [{"system": "phone", "value": f"{spec['demographics']['telephonePrefix']}{index % 1000:03d}", "use": "mobile"}],
        "birthDate": "1980-01-01",
        "text": {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic pediatric guardian.</div>"},
    }
    return url, resource


def _scenario_variant_resources(
    spec: dict, organization: str, index: int, patient_url: str, problem: dict,
    clinical: list[tuple[str, dict]], *, control: bool = False, birth_date: str | None = None,
) -> list[str]:
    """Make scenario variants differ in observable clinical topology, not prose."""
    name = problem["scenario"]
    variant = problem["variant"]
    by_type: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for row in clinical:
        by_type[row[1]["resourceType"]].append(row)
    evidence: list[str] = []
    if organization == "hmo" and name == "bleeding-risk":
        for medication_url, medication in by_type.get("MedicationRequest", []):
            if any(
                coding.get("system") == "http://www.nlm.nih.gov/research/umls/rxnorm"
                and coding.get("code") == "11289"
                for coding in medication.get("medicationCodeableConcept", {}).get("coding", [])
            ):
                medication["authoredOn"] = (
                    source_event_time(spec, "hospital-a", index) - timedelta(days=7)
                ).date().isoformat()
                evidence.append(medication_url)
    if organization == "hospital-b" and variant == "culture-informed-treatment":
        treatment = next((
            row for row in by_type.get("MedicationRequest", [])
            if any(coding.get("system") == "http://www.nlm.nih.gov/research/umls/rxnorm" and coding.get("code") == "723"
                   for coding in row[1].get("medicationCodeableConcept", {}).get("coding", []))
        ), None)
        if treatment:
            culture_base = _source_episode_base(spec, "hospital-a", index, birth_date)
            anchor = datetime.fromisoformat(spec["timelineAnchor"].replace("Z", "+00:00"))
            if culture_base >= anchor:
                culture_base = anchor - timedelta(days=4)
            culture_time = culture_base + timedelta(hours=2, minutes=30)
            authored = culture_time - timedelta(days=1) if control else culture_time + timedelta(days=3)
            if control and birth_date:
                documented = max(source_event_time(spec, "hmo", index).date(),
                                 datetime.fromisoformat(birth_date).date() + timedelta(days=1))
                authored = max(authored, datetime.fromisoformat((documented + timedelta(days=1)).isoformat() + "T00:00:00+00:00"))
                if authored >= culture_time:
                    raise ValueError("culture near miss cannot fit strictly after allergy and before culture")
            treatment[1]["authoredOn"] = authored.isoformat().replace("+00:00", "Z")
            original_culture_time = source_event_time(spec, "hospital-a", index) + timedelta(hours=2, minutes=30)
            original_authored = original_culture_time - timedelta(days=1) if control else original_culture_time + timedelta(days=3)
            if authored != original_authored:
                treatment[1]["dosageInstruction"][0]["timing"]["repeat"].setdefault("boundsPeriod", {})["start"] = treatment[1]["authoredOn"]
            evidence.append(treatment[0])
    if organization == "hospital-a" and variant == "culture-informed-treatment":
        evidence.extend(
            url for url, resource in clinical
            if resource.get("resourceType") in {"ServiceRequest", "Observation", "Specimen"}
            and (
                resource.get("resourceType") == "Specimen"
                or any(
                    coding.get("system") == "http://loinc.org" and coding.get("code") == "630-4"
                    for coding in resource.get("code", {}).get("coding", [])
                )
            )
        )
    if organization == "hospital-a" and name == "allergy-treatment-risk" and variant != "culture-informed-treatment":
        background_observation = next((
            row for row in by_type.get("Observation", [])
            if (not spec.get("remediation") or any(c.get("code") in {"LAB", "laboratory", "108252007"}
                for category in row[1].get("category", []) for c in category.get("coding", [])))
            and not any(
                coding.get("system") == "http://loinc.org" and coding.get("code") == "630-4"
                for coding in row[1].get("code", {}).get("coding", [])
            )
        ), None)
        for url, observation in ([background_observation] if background_observation else []):
            observation["code"] = {"coding": [{"system": "http://loinc.org", "code": "718-7", "display": "Hemoglobin [Mass/volume] in Blood"}]}
            observation["valueQuantity"] = {"value": 13.4, "unit": "g/dL", "system": "http://unitsofmeasure.org", "code": "g/dL"}
            observation["referenceRange"] = [{"low": {"value": 12.0, "unit": "g/dL", "system": "http://unitsofmeasure.org", "code": "g/dL"}, "high": {"value": 16.0, "unit": "g/dL", "system": "http://unitsofmeasure.org", "code": "g/dL"}}]
            observation.pop("valueCodeableConcept", None)
            observation["interpretation"] = [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "N", "display": "Normal"}]}]
            evidence.append(url)
    trend_variant = variant in {"chronic-renal-trend", "falling-hemoglobin"}
    if organization == "hospital-a" and trend_variant and by_type.get("Observation"):
        target_loinc = "6298-4" if variant == "chronic-renal-trend" else "718-7"
        candidates = [
            row for row in by_type["Observation"]
            if any(coding.get("system") == "http://loinc.org" and coding.get("code") == target_loinc
                   for coding in row[1].get("code", {}).get("coding", []))
        ]
        if not candidates:
            return evidence
        expected_time = source_event_time(spec, organization, index) + timedelta(hours=2, minutes=30)
        current_url, current = min(candidates, key=lambda row: abs((parse_fhir_time(row[1]["effectiveDateTime"]).start - expected_time).total_seconds()))
        prior_url = full_url(spec["datasetVersion"], organization, cohort_id(index), "Observation", f"scenario-{variant}-prior")
        prior = copy.deepcopy(current)
        prior["identifier"] = [strong_identifier(spec, organization, "Observation", prior_url.removeprefix("urn:uuid:"))]
        prior_time = datetime.fromisoformat(current["effectiveDateTime"].replace("Z", "+00:00")) - timedelta(days=30)
        prior["effectiveDateTime"] = prior_time.isoformat().replace("+00:00", "Z")
        prior["issued"] = (prior_time + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
        # A historical sample has its own unavailable context; it cannot borrow
        # the later episode's specimen/order or encounter.
        for field, kind in (("specimen", "Specimen"), ("encounter", "Encounter")):
            prior[field] = {"type": kind, "identifier": {
                "system": f"urn:hdp-test-bundle:synthetic:{organization}:{kind.lower()}",
                "value": prior_url.removeprefix("urn:uuid:") + "-" + field}}
        prior["basedOn"] = [{"type": "ServiceRequest", "identifier": {
            "system": f"urn:hdp-test-bundle:synthetic:{organization}:servicerequest",
            "value": prior_url.removeprefix("urn:uuid:") + "-order"}}]
        if variant == "chronic-renal-trend":
            prior["valueQuantity"]["value"] = 5.2 if not control else 5.9
        else:
            prior["valueQuantity"]["value"] = 13.1 if not control else 11.3
        clinical.append((prior_url, prior))
        evidence.extend([prior_url, current_url])
    if organization == "hospital-a" and variant == "acute-renal-precipitant" and by_type.get("Condition"):
        base_url, base = by_type["Condition"][0]
        acute_url = full_url(spec["datasetVersion"], organization, cohort_id(index), "Condition", "acute-renal-precipitant")
        acute = copy.deepcopy(base)
        acute["identifier"] = [strong_identifier(spec, organization, "Condition", acute_url.removeprefix("urn:uuid:"))]
        acute["code"] = {"coding": [{"system": "http://snomed.info/sct", "code": "14669001", "display": "Acute kidney injury"}]}
        if spec.get("remediation"):
            precipitant_time = source_event_time(spec, "hospital-a", index) - timedelta(days=2)
            acute["onsetDateTime"] = precipitant_time.isoformat().replace("+00:00", "Z")
            acute["recordedDate"] = precipitant_time.date().isoformat()
        if control:
            acute["clinicalStatus"]["coding"][0]["code"] = "resolved"
            acute["abatementDateTime"] = acute.get("onsetDateTime", acute["recordedDate"])
        clinical.append((acute_url, acute))
        evidence.extend([base_url, acute_url])
        if spec.get("remediation"):
            evidence.extend(
                url for url, observation in by_type.get("Observation", [])
                if any(
                    coding.get("system") == "http://loinc.org" and coding.get("code") == "6298-4"
                    for coding in observation.get("code", {}).get("coding", [])
                )
            )
    if variant in {"medication-dose-escalation", "interacting-exposure"} and by_type.get("MedicationRequest"):
        scenario_rx = "9997" if variant == "medication-dose-escalation" else "11289"
        candidates = [
            row for row in by_type["MedicationRequest"]
            if any(coding.get("system") == "http://www.nlm.nih.gov/research/umls/rxnorm" and coding.get("code") == scenario_rx
                   for coding in row[1].get("medicationCodeableConcept", {}).get("coding", []))
        ]
        if not candidates:
            return evidence
        current_url, current = candidates[-1]
        prior_url = full_url(spec["datasetVersion"], organization, cohort_id(index), "MedicationRequest", f"scenario-{variant}-prior")
        prior = copy.deepcopy(current)
        prior["identifier"] = [strong_identifier(spec, organization, "MedicationRequest", prior_url.removeprefix("urn:uuid:"))]
        prior["status"] = "completed"
        authored = datetime.fromisoformat(current["authoredOn"] + "T00:00:00+00:00") - timedelta(days=120)
        prior["authoredOn"] = authored.date().isoformat()
        if variant == "medication-dose-escalation":
            prior["dosageInstruction"] = [dosage_for(regimen_catalog(spec)["9997"], dose_mg=12.5,
                start=prior["authoredOn"], end=current["authoredOn"])]
            current["priorPrescription"] = {"reference": prior_url}
            if control:
                current["status"] = "stopped"
                current["dosageInstruction"][0]["timing"]["repeat"]["boundsPeriod"] = {
                    "start": current["authoredOn"], "end": current["authoredOn"]}
        else:
            prior["medicationCodeableConcept"] = {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "1191", "display": "Aspirin"}]}
            prior["status"] = "active" if not control else "stopped"
            prior["dosageInstruction"] = [dosage_for(regimen_catalog(spec)["1191"],
                start=prior["authoredOn"], end=current["authoredOn"] if control else None)]
        clinical.append((prior_url, prior))
        evidence.extend([prior_url, current_url])
    if organization == "hospital-b" and variant == "administration-after-allergy" and by_type.get("MedicationRequest"):
        candidates = [
            row for row in by_type["MedicationRequest"]
            if any(coding.get("system") == "http://www.nlm.nih.gov/research/umls/rxnorm" and coding.get("code") == "723"
                   for coding in row[1].get("medicationCodeableConcept", {}).get("coding", []))
        ]
        if not candidates:
            return evidence
        request_url, request = candidates[-1]
        statement_url = full_url(spec["datasetVersion"], organization, cohort_id(index), "MedicationAdministration", "scenario-administration")
        statement = {
            "resourceType": "MedicationAdministration",
            "meta": copy.deepcopy(request["meta"]),
            "identifier": [strong_identifier(spec, organization, "MedicationAdministration", statement_url.removeprefix("urn:uuid:"))],
            "status": "stopped" if control else "completed",
            "medicationCodeableConcept": copy.deepcopy(request["medicationCodeableConcept"]),
            "subject": {"reference": patient_url},
            "effectiveDateTime": request["authoredOn"] + "T12:00:00Z" if len(request["authoredOn"]) == 10 else request["authoredOn"],
            "request": {"reference": request_url},
            "performer": [{"actor": {"reference": full_url(spec["datasetVersion"], organization, "master", "Practitioner")}}],
            "dosage": {
                "text": request["dosageInstruction"][0]["text"],
                "dose": copy.deepcopy(request["dosageInstruction"][0]["doseAndRate"][0]["doseQuantity"]),
                "route": copy.deepcopy(request["dosageInstruction"][0]["route"]),
            },
            "text": {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic documented medication administration/exposure state.</div>"},
        }
        statement["meta"]["profile"] = ["http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-medication-administration"]
        clinical.append((statement_url, statement))
        evidence.extend([request_url, statement_url])
    return evidence


def _align_mutated_laboratory_orders(clinical: list[tuple[str, dict]], evidence_urls: list[str]) -> None:
    """A scenario analyte mutation also changes its own captured order."""
    for url, observation in clinical:
        if url not in evidence_urls or observation.get("resourceType") != "Observation":
            continue
        for reference in observation.get("basedOn", []):
            identity = reference.get("identifier", {})
            for target_url, target in clinical:
                if target.get("resourceType") != "ServiceRequest":
                    continue
                if reference.get("reference") == target_url or (identity.get("value") and any(
                        item.get("system") == identity.get("system") and item.get("value") == identity["value"]
                        for item in target.get("identifier", []))):
                    target["code"] = copy.deepcopy(observation["code"])


def _mark_scenario_evidence(
    clinical: list[tuple[str, dict]], evidence_urls: list[str], role: str,
) -> None:
    """Mark every observable scenario/control row and reconcile lab metadata."""
    _align_mutated_laboratory_orders(clinical, evidence_urls)
    evidence = set(evidence_urls)
    for url, resource in clinical:
        if url not in evidence:
            continue
        tags = resource.setdefault("meta", {}).setdefault("tag", [])
        marker = {"system": "urn:hdp-test-bundle:synthetic-scenario-role", "code": role}
        if marker not in tags:
            tags.append(marker)
        if resource.get("resourceType") != "Observation":
            continue
        quantity = resource.get("valueQuantity", {})
        reference = resource.get("referenceRange", [{}])[0]
        value = quantity.get("value")
        low = reference.get("low", {}).get("value")
        high = reference.get("high", {}).get("value")
        if not all(isinstance(item, (int, float)) for item in (value, low, high)):
            continue
        interpretation = "L" if value < low else "H" if value > high else "N"
        resource["interpretation"] = [{"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
            "code": interpretation,
            "display": {"L": "Low", "H": "High", "N": "Normal"}[interpretation],
        }]}]


def _normalize_coverage_resource(resource: dict, practitioner_reference: str | None = None) -> None:
    """Repair source-map mandatory fields without changing the compact baseline fixture."""
    rtype = resource["resourceType"]
    old = "2015-01-15T10:00:00Z"
    example_systems = {
        "http://acme.org/messagetypes": "urn:hdp-test-bundle:synthetic:communication-type",
        "http://dogs.org/local": "urn:hdp-test-bundle:synthetic:flag-code",
        "http://goodhealthhospital.org/diet-type-codes": "urn:hdp-test-bundle:synthetic:diet-type",
        "http://somehealthcare.org/fhir/CodeSystem/immunization-recommendation-date-criterion": "urn:hdp-test-bundle:synthetic:immunization-date-criterion",
    }

    def replace_example_codings(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("system") in example_systems and isinstance(value.get("code"), str):
                value["system"] = example_systems[value["system"]]
            for child in value.values():
                replace_example_codings(child)
        elif isinstance(value, list):
            for child in value:
                replace_example_codings(child)

    replace_example_codings(resource)
    for _path, coding in iter_codings(resource):
        if coding.get("system") == "urn:hdp-test-bundle:synthetic:diet-type" and coding.get("code") == "1040":
            # The compact source example used a clinically different fluid-
            # restriction label. Coverage fixtures keep one stable diet concept
            # and never treat that non-equivalent phrase as an alias.
            coding["display"] = "Low Sodium Diet"
    resource.setdefault("text", {"status": "generated", "div": f"<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic {rtype} coverage example.</div>"})
    if rtype == "Organization":
        resource.setdefault("partOf", {"identifier": {"system": "urn:hdp-test-bundle:synthetic:parent-organization", "value": "national-test-network"}})
        resource.setdefault("address", [{"line": ["Synthetic Clinical Campus"], "city": "Jerusalem", "country": "IL"}])
    elif rtype == "PractitionerRole":
        resource.setdefault("specialty", [{"coding": [{"system": "http://snomed.info/sct", "code": "408443003", "display": "General medical practice"}]}])
    elif rtype == "MedicationStatement":
        dosage = resource.setdefault("dosage", [{}])[0]
        dosage.setdefault("route", {"coding": [{"system": "http://snomed.info/sct", "code": "26643006", "display": "Oral route"}]})
        dosage.setdefault("doseAndRate", [{"doseQuantity": {"value": 1, "unit": "tablet", "system": "http://unitsofmeasure.org", "code": "1"}}])
        for coding in resource.get("category", {}).get("coding", []):
            if coding.get("system") == "http://fhir.health.gov.il/cs/il-core-medication-request-category" and coding.get("code") == "community-other":
                coding["display"] = "Community Care/Long Term Care/Home"
    elif rtype == "Immunization":
        # Use the exact SNOMED vaccine concept published by the active IL Core
        # package's own Immunization example.
        resource["vaccineCode"] = {"coding": [{
            "system": "http://snomed.info/sct", "code": "871738001",
            "display": "Live attenuated mumps vaccine",
        }]}
    elif rtype == "Location" and any(
        profile.endswith("il-hdp-location-nursing-unit")
        for profile in resource.get("meta", {}).get("profile", [])
    ):
        resource["type"] = [{"coding": [
            {
                "system": "http://fhir.health.gov.il/cs/il-core-location-physical-type",
                "code": "hospital-nursing-unit",
                "display": "Hospital nursing unit",
            },
            {
                "system": "http://terminology.hl7.org/CodeSystem/v3-RoleCode",
                "code": "HU",
                "display": "Hospital unit",
            },
        ]}]
    elif rtype == "MedicationAdministration":
        period = resource.pop("effectivePeriod", {})
        resource.setdefault("effectiveDateTime", period.get("start", old))
    elif rtype == "DocumentReference":
        resource.setdefault("authenticator", {"identifier": {"system": "urn:hdp-test-bundle:synthetic:clinician", "value": "coverage-authenticator"}})
        resource.setdefault("context", {}).setdefault("period", {"start": old, "end": "2015-01-16T10:00:00Z"})
        resource["date"] = old
    elif rtype == "AllergyIntolerance":
        resource.setdefault("onsetDateTime", "2015-01-15")
        resource["code"] = {"coding": [{"system": "http://snomed.info/sct", "code": "227493005", "display": "Cashew Nut"}]}
        resource.setdefault("reaction", [{"manifestation": [{"coding": [{"system": "http://snomed.info/sct", "code": "39579001", "display": "Anaphylaxis"}]}], "severity": "severe"}])
    elif rtype == "Observation" and any(profile.endswith("il-hdp-observation-lab") for profile in resource.get("meta", {}).get("profile", [])):
        resource.setdefault("interpretation", [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "N"}]}])
        resource.setdefault("basedOn", [{"identifier": {"system": "urn:hdp-test-bundle:synthetic:order", "value": "coverage-order"}}])
    elif rtype == "DiagnosticReport":
        resource.setdefault("conclusion", "Synthetic diagnostic conclusion.")
        resource.setdefault("presentedForm", [{"contentType": "text/plain", "data": "U3ludGhldGljIGRpYWdub3N0aWMgcmVwb3J0Lg=="}])
        resource["effectiveDateTime"] = old
    elif rtype == "Encounter":
        reason = [{"type": "Condition", "identifier": {"system": "urn:hdp-test-bundle:synthetic:reason", "value": "coverage-reason"}}]
        resource["reasonReference"] = reason
        resource["diagnosis"] = [{"condition": reason[0]}]
        resource["period"] = {"start": old, "end": "2015-01-15T12:00:00Z"}
    elif rtype == "RelatedPerson":
        resource["patient"] = {"type": "Patient", "identifier": {"system": "urn:hdp-test-bundle:synthetic-cohort-id", "value": "coverage-patient"}}
    elif rtype == "Specimen":
        resource.setdefault("collection", {})["collectedDateTime"] = old
        resource["receivedTime"] = "2015-01-15T10:30:00Z"
    elif rtype == "Communication":
        resource["sent"] = old
    elif rtype == "ImagingStudy":
        resource["started"] = old
    elif rtype == "QuestionnaireResponse":
        resource["authored"] = old
    elif rtype == "MedicationDispense":
        resource["whenHandedOver"] = old
    elif rtype == "DeviceRequest":
        resource.setdefault("occurrenceDateTime", old)
    elif rtype == "SupplyDelivery":
        resource["occurrenceDateTime"] = old


def _coverage_resources(spec: dict, organization: dict, repository_root: Path, patient_urls: list[str], coverage_key: str = "set-00") -> list[tuple[str, dict]]:
    org_id = organization["id"]
    source = read_json(input_path("bundle/certificator-bundle.v1.6.1.transaction.json"))
    entries = source.get("entry", [])
    old_patient_urls = [entry["fullUrl"] for entry in entries if entry.get("resource", {}).get("resourceType") == "Patient"]
    replacements = {old: patient_urls[min(i, len(patient_urls) - 1)] for i, old in enumerate(old_patient_urls)}
    for position, entry in enumerate(entries):
        if entry.get("resource", {}).get("resourceType") == "Patient":
            continue
        replacements[entry["fullUrl"]] = full_url(spec["datasetVersion"], org_id, "coverage", coverage_key, position, entry["resource"]["resourceType"])
    output = []
    for position, entry in enumerate(entries):
        if entry.get("resource", {}).get("resourceType") == "Patient":
            continue
        resource = _walk_replace(copy.deepcopy(entry["resource"]), replacements)
        rtype = resource["resourceType"]
        url = replacements[entry["fullUrl"]]
        old_meta = resource.get("meta", {})
        profile = old_meta.get("profile", [f"http://hl7.org/fhir/StructureDefinition/{rtype}"])[0]
        practitioner_reference = next((value for key, value in replacements.items() if "Practitioner" in key and "PractitionerRole" not in key), None)
        _normalize_coverage_resource(resource, practitioner_reference)
        if spec.get("schemaVersion") == "2.0.0" and profile.endswith("il-hdp-bp"):
            coverage_index = int(coverage_key.rsplit("-", 1)[-1])
            source_shift = {"hmo": 0, "hospital-a": 3, "hospital-b": 6}[org_id]
            bp_values = {
                "8480-6": (110 + ((coverage_index * 7 + source_shift) % 28), 90, 140),
                "8462-4": (68 + ((coverage_index * 5 + source_shift) % 20), 60, 90),
            }
            for component in resource.get("component", []):
                code = next((coding.get("code") for coding in component.get("code", {}).get("coding", []) if coding.get("system") == "http://loinc.org"), None)
                if code not in bp_values:
                    continue
                value, low, high = bp_values[code]
                component["valueQuantity"] = {"value": value, "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"}
                component["referenceRange"] = [{
                    "low": {"value": low, "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"},
                    "high": {"value": high, "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"},
                }]
                component["interpretation"] = [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "N"}]}]
        # Compact-example labels are not copied: v2 derives confidentiality only from
        # a generated fact and the reviewed sensitivity catalog.
        confidentiality = None if spec.get("schemaVersion") == "2.0.0" else next((coding.get("code") for coding in old_meta.get("security", []) if coding.get("system") == spec["metadata"]["confidentialitySystem"] and coding.get("code") in {"R", "V"}), None)
        resource["meta"] = meta(spec, org_id, profile, coverage_role(rtype, profile), confidentiality=confidentiality, archetype="compact-coverage")
        resource.pop("id", None)
        generated_identifier = strong_identifier(spec, org_id, rtype, url.removeprefix("urn:uuid:"))
        existing_identifier = resource.get("identifier")
        if isinstance(existing_identifier, list):
            non_strong = [identifier for identifier in existing_identifier if not any(
                coding.get("system") == "http://fhir.health.gov.il/cs/il-core-identifier-type" and coding.get("code") == "strong-id"
                for coding in identifier.get("type", {}).get("coding", [])
            )]
            resource["identifier"] = [generated_identifier, *non_strong]
        elif isinstance(existing_identifier, dict):
            resource["identifier"] = generated_identifier
        else:
            resource["identifier"] = [generated_identifier]
        output.append((url, resource))
        if rtype == "Specimen" and profile.endswith("il-hdp-specimen-lab"):
            pathology = copy.deepcopy(resource)
            pathology_url = full_url(spec["datasetVersion"], org_id, "coverage", coverage_key, position, "SpecimenPathology")
            pathology_profile = "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-specimen-pathology"
            pathology["meta"] = meta(spec, org_id, pathology_profile, "pathology", archetype="rare-complex")
            pathology["identifier"] = [strong_identifier(spec, org_id, "Specimen", pathology_url.removeprefix("urn:uuid:"))]
            pathology["subject"] = {"type": "Patient", "identifier": {"system": spec["identity"]["cohortIdentifierSystem"], "value": "coverage-pathology-patient"}}
            pathology["request"] = [{"type": "ServiceRequest", "identifier": {"system": "urn:hdp-test-bundle:synthetic:pathology-order", "value": f"pathology-order-{position}"}}]
            output.append((pathology_url, pathology))
    return output


def _control_medication(spec: dict, organization: str, index: int, patient_url: str, name: str, status: str,
                        birth_date: str | None = None) -> tuple[str, dict]:
    url = full_url(spec["datasetVersion"], organization, cohort_id(index), "MedicationRequest", f"control-{name}")
    code, display = {"renal-medication-risk": ("9997", "Spironolactone"), "bleeding-risk": ("11289", "Warfarin"), "allergy-treatment-risk": ("723", "Amoxicillin")}[name]
    base = source_event_time(spec, organization, index) if spec.get("schemaVersion") == "2.0.0" else datetime.fromisoformat(spec["timelineAnchor"].replace("Z", "+00:00")) - timedelta(days=30)
    archetype = primary_archetype(spec, index) if spec.get("schemaVersion") == "2.0.0" else "strict-patient-clinical"
    authored = (base - timedelta(days=90)).date()
    if birth_date and spec["demographics"].get("lifecyclePolicyCatalog"):
        authored = max(authored, datetime.fromisoformat(birth_date + "T00:00:00+00:00").date() + timedelta(days=1))
    return url, {
        "resourceType": "MedicationRequest",
        "meta": meta(spec, organization, "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-medication-request", "medication", archetype=archetype),
        "identifier": [strong_identifier(spec, organization, "MedicationRequest", url.removeprefix("urn:uuid:"))],
        "status": status, "intent": "order",
        "category": [{"coding": [{"system": "http://fhir.health.gov.il/cs/il-core-medication-request-category", "code": "community-hmo" if organization == "hmo" else "inpatient"}]}],
        "medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": code, "display": display}]},
        "subject": {"reference": patient_url}, "authoredOn": authored.isoformat(),
        "requester": {"reference": full_url(spec["datasetVersion"], organization, "master", "Practitioner")},
        "dosageInstruction": [dosage_for(regimen_catalog(spec)[code], start=authored.isoformat(),
            end=base.date().isoformat() if status == "stopped" else None)],
        "courseOfTherapyType": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/medicationrequest-course-of-therapy", "code": "acute", "display": "Short course (acute) therapy"}]},
        "extension": [{"url": "http://fhir.health.gov.il/StructureDefinition/ext-narcotic-medication", "valueBoolean": False}],
        "text": {"status": "generated", "div": f"<div xmlns=\"http://www.w3.org/1999/xhtml\">Medication status: {status}.</div>"},
    }


def _apply_matched_control(spec: dict, index: int, name: str, organization: str, patient_url: str,
                           clinical: list[tuple[str, dict]], variant_index: int,
                           birth_date: str | None = None) -> list[str]:
    # Controls modify the primary captured row, never an appended secondary
    # condition/trajectory. Selecting the last row silently erased independent
    # R29 evidence when new optional branches were added.
    by_type: dict[str, tuple[str, dict]] = {}
    for url, resource in clinical:
        by_type.setdefault(resource["resourceType"], (url, resource))
    if spec.get("remediation"):
        # Controls must mutate a laboratory result, never a growth/BP observation.
        laboratories = [(url, resource) for url, resource in clinical
                        if resource["resourceType"] == "Observation"
                        and any(c.get("code") in {"LAB", "laboratory", "108252007"}
                                for category in resource.get("category", [])
                                for c in category.get("coding", []))]
        expected_time = source_event_time(spec, organization, index) + timedelta(hours=2, minutes=30)
        laboratory = min(laboratories, key=lambda row: abs((parse_fhir_time(row[1]["effectiveDateTime"]).start - expected_time).total_seconds()), default=None)
        if laboratory:
            by_type["Observation"] = laboratory
        elif organization == "hospital-a":
            # A near miss needs its own lab context when this ordinary pathway
            # only has vital signs; do not overwrite a BP or growth measurement.
            url = full_url(spec["datasetVersion"], organization, cohort_id(index), "Observation", f"control-lab-{name}")
            event_time = source_event_time(spec, organization, index).isoformat().replace("+00:00", "Z")
            resource = {
                "resourceType": "Observation", "status": "final",
                "meta": meta(spec, organization, PROFILES["Observation"], "laboratory-result", archetype=primary_archetype(spec, index)),
                "identifier": [strong_identifier(spec, organization, "Observation", url.removeprefix("urn:uuid:"))],
                "category": [{"coding": [{"system": "http://snomed.info/sct", "code": "108252007", "display": "Laboratory procedure"}]}],
                "subject": {"reference": patient_url, "identifier": {"system": spec["identity"]["cohortIdentifierSystem"], "value": cohort_id(index)}},
                "effectiveDateTime": event_time, "issued": event_time,
                "performer": [{"reference": full_url(spec["datasetVersion"], organization, "master", "Practitioner")}],
                "basedOn": [{"type": "ServiceRequest", "identifier": {"system": f"urn:hdp-test-bundle:synthetic:{organization}:servicerequest", "value": f"control-{name}-{index}-request"}}],
                "specimen": {"type": "Specimen", "identifier": {"system": f"urn:hdp-test-bundle:synthetic:{organization}:specimen", "value": f"control-{name}-{index}-specimen"}},
                "interpretation": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "N"}]}],
            }
            clinical.append((url, resource))
            by_type["Observation"] = (url, resource)
    evidence = []
    if name == "renal-medication-risk":
        if organization == "hmo":
            url, resource = by_type["Condition"]
            resource["code"] = {"coding": [{"system": "http://snomed.info/sct", "code": "431857002", "display": "Chronic kidney disease stage 4"}]}
            resource["clinicalStatus"]["coding"][0]["code"] = "active"
            resource.pop("abatementDateTime", None)
            evidence.append(url)
        elif organization == "hospital-a":
            url, resource = by_type["Observation"]
            resource["code"] = {"coding": [{"system": "http://loinc.org", "code": "6298-4", "display": "Potassium [Moles/volume] in Blood"}]}
            resource["valueQuantity"] = {"value": 5.7, "unit": "mmol/L", "system": "http://unitsofmeasure.org", "code": "mmol/L"}
            resource["referenceRange"] = [{"low": {"value": 3.5, "unit": "mmol/L", "system": "http://unitsofmeasure.org", "code": "mmol/L"}, "high": {"value": 5.1, "unit": "mmol/L", "system": "http://unitsofmeasure.org", "code": "mmol/L"}}]
            evidence.append(url)
        elif organization == "hospital-b":
            medication = _control_medication(spec, organization, index, patient_url, name, "active" if spec.get("remediation") and variant_index in {0, 1} else "stopped", birth_date)
            if spec.get("remediation"):
                medication[1]["authoredOn"] = (source_event_time(spec, "hospital-a", index) - timedelta(days=60)).date().isoformat()
            clinical.append(medication)
            evidence.append(medication[0])
    elif name == "bleeding-risk":
        if organization == "hospital-a":
            url, resource = by_type["Observation"]
            resource["code"] = {"coding": [{"system": "http://loinc.org", "code": "718-7", "display": "Hemoglobin [Mass/volume] in Blood"}]}
            resource["valueQuantity"] = {"value": 11.8, "unit": "g/dL", "system": "http://unitsofmeasure.org", "code": "g/dL"}
            resource["referenceRange"] = [{"low": {"value": 12.0, "unit": "g/dL", "system": "http://unitsofmeasure.org", "code": "g/dL"}, "high": {"value": 16.0, "unit": "g/dL", "system": "http://unitsofmeasure.org", "code": "g/dL"}}]
            evidence.append(url)
        elif organization == "hospital-b":
            url, resource = by_type["Condition"]
            resource["code"] = {"coding": [{"system": "http://snomed.info/sct", "code": "74474003", "display": "Gastrointestinal hemorrhage"}]}
            if spec.get("remediation"):
                resource["recordedDate"] = (source_event_time(spec, "hospital-a", index) - timedelta(days=2)).date().isoformat()
                resource["onsetDateTime"] = resource["recordedDate"]
            resource["clinicalStatus"]["coding"][0]["code"] = "active"
            resource.pop("abatementDateTime", None)
            evidence.append(url)
        elif organization == "hmo":
            medication = _control_medication(spec, organization, index, patient_url, name, "active" if spec.get("remediation") and variant_index in {1, 2} else "stopped", birth_date)
            clinical.append(medication)
            evidence.append(medication[0])
    elif name == "allergy-treatment-risk":
        if organization == "hmo":
            url = full_url(spec["datasetVersion"], organization, cohort_id(index), "AllergyIntolerance", "control-non-serious")
            base = source_event_time(spec, organization, index) if spec.get("schemaVersion") == "2.0.0" else datetime.fromisoformat(spec["timelineAnchor"].replace("Z", "+00:00")) - timedelta(days=30)
            archetype = primary_archetype(spec, index) if spec.get("schemaVersion") == "2.0.0" else "strict-patient-clinical"
            onset = (base - timedelta(days=365)).date()
            recorded = base.date()
            if birth_date and spec["demographics"].get("lifecyclePolicyCatalog"):
                born = datetime.fromisoformat(birth_date + "T00:00:00+00:00").date()
                onset = max(onset, born + timedelta(days=1))
                recorded = max(recorded, onset)
            allergy = {
                "resourceType": "AllergyIntolerance", "meta": meta(spec, organization, "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-allergy-intolerance", "allergy-record", archetype=archetype),
                "identifier": [strong_identifier(spec, organization, "AllergyIntolerance", url.removeprefix("urn:uuid:"))],
                "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical", "code": "active"}]},
                "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification", "code": "confirmed"}]},
                "type": "intolerance", "category": ["medication"], "criticality": "low",
                "code": {"coding": [{"system": "http://snomed.info/sct", "code": "91936005", "display": "Allergy to penicillin"}]},
                "patient": {"reference": patient_url}, "recordedDate": recorded.isoformat(), "onsetDateTime": onset.isoformat(),
                "reaction": [{"manifestation": [{"coding": [{"system": "http://snomed.info/sct", "code": "271807003", "display": "Rash"}]}], "severity": "mild"}],
                "asserter": {"reference": patient_url},
                "recorder": {"reference": full_url(spec["datasetVersion"], organization, "master", "Practitioner")},
                "text": {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Mild intolerance documented.</div>"},
            }
            if spec.get("remediation") and variant_index in {1, 2}:
                allergy.update(type="allergy", criticality="high")
                allergy["reaction"][0]["severity"] = "severe"
                allergy["text"]["div"] = '<div xmlns="http://www.w3.org/1999/xhtml">Documented allergy history.</div>'
            clinical.append((url, allergy))
            evidence.append(url)
        elif organization == "hospital-a":
            url, resource = by_type["Observation"]
            resource["code"] = {"coding": [{"system": "http://loinc.org", "code": "630-4", "display": "Bacteria identified in Specimen by Culture"}]}
            resource.pop("valueQuantity", None)
            resource["valueCodeableConcept"] = {"coding": [{"system": "http://snomed.info/sct", "code": "112283007", "display": "Escherichia coli"}]}
            resource["referenceRange"] = [{"low": {"value": 0, "unit": "no growth", "system": "http://unitsofmeasure.org", "code": "1"}, "high": {"value": 0, "unit": "no growth", "system": "http://unitsofmeasure.org", "code": "1"}}]
            evidence.append(url)
        elif organization == "hospital-b":
            medication = _control_medication(spec, organization, index, patient_url, name, "active", birth_date)
            if spec.get("remediation") and variant_index in {1, 2}:
                documented = source_event_time(spec, "hmo", index).date()
                if birth_date:
                    documented = max(documented, datetime.fromisoformat(birth_date).date() + timedelta(days=1))
                authored = max(datetime.fromisoformat(medication[1]["authoredOn"]).date(),
                               documented + timedelta(days=1)).isoformat()
                if authored != medication[1]["authoredOn"]:
                    medication[1]["authoredOn"] = authored
                    medication[1]["dosageInstruction"][0]["timing"]["repeat"].setdefault("boundsPeriod", {})["start"] = authored
            clinical.append(medication)
            evidence.append(medication[0])
    for url, resource in clinical:
        if url in evidence:
            resource["meta"]["tag"].append({"system": "urn:hdp-test-bundle:synthetic-scenario-role", "code": "near-miss-control"})
            if resource["resourceType"] == "Observation":
                if variant_index == 0:
                    resource["method"] = {"coding": [{"system": "http://snomed.info/sct", "code": "396550006", "display": "Blood test (procedure)"}]}
                elif variant_index == 1:
                    resource["note"] = [{"text": "Repeat laboratory result."}]
                    resource["bodySite"] = {"coding": [{"system": "http://snomed.info/sct", "code": "368208006", "display": "Left arm"}]}
            if resource["resourceType"] == "MedicationRequest":
                if variant_index == 1:
                    resource["dispenseRequest"] = {"quantity": {"value": 30, "unit": "tablet"}}
                elif variant_index == 2:
                    resource["note"] = [{"text": "Medication reconciliation."}]
    _align_mutated_laboratory_orders(clinical, evidence)
    return evidence


def control_scenario(spec: dict, ordinal: int) -> tuple[str, int]:
    count = spec["problems"]["crossSourceClinicalPatients"]
    if nested(spec):
        count //= spec["populationSize"] // 1000
        ordinal = (ordinal - 1) % count + 1
    third = max(1, count // 3)
    return scenario_name(ordinal, count), ((ordinal - 1) % third) % 3


def matched_controls(spec: dict) -> list[int]:
    """Select clinically eligible controls within each fixed allocation block."""
    available_controls = [index for index in range(1, spec["populationSize"] + 1) if not allocation(spec, index)]
    control_indexes = []
    if spec.get("clinicalRealism"):
        compatible = {
            "renal-medication-risk": {"chronic-care", "medication-intensive", "oncology", "rare-complex"},
            "bleeding-risk": {"medication-intensive", "perioperative-surgical", "oncology", "rare-complex"},
            "allergy-treatment-risk": {"chronic-care", "pediatric", "medication-intensive", "perioperative-surgical", "oncology", "rare-complex"},
        }
        for position in range(1, spec["problems"]["crossSourceClinicalPatients"] + 1):
            per_block = spec["problems"]["crossSourceClinicalPatients"] // (spec["populationSize"] // 1000) if nested(spec) else spec["problems"]["crossSourceClinicalPatients"]
            block = (position - 1) // per_block if nested(spec) else 0
            name, variant = control_scenario(spec, position)
            selected = next(
                (index for index in available_controls
                 if index not in control_indexes
                 and (not nested(spec) or block_position(index)[0] == block)
                 and primary_archetype(spec, index) in compatible[name]
                 # Oncology monitoring can supply an unrelated hemoglobin trend,
                 # so it cannot serve as a falling-hemoglobin negative control.
                 and not (name == "bleeding-risk" and variant == 1
                          and primary_archetype(spec, index) == "oncology")),
                None,
            )
            if selected is None:
                raise ValueError(f"insufficient compatible matched-control patients for {name}")
            control_indexes.append(selected)
    else:
        control_indexes = available_controls[:spec["problems"]["crossSourceClinicalPatients"]]
    return control_indexes


def _support_renal_state(spec, source, index, patient_url, truth, clinical):
    """Stage 4 is true in both risk and control cases; model its adult evidence."""
    from .clinical_state import codes, stage4_creatinine
    from .clinical_policy import completed_age_years
    born = datetime.fromisoformat(truth["birthDate"]).date()
    def creatinine(when):
        age = completed_age_years(born, datetime.fromisoformat(when[:10]).date())
        return stage4_creatinine(age, truth["gender"], 20 + index % 9)
    for _, resource in clinical:
        if resource.get("resourceType") == "Observation" and "2160-0" in codes(resource):
            when = resource["effectiveDateTime"]
            resource["valueQuantity"] = {"value": creatinine(when), "unit": "mg/dL", "system": "http://unitsofmeasure.org", "code": "mg/dL"}
            resource["interpretation"] = [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "H"}]}]
            resource["method"] = {"text": "IDMS-traceable creatinine; adult CKD-EPI 2021"}
    if source != "hmo":
        return
    diagnoses = [r for _, r in clinical if r.get("resourceType") == "Condition" and "431857002" in codes(r)]
    if not diagnoses:
        return
    when = min(parse_fhir_time(r.get("onsetDateTime") or r["recordedDate"]).start for r in diagnoses)
    birthday = datetime.fromisoformat(truth["birthDate"] + "T00:00:00+00:00")
    try:
        adult = birthday.replace(year=birthday.year + 18)
    except ValueError:
        adult = birthday.replace(year=birthday.year + 18, month=3, day=1)
    # Historical diagnosis onset is retained; current CKD-EPI confirmation
    # needs two independent adult samples rather than pediatric estimates.
    when = max(when, adult + timedelta(days=100))
    anchor = datetime.fromisoformat(spec["timelineAnchor"].replace("Z", "+00:00"))
    if when + timedelta(hours=1) >= anchor:
        raise ValueError("adult longitudinal renal assessment cannot fit before the clinical horizon")
    evidence = []
    for offset in (100, 0):
        measured = when - timedelta(days=offset)
        url = full_url(spec["datasetVersion"], source, cohort_id(index), "Observation", f"renal-stage-evidence-{offset}")
        value = creatinine(measured.date().isoformat())
        resource = {
            "resourceType": "Observation", "status": "final",
            "meta": meta(spec, source, PROFILES["Observation"], "laboratory-result", archetype=truth["primaryArchetype"]),
            "identifier": [strong_identifier(spec, source, "Observation", url.removeprefix("urn:uuid:"))],
            "subject": {"reference": patient_url, "identifier": {
                "system": spec["identity"]["cohortIdentifierSystem"], "value": cohort_id(index),
            }}, "effectiveDateTime": measured.isoformat().replace("+00:00", "Z"),
            "issued": (measured + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
            "category": [{"coding": [{"system": "http://snomed.info/sct", "code": "108252007", "display": "Laboratory procedure"}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": "2160-0", "display": "Creatinine [Mass/volume] in Serum or Plasma"}]},
            "valueQuantity": {"value": value, "unit": "mg/dL", "system": "http://unitsofmeasure.org", "code": "mg/dL"},
            "method": {"text": "IDMS-traceable creatinine; adult CKD-EPI 2021"},
            "interpretation": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "H"}]}],
            "referenceRange": [{"low": {"value": .5, "unit": "mg/dL", "system": "http://unitsofmeasure.org", "code": "mg/dL"}, "high": {"value": 1.2, "unit": "mg/dL", "system": "http://unitsofmeasure.org", "code": "mg/dL"}}],
            "performer": [{"reference": full_url(spec["datasetVersion"], source, "master", "Practitioner")}],
            "specimen": {"type": "Specimen", "identifier": {"system": "urn:hdp-test-bundle:renal-sample", "value": url}},
            "basedOn": [{"type": "ServiceRequest", "identifier": {"system": "urn:hdp-test-bundle:renal-order", "value": url}}],
            "note": [{"text": "Longitudinal renal assessment supports stage 4 under clinical-state contract 1.0.0."}],
            "text": {"status": "generated", "div": (
                '<div xmlns="http://www.w3.org/1999/xhtml">'
                f'Synthetic final serum creatinine: {value} mg/dL, '
                f'collected {measured.date().isoformat()}.</div>'
            )},
        }
        clinical.append((url, resource))
        evidence.append({"reference": url})
    for condition in diagnoses:
        condition["evidence"] = [{"detail": evidence}]


def build_population(spec: dict, repository_root: Path, monitor: Callable[[], None] | None = None) -> dict[str, Any]:
    geography_path = input_path(spec["geography"]["catalog"])
    catalog = load_catalog(geography_path, repository_root)
    organizations = spec["organizations"]
    per_org: dict[str, list[tuple[str, dict]]] = {item["id"]: _master_resources(spec, item) for item in organizations}
    crosswalk = []
    answer_by_id: dict[str, dict] = {}
    scenarios = []
    patient_urls_by_org: dict[str, list[str]] = {item["id"]: [] for item in organizations}
    coverage_per_org: dict[str, list[tuple[str, dict]]] = {item["id"]: [] for item in organizations}
    clinical_by_cohort: dict[int, dict[str, list[dict]]] = {}
    control_evidence_by_cohort: dict[int, dict[str, list[str]]] = {}
    latent_truth_ledger: list[dict[str, Any]] = []
    control_indexes = matched_controls(spec)
    control_position = {index: position + 1 for position, index in enumerate(control_indexes)}
    # Choose real local capture episodes before projecting any FHIR resources.
    # The v1 capture catalog defines the no-test, referral, cancellation, result,
    # review and summary graphs. Select eligible ordinary cohorts within each
    # stable 1,000-person block so a larger nested corpus retains block one.
    ordinary_capture: set[tuple[int, str]] = set()
    if spec.get("remediation") and spec.get("clinicalRealism"):
        candidates: dict[tuple[int, str, str], list[tuple[str, int]]] = defaultdict(list)
        controls = set(control_indexes)
        for candidate_index in range(1, spec["populationSize"] + 1):
            if candidate_index in controls or allocation(spec, candidate_index):
                continue
            archetype = primary_archetype(spec, candidate_index)
            candidate_locality = (
                choose_locality(catalog, block_position(candidate_index)[1], 1000)
                if nested(spec) else choose_locality(catalog, candidate_index, spec["populationSize"])
            )
            birth_date = demographic_truth(spec, candidate_index, candidate_locality)["birthDate"]
            pathway = care_pathway(spec, candidate_index, archetype, birth_date=birth_date)
            block = (candidate_index - 1) // 1000
            for source in pathway["activeSources"]:
                variant = pathway["encounters"][source]["captureVariant"]
                rank = hashlib.sha256(
                    f"{spec['seed']}|capture-episode|{candidate_index}|{source}|{variant}".encode()
                ).hexdigest()
                candidates[(block, source, variant)].append((rank, candidate_index))
        for (block, source, _variant), ranked in candidates.items():
            block_size = min(1000, spec["populationSize"] - block * 1000)
            target = max(1, (block_size * (30 if source == "hmo" else 20) + 999) // 1000)
            ordinary_capture.update(
                (candidate_index, source) for _rank, candidate_index in sorted(ranked)[:target]
            )
    for index in range(1, spec["populationSize"] + 1):
        if monitor and (index == 1 or index % 100 == 0):
            monitor()
        locality = choose_locality(catalog, block_position(index)[1], 1000) if nested(spec) else choose_locality(catalog, index, spec["populationSize"])
        truth = demographic_truth(spec, index, locality)
        problems = allocation(spec, index)
        control_offset = control_position.get(index, 0)
        is_control = bool(control_offset)
        control_name, control_variant = control_scenario(spec, control_offset) if is_control else (None, None)
        force_all_sources = is_control or any(problem["class"] == "cross-source-clinical" for problem in problems)
        renal_stage4 = (any(p.get("scenario") == "renal-medication-risk" for p in problems)
                        or (is_control and control_name == "renal-medication-risk"))
        latent_pathway = care_pathway(
            spec, index, truth["primaryArchetype"], force_all_sources=force_all_sources,
            birth_date=truth["birthDate"],
        ) if spec.get("clinicalRealism") else None
        typed_pathway = None
        typed_captures = None
        typed_routes = None
        if spec.get("remediation"):
            typed_pathway = build_patient_pathway(
                spec, index, truth["primaryArchetype"], latent_pathway["pathwayVariant"],
                truth["birthDate"], spec["remediation"]["pathwayTemplateCatalog"],
                spec["remediation"]["clinicalContentCatalog"],
                latent_pathway["secondaryTrait"],
                renal_stage4=renal_stage4,
                gender=truth["gender"],
            )
            typed_captures, typed_routes = project_capture(
                typed_pathway, spec["remediation"]["capturePolicyCatalog"], spec["seed"], locality_code=str(locality.get("code", "")), sensitivity_catalog=spec["sensitivity"]["catalog"],
            )
            latent_truth_ledger.extend(
                capture_ledger_rows(typed_pathway, typed_captures, typed_routes)
            )
        patient_views = {}
        clinical_by_cohort[index] = {}
        control_evidence_by_cohort[index] = {}
        anchor_digest = None
        for organization in organizations:
            org_id = organization["id"]
            patient_url = full_url(spec["datasetVersion"], org_id, cohort_id(index), "Patient")
            patient_urls_by_org[org_id].append(patient_url)
            patient_problem_ids, patient_problem_kinds = _problem_for_resource(problems, org_id, "Patient")
            patient, current_anchor, view_digest = patient_resource(
                spec, org_id, truth, index, patient_problem_ids, patient_problem_kinds,
                {row["kind"]: row.get("variant") for row in problems if row["problemId"] in patient_problem_ids},
            )
            anchor_digest = anchor_digest or current_anchor
            if current_anchor != anchor_digest:
                raise AssertionError("identity anchor drift")
            patient_entries = [(patient_url, patient)]
            duplicate_views = []
            if "probable-duplicate" in patient_problem_kinds:
                duplicate_url = full_url(spec["datasetVersion"], org_id, cohort_id(index), "Patient", "duplicate-candidate")
                duplicate = copy.deepcopy(patient)
                # A duplicate candidate is a distinct observable source record. It
                # intentionally carries neither the canonical cohort key nor any
                # generator-only linkage identifier; identity truth lives only in
                # the hidden crosswalk/answer key.
                duplicate["identifier"] = [strong_identifier(
                    spec, org_id, "Patient", duplicate_url.removeprefix("urn:uuid:")
                )]
                duplicate["name"][0]["use"] = "usual"
                duplicate_variant = _problem_variant(problems, org_id, "Patient", "probable-duplicate")
                if duplicate_variant == "changed-phone":
                    duplicate.pop("telecom", None)
                    duplicate["name"][0]["given"] = [duplicate["name"][0]["given"][0] + "h"]
                elif duplicate_variant == "minor-dob-discrepancy":
                    duplicate["birthDate"] = (datetime.fromisoformat(duplicate["birthDate"] + "T00:00:00+00:00") + timedelta(days=1)).date().isoformat()
                else:
                    duplicate["name"][0]["given"] = [duplicate["name"][0]["given"][0] + "h"]
                patient_entries.append((duplicate_url, duplicate))
                duplicate_views.append({
                    "patientFullUrl": duplicate_url,
                    "status": "probable-duplicate-candidate",
                    "canonicalCohortId": cohort_id(index),
                    "observableFeatureSet": ["normalized-name", "birth-date", "sex", "phone", "address"],
                })
            per_org[org_id].extend(patient_entries)
            guardian = _pediatric_guardian(spec, org_id, index, patient_url)
            if guardian:
                per_org[org_id].append(guardian)
            problem_requires_source = any(
                _problem_for_resource(problems, org_id, resource_type)[0]
                for resource_type in ("Encounter", "Condition", "Observation", "ServiceRequest", "MedicationRequest", "AllergyIntolerance")
            )
            active_source = utilizes_source(
                spec, index, truth["primaryArchetype"], org_id,
                force_all_sources=force_all_sources,
            ) or problem_requires_source or requires_source(spec, index, org_id) if spec.get("schemaVersion") == "2.0.0" else True
            if typed_captures:
                active_source = (
                    any(row.disposition != "absent" for row in typed_captures[org_id])
                    or bool(problems) or is_control or (index, org_id) in ordinary_capture
                )
            clean_by_url: dict[str, dict] = {}
            if problems and spec.get("clinicalRealism"):
                clean_patient, _clean_anchor, _clean_view = patient_resource(
                    spec, org_id, truth, index, [], [], {},
                )
                clean_by_url[patient_url] = clean_patient
                # Rebuild the deterministic source-capture before-state without
                # overlays. The hidden answer key can therefore reverse every
                # mutation exactly instead of describing a generic correction.
                clean_clinical = _clinical_resources(
                    spec, organization, index, patient_url, [], truth["birthDate"],
                    full_capture=True, gender=truth["gender"], fixture_context=True,
                    renal_stage4=renal_stage4,
                )
                clean_clinical.extend(_sensitive_fact_resources(
                    spec, organization, index, patient_url, truth["birthDate"],
                ))
                clean_clinical.extend(_archetype_resources(
                    spec, organization, index, patient_url, truth["birthDate"],
                ))
                clean_clinical.extend(_secondary_trait_resources(
                    spec, organization, index, patient_url, truth["birthDate"],
                ))
                clean_by_url.update({url: resource for url, resource in clean_clinical})
            if typed_pathway and active_source:
                clinical, mapping_rows = project_pathway(
                    spec, organization, index, patient_url, typed_pathway,
                    typed_captures[org_id],
                )
                latent_truth_ledger.extend(mapping_rows)
                # Problem fixtures and their matched controls retain their
                # reversible source-first substrate. Selected ordinary cohorts
                # receive an additional, actual source-local capture episode;
                # its omitted resources and revoked orders are projected by the
                # reviewed branch graph rather than inferred from its label.
                if problems or is_control:
                    clinical.extend(_clinical_resources(
                        spec, organization, index, patient_url, problems, truth["birthDate"],
                        full_capture=True, gender=truth["gender"],
                        renal_stage4=renal_stage4,
                    ))
                    clinical.extend(_archetype_resources(
                        spec, organization, index, patient_url, truth["birthDate"],
                        existing_medications={fact.code for fact in typed_pathway.facts
                            if fact.system == "http://www.nlm.nih.gov/research/umls/rxnorm"},
                    ))
                    clinical.extend(_secondary_trait_resources(
                        spec, organization, index, patient_url, truth["birthDate"],
                    ))
                elif (index, org_id) in ordinary_capture:
                    capture_episode = _clinical_resources(
                        spec, organization, index, patient_url, [], truth["birthDate"],
                        full_capture=False, gender=truth["gender"],
                    )
                    for _url, resource in capture_episode:
                        resource["meta"]["source"] = f"urn:hdp-test-bundle:source:{org_id}"
                    variant = latent_pathway["encounters"][org_id]["captureVariant"]
                    capture_setting = (
                        "community" if org_id == "hmo" else
                        ("tertiary-emergency" if "emergency" in variant else
                         "tertiary-inpatient" if variant in {"inpatient-procedure-workup", "short-admission"} else
                         "tertiary-ambulatory") if org_id == "hospital-a" else
                        ("regional-emergency" if "emergency" in variant else
                         "regional-inpatient" if variant in {"inpatient-procedure-workup", "short-admission"} else
                         "regional-ambulatory")
                    )
                    for url, resource in capture_episode:
                        occurrence = occurrence_from_resource(resource)
                        latent_truth_ledger.append({
                            "rowType": "supplemental-source-capture", "cohortId": cohort_id(index),
                            "source": org_id, "captureVariant": variant,
                            "episodeId": f"supplemental-{cohort_id(index)}-{org_id}",
                            "careSetting": capture_setting,
                            "resourceFullUrl": url, "resourceType": resource["resourceType"],
                            "occurrence": ({
                                "start": occurrence.interval.start.isoformat().replace("+00:00", "Z"),
                                "end": occurrence.interval.end.isoformat().replace("+00:00", "Z"),
                            } if occurrence else None),
                        })
                    clinical.extend(capture_episode)
            else:
                clinical = _clinical_resources(
                    spec, organization, index, patient_url, problems, truth["birthDate"],
                    full_capture=bool(problems) or force_all_sources, gender=truth["gender"],
                ) if active_source else []
            if active_source and not typed_pathway:
                clinical.extend(_sensitive_fact_resources(spec, organization, index, patient_url, truth["birthDate"]))
                clinical.extend(_archetype_resources(spec, organization, index, patient_url, truth["birthDate"]))
                clinical.extend(_secondary_trait_resources(
                    spec, organization, index, patient_url, truth["birthDate"],
                ))
            scenario_problem = next((row for row in problems if row.get("scenario")), None)
            if scenario_problem and spec.get("clinicalRealism"):
                scenario_evidence_urls = _scenario_variant_resources(
                    spec, org_id, index, patient_url, scenario_problem, clinical, birth_date=truth["birthDate"],
                )
                _mark_scenario_evidence(clinical, scenario_evidence_urls, "positive-evidence")
            if is_control:
                control_evidence_by_cohort[index][org_id] = _apply_matched_control(
                    spec, index, control_name, org_id, patient_url, clinical, control_variant,
                    truth["birthDate"],
                )
                if spec.get("clinicalRealism"):
                    control_problem = {"scenario": control_name, "variant": SCENARIO_VARIANTS[control_name][control_variant]}
                    control_evidence_by_cohort[index][org_id].extend(_scenario_variant_resources(
                        spec, org_id, index, patient_url, control_problem, clinical, control=True, birth_date=truth["birthDate"],
                    ))
                    _mark_scenario_evidence(
                        clinical, control_evidence_by_cohort[index][org_id], "near-miss-control",
                    )
            if spec.get("remediation"):
                from .clinical_state import acute_snapshot, codes, age_years
                for _, resource in clinical:
                    if resource.get("resourceType") == "Condition":
                        acute_snapshot(resource, spec["timelineAnchor"])
                    if resource.get("resourceType") == "Observation":
                        if codes(resource) & {"29463-7", "8302-2"}:
                            resource.pop("referenceRange", None)
                            resource.pop("interpretation", None)
                        if "19868-9" in codes(resource) and age_years(truth["birthDate"], resource["effectiveDateTime"]) < 6:
                            for field in ("valueQuantity", "valueString", "referenceRange", "interpretation"):
                                resource.pop(field, None)
                            resource["status"] = "cancelled"
                            resource["dataAbsentReason"] = {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/data-absent-reason", "code": "not-performed"}]}
                renal_scenario = (scenario_problem and scenario_problem.get("scenario") == "renal-medication-risk") or (is_control and control_name == "renal-medication-risk")
                if renal_scenario:
                    _support_renal_state(spec, org_id, index, patient_url, truth, clinical)
            clinical_by_cohort[index][org_id] = [resource for _, resource in clinical]
            per_org[org_id].extend(clinical)
            patient_views[org_id] = {
                "patientFullUrl": patient_url, "demographicDigest": view_digest,
                "anchorStatus": "canonical-anchor", "duplicateCandidates": duplicate_views,
                **({
                    "clinicalCapture": {
                        "active": bool(active_source),
                        "variant": latent_pathway["encounters"][org_id]["captureVariant"],
                        "mode": latent_pathway["encounters"][org_id]["captureMode"],
                        "workflowFamily": latent_pathway["encounters"][org_id]["workflowFamily"],
                        "resourceTypes": latent_pathway["encounters"][org_id]["captureResourceTypes"],
                    },
                } if latent_pathway else {}),
                **({
                    "latentCaptureDispositions": [
                        {
                            "eventId": row.latent_event_id,
                            "disposition": row.disposition,
                            "provenanceRequired": row.provenance_required,
                        }
                        for row in typed_captures[org_id]
                    ],
                } if typed_captures else {}),
            }
            for problem in problems:
                participating = [(url, res) for url, res in [*patient_entries, *clinical] if any(
                    tag.get("system") == spec["metadata"]["problemIdSystem"] and tag.get("code") == problem["problemId"]
                    for tag in res["meta"]["tag"]
                )]
                if not participating:
                    continue
                entry = answer_by_id.setdefault(problem["problemId"], {
                    **problem, "catalogVersion": spec["problems"]["catalogVersion"], "cohortId": cohort_id(index),
                    **({"expectedConclusions": [problem["variant"], *additional_fixture_conclusions(problem["variant"])]} if spec.get("remediation") and problem.get("scenario") else {}),
                    "participants": [], "transformation": problem["kind"],
                    "originalCleanTruth": {
                        "state": "pre-overlay source capture generated from the declared pathway",
                        "reconstruction": "Apply every participantState operation: replace restores the complete clean resource and remove deletes an overlay-only resource.",
                        "participantStates": [],
                    },
                    "mutation": {
                        "variant": problem.get("variant", "legacy-core"),
                        "affectedFields": {
                            "conflicting-diagnosis": ["Condition.code", "Condition.clinicalStatus", "Condition.onset[x]"],
                            "discordant-lab": ["Observation.valueQuantity", "Observation.specimen", "Observation.method"],
                            "timestamp-order": ["Observation.effective[x]", "Observation.issued"],
                            "semantically-wrong-code": ["Condition.code"],
                            "unit-mismatch": ["Observation.valueQuantity.unit", "Observation.valueQuantity.code", "Observation.valueQuantity.value"],
                            "stale-demographics": ["Patient.meta.lastUpdated", "Patient.address.period", "Patient.telecom"],
                            "probable-duplicate": ["Patient.name", "Patient.telecom", "Patient.birthDate"],
                            "cross-source-clinical-risk": ["cross-source decisive evidence graph"],
                        }[problem["kind"]],
                    },
                    "whyProblematic": "Observed source facts are inconsistent, semantically incompatible, implausibly ordered, or form the declared cross-source safety pattern.",
                    "expectedTruth": {
                        "acceptableResolutionSet": [
                            "apply the exact hidden participantState replacement/removal operations",
                            "record a designated-reviewer reconciliation that explicitly supersedes those exact operations",
                        ],
                        "hiddenCorrectionNotEmitted": True,
                    },
                    "detectorEvidenceRequired": [f"detect-{problem['kind']}", "observable FHIR facts only"],
                    "knownDetectorLimitations": "The detector is a fixture oracle, not authorized clinical decision support.",
                    "severityRationale": f"{problem['severity']} synthetic interoperability review priority",
                    "impact": "May mislead record linkage, longitudinal interpretation, or source reconciliation if left unexplained.",
                    "profileValid": True,
                    "nearMissControlRelationship": {"decisiveFields": [], "comparatorResources": []},
                    "reviewerQuestion": "Can the observed facts be reconciled to the declared clean state using only the answer-key evidence?",
                    "detectionRule": f"detect-{problem['kind']}",
                })
                existing_participants = {
                    (row["organization"], row["fullUrl"]) for row in entry["participants"]
                }
                existing_states = {
                    (row["organization"], row["fullUrl"])
                    for row in entry["originalCleanTruth"]["participantStates"]
                }
                for url, resource in participating:
                    key = (org_id, url)
                    if key not in existing_participants:
                        entry["participants"].append({
                            "organization": org_id,
                            "resourceType": resource["resourceType"],
                            "fullUrl": url,
                            "strongIdentifier": resource["identifier"][0]["value"],
                        })
                        existing_participants.add(key)
                    if key not in existing_states:
                        clean_resource = clean_by_url.get(url)
                        state = {
                            "organization": org_id,
                            "fullUrl": url,
                            "operation": "replace" if clean_resource is not None else "remove",
                        }
                        if clean_resource is not None:
                            state["resource"] = clean_resource
                        entry["originalCleanTruth"]["participantStates"].append(state)
                        existing_states.add(key)
                entry["nearMissControlRelationship"] = {
                    "decisiveFields": entry["mutation"]["affectedFields"],
                    "comparatorResources": [{"organization": state["organization"], "fullUrl": state["fullUrl"], "cleanOperation": state["operation"]} for state in entry["originalCleanTruth"]["participantStates"]],
                    "comparison": "Compare the emitted fields with the exact replacement state; removal identifies a source assertion absent from clean history.",
                }
        crosswalk.append({
            "cohortId": cohort_id(index), "identityAnchorSha256": anchor_digest,
            "primaryArchetype": truth["primaryArchetype"],
            "ordinaryPathwayEligible": not problems and not is_control,
            "evaluationClass": "problem" if problems else "matched-control" if is_control else "ordinary",
            **({
                "problemAssignments": [{key: row.get(key) for key in ("problemId", "class", "kind", "scenario", "variant")} for row in problems],
                "matchedControlScenario": control_name,
                "matchedControlVariant": SCENARIO_VARIANTS[control_name][control_variant] if is_control else None,
                "sensitivityFact": fact_for_patient(spec, index),
                "sourceUse": sorted(org_id for org_id, view in patient_views.items() if view.get("clinicalCapture", {}).get("active")),
            } if spec.get("remediation") else {}),
            **({
                "pathwayVariant": latent_pathway["pathwayVariant"],
                "secondaryTrait": latent_pathway["secondaryTrait"],
                **({"secondaryBranch": typed_pathway.secondary_branch} if typed_pathway else {}),
            } if latent_pathway else {}),
            "geographyStratum": {"localityCode": locality["code"], "district": locality["district"]},
            "sources": patient_views,
        })
        if problems and problems[0]["class"] == "cross-source-clinical":
            name = problems[0]["scenario"]
            # v1/v2 legacy allocations predate named topology variants. Their
            # historical predicates and structural labels remain selected below.
            scenario_variant = problems[0].get("variant", "legacy-core")
            participant_rows = answer_by_id[problems[0]["problemId"]]["participants"]
            if spec.get("clinicalRealism"):
                strict_three = bool(spec.get("remediation"))
                per_source = {org["id"]: evaluate_variant(name, scenario_variant, clinical_by_cohort[index][org["id"]], require_three_source_context=strict_three) for org in organizations}
                combined = evaluate_variant(name, scenario_variant, [resource for org in organizations for resource in clinical_by_cohort[index][org["id"]]], require_three_source_context=strict_three)
                subset_table = subset_truth_table_variant(name, scenario_variant, clinical_by_cohort[index], require_three_source_context=strict_three)
            else:
                per_source = {org["id"]: evaluate(name, clinical_by_cohort[index][org["id"]]) for org in organizations}
                combined = evaluate(name, [resource for org in organizations for resource in clinical_by_cohort[index][org["id"]]])
                subset_table = subset_truth_table(name, clinical_by_cohort[index])
            expected_table = {key: False for key in subset_table}
            expected_table["hmo+hospital-a+hospital-b"] = True
            discovery_mode = "three-source-required"
            if not spec.get("remediation") and name == "allergy-treatment-risk" and (not spec.get("clinicalRealism") or scenario_variant != "culture-informed-treatment"):
                discovery_mode = "two-source-discovery-with-third-source-enrichment"
                expected_table["hmo+hospital-b"] = True
            if subset_table != expected_table:
                raise AssertionError(f"scenario subset truth-table mismatch for {name}/{index}: {subset_table}")
            conditional = conditional_fixture_conclusions(scenario_variant, [r for rows in clinical_by_cohort[index].values() for r in rows]) if spec.get("remediation") else []
            if conditional:
                answer_by_id[problems[0]["problemId"]]["conditionalExpectedConclusions"] = conditional
                answer_by_id[problems[0]["problemId"]]["expectedConclusions"].extend(v for v in conditional if v not in answer_by_id[problems[0]["problemId"]]["expectedConclusions"])
            scenario_confidentiality, scenario_sensitivity, _ = _sensitivity(spec, index)
            scenarios.append({
                "scenarioId": f"scenario-{index:04d}", "name": name, "cohortId": cohort_id(index),
                **({"conditionalExpectedConclusions": conditional} if conditional else {}),
                "problemId": problems[0]["problemId"], "predicate": f"aggregate-{name}" + (f":{scenario_variant}" if spec.get("clinicalRealism") else ""),
                "perSource": per_source, "aggregate": combined,
                "discoveryMode": discovery_mode, "subsetTruthTable": subset_table,
                "structuralVariant": scenario_variant if spec.get("clinicalRealism") else ("core-exposure-arc", "monitored-course", "reconciliation-enriched")[((int(problems[0]["problemId"].split("-")[-1]) - 1) // 3) % 3],
                "expectedSubsetTruthTable": expected_table,
                "sourceEvidence": {
                    org["id"]: [row["fullUrl"] for row in participant_rows if row["organization"] == org["id"]]
                    for org in organizations
                },
                "aggregateEvidence": [row["fullUrl"] for row in participant_rows],
                "mobilityBehavior": {
                    "confidentiality": scenario_confidentiality or "ordinary-unlabeled",
                    "requiredClinicalSensitivityAuthorization": scenario_sensitivity if scenario_confidentiality == "R" else [],
                    "excludedAsVeryRestricted": scenario_confidentiality == "V",
                },
                "testStatus": "pass",
            })
    for organization in organizations:
        org_id = organization["id"]
        if spec["populationSize"] >= 2:
            if spec.get("clinicalRealism"):
                selected = patient_urls_by_org[org_id][-2:]
                coverage_per_org[org_id].extend(
                    _coverage_resources(spec, organization, repository_root, selected, "isolated-set-00")
                )
                continue
            if spec.get("schemaVersion") == "2.0.0":
                rare_count = next(row["count"] for row in distribution(spec) if row["id"] == "rare-complex")
                # A rounded-zero rare cohort must not select the entire population
                # via [-0:], attaching historical adult fixtures to infants.
                eligible = [url for url, resource in per_org[org_id]
                            if resource.get("resourceType") == "Patient"
                            and resource.get("birthDate", "9999") <= "1997-01-15"]
                selected = patient_urls_by_org[org_id][-rare_count:] if rare_count else eligible[-2:]
                if not selected:
                    raise ValueError("legacy coverage requires an adult patient for historical fixtures")
                coverage_count = max(1, (len(selected) + 1) // 2)
            else:
                coverage_count = min(10, max(1, spec["populationSize"] // 100))
                selected = patient_urls_by_org[org_id][-min(20, len(patient_urls_by_org[org_id])):]
            for coverage_index in range(coverage_count):
                pair = selected[(coverage_index * 2) % len(selected):(coverage_index * 2) % len(selected) + 2]
                if len(pair) < 2:
                    pair = [*pair, selected[0]]
                per_org[org_id].extend(_coverage_resources(spec, organization, repository_root, pair, f"set-{coverage_index:02d}"))
    for offset, index in enumerate(control_indexes):
        name, control_variant = control_scenario(spec, offset + 1)
        scenario_variant = SCENARIO_VARIANTS[name][control_variant] if spec.get("clinicalRealism") else ("core-exposure-arc", "monitored-course", "reconciliation-enriched")[control_variant]
        if spec.get("clinicalRealism"):
            per_source = {org["id"]: evaluate_variant(name, scenario_variant, clinical_by_cohort[index][org["id"]]) for org in organizations}
            combined = evaluate_variant(name, scenario_variant, [resource for org in organizations for resource in clinical_by_cohort[index][org["id"]]])
            subset_table = subset_truth_table_variant(name, scenario_variant, clinical_by_cohort[index])
        else:
            per_source = {org["id"]: evaluate(name, clinical_by_cohort[index][org["id"]]) for org in organizations}
            combined = evaluate(name, [resource for org in organizations for resource in clinical_by_cohort[index][org["id"]]])
            subset_table = subset_truth_table(name, clinical_by_cohort[index])
        expected_control = {key: False for key in subset_table}
        if subset_table != expected_control:
            raise AssertionError(f"control subset truth-table mismatch for {name}/{index}: {subset_table}")
        source_evidence = control_evidence_by_cohort[index]
        conditional = conditional_fixture_conclusions(scenario_variant, [r for rows in clinical_by_cohort[index].values() for r in rows]) if spec.get("remediation") else []
        scenarios.append({"scenarioId": f"control-{offset + 1:04d}", "name": name, "cohortId": cohort_id(index), "matchedControl": True, **({"additionalExpectedConclusions": additional_fixture_conclusions(scenario_variant, control=True) + [v for v in conditional if v not in additional_fixture_conclusions(scenario_variant, control=True)], **({"conditionalExpectedConclusions": conditional} if conditional else {})} if spec.get("remediation") else {}), "predicate": f"aggregate-{name}" + (f":{scenario_variant}" if spec.get("clinicalRealism") else ""), "perSource": per_source, "aggregate": combined, "discoveryMode": "matched-near-miss-control", "structuralVariant": scenario_variant, "subsetTruthTable": subset_table, "expectedSubsetTruthTable": expected_control, "sourceEvidence": source_evidence, "aggregateEvidence": [url for org in organizations for url in source_evidence.get(org["id"], [])], "testStatus": "pass"})
    assign_deaths(spec, per_org, crosswalk, clinical_by_cohort)
    condition_allocation = None
    if spec["terminology"]["catalog"] == "tools/synthetic_population/data/terminology.v2.4.0.json":
        from .condition_allocation import allocate_condition_coding
        condition_allocation = allocate_condition_coding(per_org, spec)
    bundles = {}
    coverage_bundles = {}
    for organization in organizations:
        org_id = organization["id"]
        entries = _order_transaction_entries([
            {"fullUrl": url, "resource": resource, "request": {"method": "POST", "url": resource["resourceType"]}}
            for url, resource in per_org[org_id]
        ])
        bundles[org_id] = {"resourceType": "Bundle", "type": "transaction", "meta": {"tag": [{"system": spec["metadata"]["datasetTagSystem"], "code": spec["datasetVersion"]}, {"system": spec["metadata"]["sourceTagSystem"], "code": org_id}]}, "entry": entries}
        if spec.get("clinicalRealism"):
            coverage_bundles[org_id] = {
                "resourceType": "Bundle", "type": "collection",
                "meta": {"tag": [
                    {"system": spec["metadata"]["datasetTagSystem"], "code": spec["datasetVersion"]},
                    {"system": spec["metadata"]["sourceTagSystem"], "code": org_id},
                    {"system": "urn:hdp-test-bundle:synthetic-artifact-role", "code": "isolated-conformance-coverage"},
                ]},
                "entry": [{"fullUrl": url, "resource": resource} for url, resource in coverage_per_org[org_id]],
            }
    answer = {"catalogVersion": spec["problems"]["catalogVersion"], "specSha256": sha256_json(spec), "count": len(answer_by_id), "problems": [answer_by_id[key] for key in sorted(answer_by_id)]}
    scenario_evidence = {"count": len(scenarios), "scenarios": scenarios}
    patient_accounting = {
        "canonicalPersonCount": len(crosswalk),
        "anchorPatientCountBySource": {
            org_id: sum(1 for row in crosswalk if row["sources"].get(org_id, {}).get("anchorStatus") == "canonical-anchor")
            for org_id in sorted(bundles)
        },
        "duplicatePatientRecordCountBySource": {
            org_id: sum(len(row["sources"].get(org_id, {}).get("duplicateCandidates", [])) for row in crosswalk)
            for org_id in sorted(bundles)
        },
    }
    patient_accounting["totalPatientResourcesBySource"] = {
        org_id: patient_accounting["anchorPatientCountBySource"][org_id] + patient_accounting["duplicatePatientRecordCountBySource"][org_id]
        for org_id in sorted(bundles)
    }
    return {
        "bundles": bundles,
        "coverageBundles": coverage_bundles,
        "crosswalk": {"count": len(crosswalk), "patientAccounting": patient_accounting, "patients": crosswalk},
        "answerKey": answer, "scenarioEvidence": scenario_evidence, "geography": catalog,
        "latentTruthLedger": latent_truth_ledger,
        "conditionAllocation": condition_allocation,
    }
