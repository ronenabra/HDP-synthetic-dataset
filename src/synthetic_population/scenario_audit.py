from __future__ import annotations

import copy
from datetime import datetime, timedelta
from typing import Any

from .observable import patient_graphs
from .observable import codings
from .predicates import evaluate_variant, renal_trend_results, subset_truth_table_variant
from .catalogs import load_catalog
from .problems import SCENARIO_VARIANTS, conditional_fixture_conclusions


def _future(value: str) -> str:
    point = datetime.fromisoformat(value.replace("Z", "+00:00")) + timedelta(days=3650)
    if len(value) == 10:
        return point.date().isoformat()
    return point.isoformat().replace("+00:00", "Z")


def _tagged_evidence(resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return resources


def _counterfactuals(name: str, variant: str, resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mutations = []
    rxnorm = {"renal-medication-risk": "9997", "bleeding-risk": "11289", "allergy-treatment-risk": "723"}[name]
    loinc = {"renal-medication-risk": "6298-4", "bleeding-risk": "718-7"}.get(name)

    decisive = {
        "renal-medication-risk": [
            ("renal-condition", "Condition", "http://snomed.info/sct", "431857002"),
            ("renal-medication", "MedicationRequest", "http://www.nlm.nih.gov/research/umls/rxnorm", "9997"),
            ("abnormal-potassium", "Observation", "http://loinc.org", "6298-4"),
        ],
        "bleeding-risk": [
            ("anticoagulant", "MedicationRequest", "http://www.nlm.nih.gov/research/umls/rxnorm", "11289"),
            ("bleeding-condition", "Condition", "http://snomed.info/sct", "74474003"),
            ("low-hemoglobin", "Observation", "http://loinc.org", "718-7"),
        ],
        "allergy-treatment-risk": [
            ("allergy", "AllergyIntolerance", "http://snomed.info/sct", "91936005"),
            ("treatment", "MedicationRequest", "http://www.nlm.nih.gov/research/umls/rxnorm", "723"),
            ("culture-enrichment", "Observation", "http://loinc.org", "630-4"),
        ],
    }[name]
    if variant == "acute-renal-precipitant":
        decisive.append(("acute-precipitant", "Condition", "http://snomed.info/sct", "14669001"))
    if variant == "interacting-exposure":
        decisive.append(("interacting-medication", "MedicationRequest", "http://www.nlm.nih.gov/research/umls/rxnorm", "1191"))
    if variant == "administration-after-allergy":
        decisive.append(("administration", "MedicationAdministration", "http://www.nlm.nih.gov/research/umls/rxnorm", "723"))
    if variant == "culture-informed-treatment":
        decisive.append(("culture-order", "ServiceRequest", "http://loinc.org", "630-4"))
        decisive.append(("culture-specimen", "Specimen", "http://snomed.info/sct", "119297000"))
    for label, resource_type, system, code in decisive:
        removed = [
            row for row in copy.deepcopy(resources)
            if not (row.get("resourceType") == resource_type and (system, code) in set(codings(row)))
        ]
        mutations.append({
            "mutation": f"remove-{label}",
            "passed": not evaluate_variant(name, variant, removed, require_three_source_context=True),
        })

    reversed_time = copy.deepcopy(resources)
    medications = [row for row in reversed_time if row.get("resourceType") == "MedicationRequest" and row.get("status") == "active" and ("http://www.nlm.nih.gov/research/umls/rxnorm", rxnorm) in set(codings(row.get("medicationCodeableConcept")))]
    if medications:
        for medication in medications:
            medication["authoredOn"] = (
                "1900-01-01" if name == "allergy-treatment-risk"
                else _future(medication["authoredOn"])
            )
        mutations.append({"mutation": "reverse-exposure-order", "passed": not evaluate_variant(name, variant, reversed_time, require_three_source_context=True)})

    inactive = copy.deepcopy(resources)
    medications = [row for row in inactive if row.get("resourceType") == "MedicationRequest" and ("http://www.nlm.nih.gov/research/umls/rxnorm", rxnorm) in set(codings(row.get("medicationCodeableConcept")))]
    if medications:
        for medication in medications:
            medication["status"] = "entered-in-error"
        mutations.append({"mutation": "invalidate-exposure-status", "passed": not evaluate_variant(name, variant, inactive, require_three_source_context=True)})

    if name in {"renal-medication-risk", "bleeding-risk"}:
        ended = copy.deepcopy(resources)
        for medication in ended:
            if medication.get("resourceType") == "MedicationRequest" and (
                "http://www.nlm.nih.gov/research/umls/rxnorm", rxnorm
            ) in set(codings(medication.get("medicationCodeableConcept"))):
                medication.setdefault("dosageInstruction", [{}])[0].setdefault("timing", {}).setdefault("repeat", {})["boundsPeriod"] = {
                    "start": "1900-01-01", "end": "1900-01-02",
                }
        mutations.append({
            "mutation": "stop-medication-before-risk-window",
            "passed": not evaluate_variant(name, variant, ended, require_three_source_context=True),
        })

    if name in {"renal-medication-risk", "bleeding-risk"}:
        wrong_unit = copy.deepcopy(resources)
        observations = [row for row in wrong_unit if row.get("resourceType") == "Observation" and row.get("valueQuantity") and ("http://loinc.org", loinc) in set(codings(row.get("code")))]
        if observations:
            for observation in observations:
                observation["valueQuantity"]["code"] = "kg"
            mutations.append({"mutation": "wrong-observation-unit", "passed": not evaluate_variant(name, variant, wrong_unit, require_three_source_context=True)})

    if variant == "administration-after-allergy":
        wrong_event = copy.deepcopy(resources)
        administrations = [row for row in wrong_event if row.get("resourceType") == "MedicationAdministration"]
        if administrations:
            for administration in administrations:
                administration["resourceType"] = "MedicationStatement"
            mutations.append({"mutation": "substitute-statement-for-administration", "passed": not evaluate_variant(name, variant, wrong_event, require_three_source_context=True)})
    if variant in {"chronic-renal-trend", "falling-hemoglobin"}:
        reversed_trend = copy.deepcopy(resources)
        rows = [
            row for row in (renal_trend_results(reversed_trend) if variant == "chronic-renal-trend" else _tagged_evidence(reversed_trend)) if row.get("resourceType") == "Observation"
            and ("http://loinc.org", loinc) in set(codings(row.get("code")))
            and isinstance(row.get("valueQuantity", {}).get("value"), (int, float))
        ]
        ordered = sorted(rows, key=lambda row: row.get("effectiveDateTime", ""))
        values = [row["valueQuantity"]["value"] for row in ordered]
        for row, value in zip(ordered, sorted(values) if variant == "falling-hemoglobin" else reversed(values)):
            row["valueQuantity"]["value"] = value
        mutations.append({"mutation": "reverse-repeated-lab-trend", "passed": not evaluate_variant(name, variant, reversed_trend, require_three_source_context=True)})
        if ordered:
            removed_one = copy.deepcopy(resources)
            if variant in {"falling-hemoglobin", "chronic-renal-trend"}:
                # Remove the historical comparator series while retaining the
                # current low result; background results can predate the risk window.
                retained = ordered[-1].get("_fullUrl")
                victims = {row.get("_fullUrl") for row in ordered[:-1]}
                removed_one = [row for row in removed_one if row.get("_fullUrl") not in victims or row.get("_fullUrl") == retained]
                label = "remove-required-trend-history"
            mutations.append({"mutation": label, "passed": not evaluate_variant(name, variant, removed_one, require_three_source_context=True)})
    if variant == "medication-dose-escalation":
        ordered_medications = sorted(
            (row for row in _tagged_evidence(resources) if row.get("resourceType") == "MedicationRequest" and ("http://www.nlm.nih.gov/research/umls/rxnorm", "9997") in set(codings(row.get("medicationCodeableConcept")))),
            key=lambda row: row.get("authoredOn", ""),
        )
        if ordered_medications:
            removed_low = [row for row in copy.deepcopy(resources) if row.get("_fullUrl") != ordered_medications[0].get("_fullUrl")]
            mutations.append({"mutation": "remove-low-dose-event", "passed": not evaluate_variant(name, variant, removed_low, require_three_source_context=True)})
        if len(ordered_medications) >= 2:
            reversed_doses = copy.deepcopy(resources)
            dose_rows = sorted(
                (row for row in _tagged_evidence(reversed_doses)
                 if row.get("resourceType") == "MedicationRequest"
                 and ("http://www.nlm.nih.gov/research/umls/rxnorm", "9997") in set(codings(row.get("medicationCodeableConcept")))),
                key=lambda row: row.get("authoredOn", ""),
            )
            first = dose_rows[0]["dosageInstruction"][0]["doseAndRate"][0]["doseQuantity"]
            second = dose_rows[1]["dosageInstruction"][0]["doseAndRate"][0]["doseQuantity"]
            first["value"], second["value"] = max(first["value"], second["value"]), min(first["value"], second["value"])
            mutations.append({
                "mutation": "reverse-dose-order-retaining-dose-range",
                "passed": not evaluate_variant(name, variant, reversed_doses, require_three_source_context=True),
            })
    if variant == "interacting-exposure":
        nonoverlap = copy.deepcopy(resources)
        for row in nonoverlap:
            if row.get("resourceType") == "MedicationRequest" and ("http://www.nlm.nih.gov/research/umls/rxnorm", "1191") in set(codings(row.get("medicationCodeableConcept"))):
                repeat = row.setdefault("dosageInstruction", [{}])[0].setdefault("timing", {}).setdefault("repeat", {})
                repeat["boundsPeriod"] = {"start": "1900-01-01", "end": "1900-01-02"}
        mutations.append({"mutation": "eliminate-interacting-exposure-overlap", "passed": not evaluate_variant(name, variant, nonoverlap, require_three_source_context=True)})
    if name == "allergy-treatment-risk":
        late_allergy = copy.deepcopy(resources)
        for row in late_allergy:
            if row.get("resourceType") == "AllergyIntolerance" and ("http://snomed.info/sct", "91936005") in set(codings(row.get("code"))):
                row["onsetDateTime"] = "2099-01-01T00:00:00Z"
        mutations.append({"mutation": "record-allergy-after-treatment", "passed": not evaluate_variant(name, variant, late_allergy, require_three_source_context=True)})
    if variant == "culture-informed-treatment":
        late_culture = copy.deepcopy(resources)
        for row in late_culture:
            if row.get("resourceType") == "Observation" and ("http://loinc.org", "630-4") in set(codings(row.get("code"))):
                row["effectiveDateTime"] = "2099-01-01T00:00:00Z"
        mutations.append({"mutation": "move-culture-after-treatment", "passed": not evaluate_variant(name, variant, late_culture, require_three_source_context=True)})
    matched_control = copy.deepcopy(resources)
    control_type, control_system, control_code = {
        "renal-medication-risk": ("Observation", "http://loinc.org", "6298-4"),
        "bleeding-risk": ("Observation", "http://loinc.org", "718-7"),
        "allergy-treatment-risk": ("MedicationRequest", "http://www.nlm.nih.gov/research/umls/rxnorm", "723"),
    }[name]
    control_resources = [
        row for row in matched_control
        if row.get("resourceType") == control_type and (control_system, control_code) in set(codings(row))
    ]
    if control_resources:
        for control_resource in control_resources:
            coding_container = (
                control_resource.get("medicationCodeableConcept")
                if control_type == "MedicationRequest" else control_resource.get("code")
            )
            for coding in coding_container.get("coding", []):
                if coding.get("system") == control_system and coding.get("code") == control_code:
                    coding["code"] = f"matched-control-{control_code}"
                    coding.pop("display", None)
        mutations.append({
            "mutation": "replace-decisive-code-with-matched-control",
            "passed": not evaluate_variant(name, variant, matched_control, require_three_source_context=True),
        })
    documentation_only = copy.deepcopy(resources)
    for row in documentation_only:
        if row.get("resourceType") == "Observation" and row.get("issued"):
            row["issued"] = "2099-01-01T00:00:00Z"
        if row.get("resourceType") == "MedicationStatement" and row.get("dateAsserted"):
            row["dateAsserted"] = "1900-01-01T00:00:00Z"
    mutations.append({
        "mutation": "change-documentation-time-only",
        "passed": evaluate_variant(name, variant, documentation_only, require_three_source_context=True),
    })
    return mutations


def build_scenario_causality_audit(
    bundles: dict[str, dict[str, Any]], crosswalk: dict[str, Any],
    scenario_evidence: dict[str, Any], spec: dict[str, Any],
) -> dict[str, Any]:
    graphs, _patients = patient_graphs(bundles, spec, include_demographics=False)
    timing_catalog = load_catalog(spec["remediation"]["scenarioTimingCatalog"])
    if set(timing_catalog) != {"catalogVersion", "status", "disclaimer", "variantContracts", "scenarios"}:
        raise ValueError("scenario timing catalog has unknown or missing root fields")
    contract_fields = {
        "name", "variant", "eventRoles", "sourceOwnership", "occurrenceFields",
        "exposureDerivation", "relations", "matchedControl",
    }
    contracts = {}
    for contract in timing_catalog["variantContracts"]:
        if set(contract) != contract_fields or set(contract["sourceOwnership"]) != {"hmo", "hospital-a", "hospital-b"}:
            raise ValueError("scenario variant contract has unknown, missing, or incomplete fields")
        if not all(contract[field] for field in ("eventRoles", "occurrenceFields", "exposureDerivation", "relations", "matchedControl")):
            raise ValueError("scenario variant contract contains an empty executable requirement")
        key = (contract["name"], contract["variant"])
        if key in contracts:
            raise ValueError(f"duplicate scenario variant contract: {key}")
        contracts[key] = contract
    expected_contracts = {
        (name, variant) for name, variants in SCENARIO_VARIANTS.items() for variant in variants
    }
    if set(contracts) != expected_contracts:
        raise ValueError("scenario variant contracts do not cover the complete structural-variant registry")
    rows = []
    failures = []
    for scenario in scenario_evidence.get("scenarios", []):
        cohort = scenario["cohortId"]
        name = scenario["name"]
        variant = scenario["structuralVariant"]
        resources = graphs.get(cohort, [])
        observed = evaluate_variant(name, variant, resources, require_three_source_context=True)
        expected = not scenario.get("matchedControl", False)
        contract = contracts[(name, variant)]
        subsets = subset_truth_table_variant(name, variant, {source: [row for row in resources if row.get("_source") == source] for source in ("hmo", "hospital-a", "hospital-b")}, require_three_source_context=True)
        proper_subsets_false = all(
            not value for key, value in subsets.items()
            if key != "hmo+hospital-a+hospital-b"
        )
        mutations = _counterfactuals(name, variant, resources) if expected else []
        conditional = conditional_fixture_conclusions(variant, resources)
        conditional_declared = scenario.get("conditionalExpectedConclusions", [])
        conditional_correct = conditional == conditional_declared
        passed = observed == expected and proper_subsets_false and all(row["passed"] for row in mutations) and conditional_correct
        if not passed:
            failures.append(scenario["scenarioId"])
        rows.append({
            "scenarioId": scenario["scenarioId"], "cohortId": cohort,
            "name": name, "variant": variant, "matchedControl": not expected,
            "aggregateExpected": expected, "aggregateObserved": observed,
            "allProperSourceSubsetsFalse": proper_subsets_false, "recomputedSubsetTruthTable": subsets,
            "timingPolicy": timing_catalog["scenarios"]["culture-informed-treatment" if variant == "culture-informed-treatment" else name],
            "variantContract": contract,
            "counterfactuals": mutations, "passed": passed,
            **({"conditionalPositiveConclusions": conditional, "conditionalLabelsCorrect": conditional_correct,
                "assignedControlVariantNegative": not observed if not expected else None} if conditional or conditional_declared else {}),
        })
    scenario_cohorts = {row["cohortId"] for row in scenario_evidence.get("scenarios", [])}
    incidental = []
    for declaration in crosswalk["patients"]:
        cohort = declaration["cohortId"]
        if cohort in scenario_cohorts or not declaration.get("ordinaryPathwayEligible", True):
            continue
        for name, variants in SCENARIO_VARIANTS.items():
            for variant in variants:
                if evaluate_variant(name, variant, graphs.get(cohort, []), require_three_source_context=True):
                    incidental.append({"cohortId": cohort, "name": name, "variant": variant})
    if incidental:
        failures.extend(f"incidental:{row['cohortId']}:{row['name']}:{row['variant']}" for row in incidental)
    exercised = {(row["name"], row["variant"]) for row in rows if not row["matchedControl"]}
    return {
        "schemaVersion": "4.0.0", "temporalRegistry": "explicit-primary-clinical-occurrence-v1",
        "timingCatalogVersion": timing_catalog["catalogVersion"],
        "scenarioCount": len(rows), "positiveCount": sum(not row["matchedControl"] for row in rows),
        "controlCount": sum(row["matchedControl"] for row in rows),
        "variantContractCount": len(contracts),
        "exercisedPositiveVariantCount": len(exercised),
        "unexercisedPilotVariantContracts": [
            {"name": name, "variant": variant}
            for name, variant in sorted(set(contracts) - exercised)
        ],
        "incidentalOrdinaryPositiveCount": len(incidental),
        "incidentalOrdinaryPositives": incidental,
        "failureCount": len(failures), "failures": failures, "scenarios": rows,
        "status": "pass" if not failures else "fail",
    }
