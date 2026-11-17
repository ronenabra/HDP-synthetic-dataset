from __future__ import annotations

import math
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from .catalogs import load_catalog
from .clinical_semantics import meaningful_result, population_semantic_audit
from .clinical_catalog import (
    age_applicability,
    birth_date_or_none,
    completed_age_years,
    unresolved_age_applicability,
    validate_clinical_content_catalog,
)
from .observable import codings, patient_graphs, references
from .temporal import monotonic_trend, occurrence_from_resource
from .typed_grammar import ROLE_TYPES
from .utils import read_json


INTENSIVE = {"chronic-care", "medication-intensive", "oncology"}
GROWTH_CODES = {"29463-7", "8302-2", "39156-5"}
TRAJECTORY_CLASSES = (
    "stable", "resolving", "persistent", "worsening", "improving", "exposure-related",
)


def _age_band(age_days: int) -> str:
    if age_days < 365:
        return "infant"
    if age_days < 12 * 365:
        return "child"
    if age_days < 18 * 365:
        return "adolescent"
    if age_days < 65 * 365:
        return "adult"
    return "older-adult"


def _observable_role(resource: dict[str, Any], family_roles: dict[tuple[str, str], set[str]]) -> str:
    for system, code in codings(resource.get("type")):
        if system == "urn:hdp-test-bundle:synthetic:encounter-type":
            return code
    for system, code in codings(resource.get("category")):
        if system == "urn:hdp-test-bundle:synthetic:service-request-category":
            return code
    candidates = {
        role for coding in codings(resource) for role in family_roles.get(coding, set())
        if resource.get("resourceType") in ROLE_TYPES[role]
    }
    if len(candidates) == 1:
        return next(iter(candidates))
    return resource.get("resourceType", "unknown")


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def build_clinical_depth_audit(
    bundles: dict[str, dict[str, Any]], crosswalk: dict[str, Any], spec: dict[str, Any],
) -> dict[str, Any]:
    graphs, patients = patient_graphs(bundles, spec, include_demographics=False)
    semantic_audit = population_semantic_audit(bundles, spec)
    graphs = {cohort: [r for r in resources if meaningful_result(r)] for cohort, resources in graphs.items()}
    declarations = {row["cohortId"]: row for row in crosswalk["patients"]}
    content = load_catalog(spec["remediation"]["clinicalContentCatalog"])
    repository_root = Path(__file__).resolve().parent
    reference_context = read_json(repository_root / spec["remediation"]["referenceContext"])
    catalog_audit = validate_clinical_content_catalog(
        spec["remediation"]["clinicalContentCatalog"],
        spec["remediation"]["pathwayTemplateCatalog"], repository_root,
        reference_context,
    )
    analyte_catalog = {row["code"]: row for row in content["analytes"]}
    birth_dates = {cohort: birth_date_or_none(patient) for cohort, patient in patients.items()}
    pathway_catalog = load_catalog(spec["remediation"]["pathwayTemplateCatalog"])
    family_map: dict[tuple[str, str], set[str]] = defaultdict(set)
    family_roles: dict[tuple[str, str], set[str]] = defaultdict(set)
    configured_variants = []
    configured_roles = set()
    for configured_archetype, variants in pathway_catalog["archetypes"].items():
        for variant in variants:
            configured_variants.append(f"{configured_archetype}/{variant['id']}")
            for event in variant["events"]:
                key = (event["system"], event["code"])
                family_map[key].add(event["family"])
                family_roles[key].add(event["role"])
                configured_roles.add(event["role"])
    ordinary = {
        cohort: resources for cohort, resources in graphs.items()
        if declarations.get(cohort, {}).get("ordinaryPathwayEligible", True)
    }
    analyte_patients: dict[str, set[str]] = defaultdict(set)
    condition_patients: dict[str, set[str]] = defaultdict(set)
    medication_patients: dict[str, set[str]] = defaultdict(set)
    from .clinical_policy import population_branch_quantitative_findings
    quantitative_errors = [f"{row['source']}:{row['resource']}:{row['reason']}"
                           for row in population_branch_quantitative_findings(bundles, content)]
    quantitative_age_policies: Counter[str] = Counter()
    resource_counts = []
    repeated_intensive = 0
    intensive_count = 0
    pediatric_rows = []
    vaccine_codes = set()
    vaccine_patterns = set()
    for cohort, resources in ordinary.items():
        resource_counts.append(len(resources))
        analyte_dates: dict[str, set[date]] = defaultdict(set)
        growth_dates: set[date] = set()
        patient_vaccines = []
        for resource in resources:
            resource_codes = set(codings(resource))
            if resource.get("resourceType") == "Condition":
                condition_patients.update({code: condition_patients[code] | {cohort} for system, code in resource_codes if system == "http://snomed.info/sct"})
            if resource.get("resourceType") in {"MedicationRequest", "MedicationDispense", "MedicationStatement", "MedicationAdministration"}:
                for system, code in resource_codes:
                    if system == "http://www.nlm.nih.gov/research/umls/rxnorm":
                        medication_patients[code].add(cohort)
            if resource.get("resourceType") == "Observation":
                occurrence = occurrence_from_resource(resource)
                quantitative_nodes = [(resource.get("code", {}), resource, "valueQuantity")]
                quantitative_nodes.extend(
                    (component.get("code", {}), component, f"component[{position}].valueQuantity")
                    for position, component in enumerate(resource.get("component", []))
                )
                for code_value, value_node, value_path in quantitative_nodes:
                    code = next((code for system, code in codings(code_value) if system == "http://loinc.org"), None)
                    if code and occurrence:
                        analyte_patients[code].add(cohort)
                        analyte_dates[code].add(occurrence.interval.start.date())
                        if code in GROWTH_CODES:
                            growth_dates.add(occurrence.interval.start.date())
                    if code and "valueQuantity" in value_node:
                        catalog_row = analyte_catalog.get(code)
                        quantity = value_node["valueQuantity"]
                        ranges = value_node.get("referenceRange", [])
                        interpretation = {value for system, value in codings(value_node.get("interpretation")) if system.endswith("v3-ObservationInterpretation")}
                        if value_node is resource and any(row["code"] == code for row in content.get("branchQuantitativeModels", [])):
                            # Shared branch checks include value/unit/status/time and
                            # metadata, without clinical-reference/abnormal-rate credit.
                            continue
                        if not catalog_row:
                            quantitative_errors.append(f"{cohort}:{resource.get('_fullUrl')}:{value_path}:{code}:unknown-analyte")
                            continue
                        born = birth_dates.get(cohort)
                        age_years = completed_age_years(born, occurrence.interval.start.date()) if born and occurrence else None
                        age_policy = age_applicability(content, code, age_years) if age_years is not None else unresolved_age_applicability(code)
                        quantitative_age_policies[f"{code}:{age_policy['id']}"] += 1
                        common_error = quantity.get("code") != catalog_row["unit"] \
                            or resource.get("status") != "final" \
                            or not isinstance(quantity.get("value"), (int, float)) \
                            or occurrence is None
                        numeric_value = quantity.get("value") if isinstance(quantity.get("value"), (int, float)) else 0
                        expected_interpretation = {
                            "H" if numeric_value > catalog_row["high"]
                            else "L" if numeric_value < catalog_row["low"] else "N"
                        }
                        if age_policy["referenceRangeHandling"] == "required":
                            range_error = not ranges \
                                or ranges[0].get("low", {}).get("value") != catalog_row["low"] \
                                or ranges[0].get("high", {}).get("value") != catalog_row["high"]
                        else:
                            range_error = bool(ranges)
                        if age_policy["interpretationHandling"] == "required":
                            interpretation_error = interpretation != expected_interpretation
                        else:
                            # Pediatric BMI has no loaded age/sex reference catalog;
                            # carrying adult interpretation metadata would be a false
                            # clinical claim. Unit/status/value checks remain active.
                            interpretation_error = bool(interpretation)
                        age_error = range_error or interpretation_error
                        if common_error or age_error:
                            quantitative_errors.append(f"{cohort}:{resource.get('_fullUrl')}:{value_path}:{code}")
            if resource.get("resourceType") == "Immunization":
                code = next((code for system, code in codings(resource.get("vaccineCode")) if system == "http://snomed.info/sct"), None)
                if code:
                    vaccine_codes.add(code)
                    patient_vaccines.append(code)
        archetype = declarations[cohort]["primaryArchetype"]
        if archetype in INTENSIVE:
            intensive_count += 1
            repeated_intensive += int(any(len(dates) >= 2 for dates in analyte_dates.values()))
        if archetype == "pediatric":
            born = birth_dates.get(cohort)
            anchor = date.fromisoformat(spec["timelineAnchor"][:10])
            age_days = (anchor - born).days if born is not None else None
            eligible_two_dates = age_days is not None and age_days >= 180
            pediatric_rows.append({
                "cohortId": cohort, "ageDays": age_days,
                "growthDateCount": len(growth_dates), "hasGrowth": bool(growth_dates),
                "twoDateEligible": eligible_two_dates,
                "twoDatePassed": not eligible_two_dates or len(growth_dates) >= 2,
                "immunizationCount": len(patient_vaccines),
            })
            vaccine_patterns.add(tuple(sorted(patient_vaccines)))
    # Pediatric safety applies to every declared pediatric person, including
    # intentional-problem and matched-control graphs; only corpus breadth floors
    # use the narrower ordinary-population denominator.
    pediatric_rows = []
    vaccine_codes = set()
    vaccine_patterns = set()
    bmi_mismatches = []
    bmi_age_policy_counts: Counter[str] = Counter()
    for cohort, declaration in declarations.items():
        born = birth_dates.get(cohort)
        resources = graphs.get(cohort, [])
        growth_dates: set[date] = set()
        patient_vaccines = []
        measurements: dict[tuple[date, str], float] = {}
        for resource in resources:
            occurrence = occurrence_from_resource(resource)
            if resource.get("resourceType") == "Observation" and occurrence:
                code = next((value for system, value in codings(resource.get("code")) if system == "http://loinc.org"), None)
                if code in GROWTH_CODES:
                    day = occurrence.interval.start.date()
                    growth_dates.add(day)
                    if isinstance(resource.get("valueQuantity", {}).get("value"), (int, float)):
                        measurements[(day, code)] = float(resource["valueQuantity"]["value"])
            if resource.get("resourceType") == "Immunization":
                code = next((value for system, value in codings(resource.get("vaccineCode")) if system == "http://snomed.info/sct"), None)
                if code:
                    vaccine_codes.add(code)
                    patient_vaccines.append(code)
        for (day, code), bmi in measurements.items():
            if code != "39156-5":
                continue
            age_years = completed_age_years(born, day) if born is not None else None
            age_policy = age_applicability(content, code, age_years) if age_years is not None else unresolved_age_applicability(code)
            bmi_age_policy_counts[age_policy["id"]] += 1
            if not age_policy["requiresDerivedPair"]:
                continue
            height = measurements.get((day, "8302-2"))
            weight = measurements.get((day, "29463-7"))
            if (
                height is None or weight is None or not math.isfinite(bmi) or bmi <= 0
                or not isinstance(height, (int, float)) or not isinstance(weight, (int, float))
                or not math.isfinite(height) or not math.isfinite(weight) or height <= 0 or weight <= 0
                or abs(weight / ((height / 100) ** 2) - bmi) > .05
            ):
                bmi_mismatches.append({"cohortId": cohort, "date": day.isoformat()})
        if declaration["primaryArchetype"] != "pediatric":
            continue
        anchor = date.fromisoformat(spec["timelineAnchor"][:10])
        age_days = (anchor - born).days if born is not None else None
        eligible_two_dates = age_days is not None and age_days >= 180
        pediatric_rows.append({
            "cohortId": cohort, "ageDays": age_days,
            "growthDateCount": len(growth_dates), "hasGrowth": bool(growth_dates),
            "twoDateEligible": eligible_two_dates,
            "twoDatePassed": not eligible_two_dates or len(growth_dates) >= 2,
            "immunizationCount": len(patient_vaccines),
        })
        vaccine_patterns.add(tuple(sorted(patient_vaccines)))
    distribution: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    trajectory_counts: Counter[str] = Counter()
    trajectory_patients: dict[str, set[str]] = defaultdict(set)
    reason_strata: dict[str, Counter[str]] = defaultdict(Counter)
    reason_resources: dict[str, list[dict]] = defaultdict(list)
    diagnosis_strata: dict[str, Counter[str]] = defaultdict(Counter)
    compatibility_failures: dict[str, list[str]] = {
        "conditionMedication": [], "encounterReason": [], "orderResult": [],
        "specimenTest": [], "ageMeasurement": [], "rangeInterpretation": list(quantitative_errors),
        "procedureFollowUp": [],
    }

    def add_distribution(dimension: str, key: str, cohort: str) -> None:
        row = distribution[dimension].setdefault(key, {"resourceCount": 0, "patients": set()})
        row["resourceCount"] += 1
        row["patients"].add(cohort)

    for cohort, resources in ordinary.items():
        declaration = declarations[cohort]
        patient = patients[cohort]
        born = birth_dates.get(cohort)
        if born is None:
            compatibility_failures["ageMeasurement"].append(f"{cohort}: missing or invalid birthDate")
        by_url = {resource.get("_fullUrl"): resource for resource in resources}
        has_indication = any(resource.get("resourceType") in {
            "Condition", "Procedure", "AllergyIntolerance", "Encounter", "ServiceRequest",
        } for resource in resources)
        analyte_series: dict[str, list[tuple[date, float]]] = defaultdict(list)
        for resource in resources:
            occurrence = occurrence_from_resource(resource)
            event_day = occurrence.interval.start.date() if occurrence else date.fromisoformat(spec["timelineAnchor"][:10])
            age_days = (event_day - born).days if born is not None else None
            role = _observable_role(resource, family_roles)
            clinical_families = sorted({
                family for coding in codings(resource) for family in family_map.get(coding, set())
            }) or ["unmapped-reviewed-background"]
            add_distribution("ageBand", _age_band(max(0, age_days)) if age_days is not None else "unknown", cohort)
            add_distribution("archetype", declaration["primaryArchetype"], cohort)
            add_distribution("materialVariant", f"{declaration['primaryArchetype']}/{declaration['pathwayVariant']}", cohort)
            add_distribution("source", resource.get("_source", "unknown"), cohort)
            add_distribution("eventRole", role, cohort)
            for family in clinical_families:
                add_distribution("conceptFamily", family, cohort)
            if resource.get("resourceType") == "Observation" and occurrence:
                nodes = [(resource.get("code", {}), resource)] + [
                    (component.get("code", {}), component) for component in resource.get("component", [])
                ]
                for code_node, value_node in nodes:
                    analyte_code = next((value for system, value in codings(code_node) if system == "http://loinc.org"), None)
                    value = value_node.get("valueQuantity", {}).get("value")
                    if analyte_code:
                        add_distribution("analyte", analyte_code, cohort)
                    if analyte_code and isinstance(value, (int, float)):
                        analyte_series[analyte_code].append((event_day, float(value)))
                if age_days is not None and age_days < 0:
                    compatibility_failures["ageMeasurement"].append(f"{cohort}:{resource.get('_fullUrl')}: pre-birth observation")
            if resource.get("resourceType") in {
                "MedicationRequest", "MedicationDispense", "MedicationStatement", "MedicationAdministration",
            } and not has_indication:
                compatibility_failures["conditionMedication"].append(
                    f"{cohort}:{resource.get('_fullUrl')}: medication has no observable condition/procedure context"
                )
            if resource.get("resourceType") == "Encounter":
                reason_codes = [
                    f"{system}|{code}" for system, code in codings(resource.get("reasonCode"))
                    if system in {"http://snomed.info/sct", "http://loinc.org"}
                ]
                if not reason_codes and not resource.get("reasonReference") and not resource.get("diagnosis"):
                    compatibility_failures["encounterReason"].append(
                        f"{cohort}:{resource.get('_fullUrl')}: encounter has no observable reason"
                    )
                if role != "follow-up":
                    class_code = next((code for system, code in codings(resource.get("class"))), "unknown")
                    stratum = f"{resource.get('_source', 'unknown')}|{class_code}"
                    reason_strata[stratum].update(reason_codes or ["reference-only"])
                    reason_resources[stratum].append(resource)
            if resource.get("resourceType") == "Condition":
                diagnosis_strata[resource.get("_source", "unknown")].update(
                    f"{system}|{code}" for system, code in codings(resource.get("code"))
                    if system == "http://snomed.info/sct"
                )
            for reference in references(resource.get("basedOn")):
                if reference not in by_url or by_url[reference].get("resourceType") != "ServiceRequest":
                    compatibility_failures["orderResult"].append(
                        f"{cohort}:{resource.get('_fullUrl')}: basedOn does not resolve to ServiceRequest"
                    )
            for reference in references(resource.get("specimen")):
                if reference not in by_url or by_url[reference].get("resourceType") != "Specimen":
                    compatibility_failures["specimenTest"].append(
                        f"{cohort}:{resource.get('_fullUrl')}: specimen reference does not resolve"
                    )
            if resource.get("resourceType") == "Procedure":
                for reference in references({"basedOn": resource.get("basedOn"), "reasonReference": resource.get("reasonReference")}):
                    if reference not in by_url:
                        compatibility_failures["procedureFollowUp"].append(
                            f"{cohort}:{resource.get('_fullUrl')}: procedure indication reference does not resolve"
                        )
        for analyte_code, values in analyte_series.items():
            ordered_values = [value for _day, value in sorted(values)]
            if len({day for day, _value in values}) < 2:
                continue
            row = analyte_catalog.get(analyte_code)
            midpoint = (row["low"] + row["high"]) / 2 if row else ordered_values[0]
            if monotonic_trend(ordered_values, "stable", tolerance=max(.05, abs(midpoint) * .05)):
                trajectory = "persistent" if row and all(value < row["low"] or value > row["high"] for value in ordered_values) else "stable"
            elif abs(ordered_values[-1] - midpoint) < abs(ordered_values[0] - midpoint):
                trajectory = "improving" if row and row["low"] <= ordered_values[-1] <= row["high"] else "resolving"
            else:
                trajectory = "worsening"
            trajectory_counts[trajectory] += 1
            trajectory_patients[trajectory].add(cohort)

    configured_distribution_keys = {
        "ageBand": ["infant", "child", "adolescent", "adult", "older-adult", "unknown"],
        "archetype": sorted(pathway_catalog["archetypes"]),
        "materialVariant": sorted(configured_variants),
        "source": ["hmo", "hospital-a", "hospital-b"],
        "eventRole": sorted(configured_roles),
        "conceptFamily": sorted({family for families in family_map.values() for family in families}),
        "analyte": sorted(analyte_catalog),
        "repeatedTrajectoryClass": list(TRAJECTORY_CLASSES),
    }
    for trajectory in TRAJECTORY_CLASSES:
        distribution["repeatedTrajectoryClass"][trajectory] = {
            "resourceCount": trajectory_counts[trajectory], "patients": trajectory_patients[trajectory],
        }
    distribution_rows = []
    for dimension, keys in configured_distribution_keys.items():
        for key in keys:
            row = distribution[dimension].get(key, {"resourceCount": 0, "patients": set()})
            distribution_rows.append({
                "dimension": dimension, "key": key,
                "resourceCount": row["resourceCount"], "patientCount": len(row["patients"]),
                "patientRate": round(len(row["patients"]) / max(1, len(ordinary)), 6),
            })

    def concentration_rows(values: dict[str, Counter[str]]) -> list[dict[str, Any]]:
        output = []
        for stratum, counts in sorted(values.items()):
            denominator = sum(counts.values())
            top_code, top_count = counts.most_common(1)[0] if counts else (None, 0)
            output.append({
                "stratum": stratum, "denominator": denominator, "topConcept": top_code,
                "topCount": top_count, "topShare": round(top_count / max(1, denominator), 6),
            })
        return output
    encounter_reason_concentration = concentration_rows(reason_strata)
    source_diagnosis_concentration = concentration_rows(diagnosis_strata)
    total_resources = sum(len(bundle["entry"]) for bundle in bundles.values())
    repeated_rate = repeated_intensive / max(1, intensive_count)
    pediatric_eligible = [row for row in pediatric_rows if row["twoDateEligible"]]
    pediatric_repeat_rate = sum(row["twoDatePassed"] for row in pediatric_eligible) / max(1, len(pediatric_eligible))
    failures = [f"{row['finding']} {row['cohortId']}: {row['reason']}" for row in semantic_audit["findings"]]
    if any(not row["hasGrowth"] for row in pediatric_rows):
        failures.append("a pediatric patient lacks an observable growth/vital measurement")
    if pediatric_repeat_rate < .50:
        failures.append("fewer than 50% of eligible pediatric patients have two growth dates")
    if len(vaccine_codes) < 5 or len(vaccine_patterns) < 3:
        failures.append("pediatric immunization concept/pattern floor failed")
    if bmi_mismatches:
        failures.append("pediatric BMI lacks a same-date recomputable height/weight pair")
    canonical_floor_applicable = spec["populationSize"] >= 1000
    if canonical_floor_applicable and (len(analyte_patients) < 12 or sum(len(cohorts) >= 50 for cohorts in analyte_patients.values()) < 8):
        failures.append("ordinary analyte breadth/count floor failed")
    if quantitative_errors:
        failures.append("quantitative metadata compatibility failed")
    compatibility_failure_count = sum(len(rows) for rows in compatibility_failures.values())
    if compatibility_failure_count:
        failures.extend(
            f"clinical compatibility {category}: {len(rows)} failures; {rows[0]}"
            for category, rows in compatibility_failures.items() if rows
        )
    if repeated_rate < .30:
        failures.append("intensive repeated-analyte floor failed")
    if canonical_floor_applicable and (len(condition_patients) < 12 or len(medication_patients) < 8):
        failures.append("condition/medication family floor failed")
    canonical_projection = math.ceil(total_resources * 1000 / spec["populationSize"])
    if canonical_projection >= 60000 or _percentile(resource_counts, .95) >= 75:
        failures.append("resource size ceiling exceeded")
    release_dispositions = []
    concentration_failed = False
    if canonical_floor_applicable:
        for row in encounter_reason_concentration + source_diagnosis_concentration:
            if row["denominator"] < 10 or row["topShare"] <= .35:
                continue
            disposition = (None
                           if row in encounter_reason_concentration else None)
            concentration_failed = True
    if concentration_failed:
        failures.append("encounter reason/source diagnosis concentration exceeds 35%")
    source_bundle_bytes = sum(
        len(json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        for bundle in bundles.values()
    )
    projection_factor = 1000 / spec["populationSize"]
    projected_resources = math.ceil(total_resources * projection_factor)
    projected_bytes = math.ceil(source_bundle_bytes * projection_factor)
    return {
        "schemaVersion": "4.1.0", "semanticAudit": semantic_audit, "resultContent": semantic_audit["resultContent"], "ordinaryPatientCount": len(ordinary),
        "clinicalContentCatalog": catalog_audit,
        "canonicalFixtureFloorsApplied": canonical_floor_applicable,
        "canonicalProjectionFactor": round(projection_factor, 6),
        "analytes": {code: {"patientCount": len(analyte_patients.get(code, set()))} for code in sorted(analyte_catalog)},
        "analyteConceptCount": len(analyte_patients),
        "analytesWithAtLeast50Patients": sum(len(rows) >= 50 for rows in analyte_patients.values()),
        "quantitativeCompatibilityFailureCount": len(quantitative_errors),
        "quantitativeCompatibilityFailures": quantitative_errors[:100],
        "quantitativeAgeApplicability": dict(sorted(quantitative_age_policies.items())),
        "conditionFamilyCount": len(condition_patients), "medicationIngredientFamilyCount": len(medication_patients),
        "intensivePatientCount": intensive_count, "repeatedRelevantAnalyteCount": repeated_intensive,
        "repeatedRelevantAnalyteRate": round(repeated_rate, 6),
        "pediatric": {
            "patientCount": len(pediatric_rows), "twoDateEligibleCount": len(pediatric_eligible),
            "twoDatePassRate": round(pediatric_repeat_rate, 6),
            "vaccineConceptCount": len(vaccine_codes), "schedulePatternCount": len(vaccine_patterns),
            "bmiRecomputationFailureCount": len(bmi_mismatches), "bmiRecomputationFailures": bmi_mismatches,
            "bmiAgeApplicability": dict(sorted(bmi_age_policy_counts.items())),
            "rows": pediatric_rows,
        },
        "compatibility": {
            key: {"failureCount": len(rows), "failures": rows[:100]}
            for key, rows in compatibility_failures.items()
        },
        "compatibilityFailureCount": compatibility_failure_count,
        "encounterReasonConcentration": encounter_reason_concentration,
        "sourceDiagnosisConcentration": source_diagnosis_concentration,
        "distributionRows": distribution_rows,
        "distributionDimensions": sorted(configured_distribution_keys),
        "zeroCountConfiguredRowCount": sum(row["resourceCount"] == 0 for row in distribution_rows),
        "pilotCanonicalProjection": {
            "projectedExactResourceCount": projected_resources,
            "projectedSourceBundleBytes": projected_bytes,
            "projectedValidationUnits": projected_resources + 3,
            "basis": "linear projection from exact resources and serialized source Bundles",
        },
        "size": {"exactResourceCount": total_resources, "patientLinkedClinicalP95": _percentile(resource_counts, .95),
                 **({"resourceCeilingUnit": "per 1000 patients", "canonicalProjectedExactResourceCount": canonical_projection}
                    if spec["populationSize"] > 1000 else {})},
        "failureCount": len(failures), "failures": failures,
        "status": "pass" if not failures else "fail",
    }
