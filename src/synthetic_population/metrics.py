from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime
from math import ceil, isfinite
from typing import Any, Iterable

from .catalogs import load_catalog
from .clinical_catalog import age_applicability, birth_date_or_none, completed_age_years, unresolved_age_applicability
from .temporal import occurrence_from_resource
from .lifecycle import build_lifecycle_audit
from .clinical_policy import population_branch_quantitative_findings


EVENT_KEYS = {
    "authoredOn", "recordedDate", "effectiveDateTime", "issued", "receivedTime",
    "occurrenceDateTime", "performedDateTime", "date", "sent", "started", "whenHandedOver", "onsetDateTime",
}


def _walk(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _dates(resource: dict[str, Any]) -> set[date]:
    values = []
    for key, value in _walk(resource):
        if key in EVENT_KEYS and isinstance(value, str) and len(value) >= 10:
            values.append(value)
        elif key in {"start", "end"} and isinstance(value, str) and len(value) >= 10:
            values.append(value)
    return {date.fromisoformat(value[:10]) for value in values}


def _codes(value: Any) -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        if isinstance(value.get("system"), str) and isinstance(value.get("code"), str):
            yield value["system"], value["code"]
        for item in value.values():
            yield from _codes(item)
    elif isinstance(value, list):
        for item in value:
            yield from _codes(item)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = ceil(position)
    if lower == upper:
        return ordered[lower]
    return round(ordered[lower] * (upper - position) + ordered[upper] * (position - lower), 4)


def _distribution_guards(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Combine the legacy reviewed guards with v4's consumed content rows."""
    guards = dict(load_catalog(spec["clinicalDistributions"]["catalog"])["guardrails"])
    remediation = spec.get("remediation")
    if not remediation:
        return guards
    for analyte in load_catalog(remediation["clinicalContentCatalog"])["analytes"]:
        span = float(analyte["high"]) - float(analyte["low"])
        generated = {
            "label": analyte["display"], "unit": analyte["unit"],
            "referenceLow": analyte["low"], "referenceHigh": analyte["high"],
            "ordinaryMin": analyte["low"], "ordinaryMax": analyte["high"],
            "abnormalLow": round(float(analyte["low"]) - max(.1, span * .2), 4),
            "abnormalHigh": round(float(analyte["high"]) + max(.1, span * .2), 4),
            "minimumOrdinaryAbnormalRate": 0.0,
            "maximumOrdinaryAbnormalRate": 0.15,
            "maximumUntaggedValue": round(float(analyte["high"]) + max(.2, span * .25), 4),
            "allowedStatuses": ["final"], "requiresInterpretation": True,
            "requiresEventTime": True,
            "stratification": ["age", "archetype"],
            "correlations": ["v4 deterministic pathway and age strata"],
        }
        if analyte["code"] in guards:
            existing = guards[analyte["code"]]
            if (existing["unit"], existing["referenceLow"], existing["referenceHigh"]) != (
                analyte["unit"], analyte["low"], analyte["high"]
            ):
                raise ValueError(f"clinical content conflicts with reviewed distribution guard {analyte['code']}")
        else:
            guards[analyte["code"]] = generated
    for model in load_catalog(remediation["clinicalContentCatalog"]).get("branchQuantitativeModels", []):
        guards[model["code"]] = {"label": model["display"], "unit": model["unit"],
            "ordinaryMin": model["generationMinimum"], "ordinaryMax": model["generationMaximum"],
            "allowedStatuses": model["allowedStatuses"], "requiresEventTime": model["requiresEventTime"],
            "requiresInterpretation": False}
    return guards


def _patient_maps(bundles: dict[str, dict[str, Any]], spec: dict[str, Any]) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    system = spec["identity"]["cohortIdentifierSystem"]
    url_to_cohort: dict[str, str] = {}
    hmo_patients: dict[str, dict[str, Any]] = {}
    for organization, bundle in bundles.items():
        for entry in bundle["entry"]:
            resource = entry["resource"]
            if resource.get("resourceType") != "Patient":
                continue
            cohort = next((item.get("value") for item in resource.get("identifier", []) if item.get("system") == system), None)
            if cohort:
                url_to_cohort[entry["fullUrl"]] = cohort
                if organization == "hmo" and cohort not in hmo_patients:
                    hmo_patients[cohort] = resource
    return url_to_cohort, hmo_patients


def _cohort(resource: dict[str, Any], url_to_cohort: dict[str, str], cohort_system: str) -> str | None:
    if resource.get("resourceType") == "Patient":
        identifiers = resource.get("identifier", [])
        return next((item.get("value") for item in identifiers if item.get("system") == cohort_system), None)
    for _, value in _walk(resource):
        if isinstance(value, str) and value in url_to_cohort:
            return url_to_cohort[value]
    for _, value in _walk(resource):
        if isinstance(value, dict) and value.get("system") == cohort_system:
            return value.get("value")
    return None


def realized_problem_allocation(bundles: dict[str, dict], spec: dict) -> dict:
    """Count emitted problem IDs, deduplicating repeated source assertions."""
    url_to_cohort, _ = _patient_maps(bundles, spec)
    by_cohort = defaultdict(set)
    ids = set()
    safety_ids = set()
    for bundle in bundles.values():
        for entry in bundle['entry']:
            resource = entry['resource']
            tags = resource.get('meta', {}).get('tag', [])
            if resource.get('resourceType') == 'Condition' and any(
                tag.get('system') == spec['metadata']['problemKindSystem']
                and tag.get('code') == 'local-only-condition-coding' for tag in tags
            ):
                continue  # S18 quality coding is outside the pre-existing Q fixture quota.
            observed = {tag['code'] for tag in tags if tag.get('system') == spec['metadata']['problemIdSystem']}
            ids.update(observed)
            cohort = _cohort(resource, url_to_cohort, spec['identity']['cohortIdentifierSystem'])
            if cohort:
                by_cohort[cohort].update(observed)
            if any(tag.get('system') == spec['metadata']['problemKindSystem'] and tag.get('code') == 'cross-source-clinical-risk' for tag in tags):
                safety_ids.update(observed)
    return {
        'allocationPolicyVersion': spec['problems'].get('allocationPolicyVersion', 'legacy-unversioned'),
        'patientsWithOne': sum(bool(values) for values in by_cohort.values()),
        'patientsWithSecond': sum(len(values) >= 2 for values in by_cohort.values()),
        'crossSourceClinicalPatients': len(safety_ids), 'problemIdCount': len(ids),
        'populationCoverageClaim': 'realized assignments only; fixture branch coverage is separate',
    }


def build_quality_metrics(bundles: dict[str, dict[str, Any]], crosswalk: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    from .capture import observable_origin, metadata_only_record
    url_to_cohort, hmo_patients = _patient_maps(bundles, spec)
    patient_birth_dates: dict[str, date] = {}
    # Resolve age from the canonical HMO Patient. Falling back to an unrelated
    # source copy would make a missing canonical DOB appear resolved.
    for cohort, patient in hmo_patients.items():
        birth = birth_date_or_none(patient)
        if birth is not None:
            patient_birth_dates[cohort] = birth
    content = load_catalog(spec["remediation"]["clinicalContentCatalog"]) if spec.get("remediation") else {"modelDefinitions": {}}
    from .clinical_state import population_state_audit
    state_audit = population_state_audit(bundles, spec) if spec.get("remediation") else {}
    supported_renal = set(state_audit.get("supportedStage4Cohorts", []))
    performed_care: dict[str, set[str]] = defaultdict(set)
    dates_by_cohort: dict[str, set[date]] = defaultdict(set)
    source_graphs: dict[str, dict[str, list[tuple[Any, ...]]]] = defaultdict(lambda: defaultdict(list))
    archetype_graphs: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    numeric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    encounters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mammography_cohorts: set[str] = set()
    problem_system = spec["metadata"]["problemMarkerSystem"]
    for organization, bundle in bundles.items():
        for entry in bundle["entry"]:
            resource = entry["resource"]
            cohort = _cohort(resource, url_to_cohort, spec["identity"]["cohortIdentifierSystem"])
            if cohort:
                if resource["resourceType"] not in {"Patient", "RelatedPerson"}:
                    dates_by_cohort[cohort].update(_dates(resource))
                graph_row = (
                        resource["resourceType"],
                        tuple(profile.rsplit("/", 1)[-1] for profile in resource.get("meta", {}).get("profile", [])),
                        tuple(sorted(code for system, code in _codes({key: resource.get(key) for key in ("code", "medicationCodeableConcept", "category")}) if system in {"http://loinc.org", "http://snomed.info/sct", "http://www.nlm.nih.gov/research/umls/rxnorm"})),
                        resource.get("status"),
                        tuple(sorted((_dates(resource)))),
                )
                if resource["resourceType"] != "Patient":
                    archetype_graphs[cohort].append(graph_row)
                if resource["resourceType"] not in {"Patient", "RelatedPerson"} and not any(
                    tag.get("system") == "urn:hdp-test-bundle:synthetic-archetype" and tag.get("code") == "compact-coverage"
                    for tag in resource.get("meta", {}).get("tag", [])
                ):
                    if not metadata_only_record(resource):
                        source_graphs[cohort][organization].append(graph_row)
                        if observable_origin(resource, organization) == organization:
                            performed_care[cohort].add(organization)
            if resource.get("resourceType") == "Observation":
                quantities = []
                if isinstance(resource.get("valueQuantity", {}).get("value"), (int, float)):
                    quantities.append((resource.get("code"), resource["valueQuantity"], resource.get("referenceRange", []), resource.get("interpretation", [])))
                quantities.extend(
                    (component.get("code"), component["valueQuantity"], component.get("referenceRange", []), component.get("interpretation", []))
                    for component in resource.get("component", [])
                    if isinstance(component.get("valueQuantity", {}).get("value"), (int, float))
                )
                intentional = any(tag.get("system") in {problem_system, "urn:hdp-test-bundle:synthetic-scenario-role"} for tag in resource.get("meta", {}).get("tag", []))
                for codeable, quantity, reference_ranges, interpretation in quantities:
                    loinc = next((code for system, code in _codes(codeable) if system == "http://loinc.org"), None)
                    if not loinc:
                        continue
                    reference = reference_ranges[0] if reference_ranges else {}
                    observation = occurrence_from_resource(resource)
                    observation_dates = [observation.interval.start.date()] if observation else []
                    birth = patient_birth_dates.get(cohort)
                    age_years = completed_age_years(birth, observation_dates[0]) if birth and observation_dates else None
                    applicability = age_applicability(content, loinc, age_years) if age_years is not None else unresolved_age_applicability(loinc)
                    numeric[loinc].append({
                        "value": float(quantity["value"]), "unit": quantity.get("code"),
                        "referenceLow": reference.get("low", {}).get("value"),
                        "referenceHigh": reference.get("high", {}).get("value"),
                        "intentional": intentional, "status": resource.get("status"),
                        "hasInterpretation": bool(interpretation), "hasEventTime": observation is not None,
                        "interpretations": [code for system, code in _codes(interpretation) if system == "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation"],
                        "problemTagged": any(tag.get("system") == problem_system for tag in resource.get("meta", {}).get("tag", [])),
                        "ageYears": age_years,
                        "supportedRenal": loinc == "2160-0" and cohort in supported_renal,
                        "ageApplicability": applicability,
                    })
            if resource.get("resourceType") == "Encounter" and cohort:
                period = resource.get("period", {})
                duration_hours = None
                if period.get("start") and period.get("end"):
                    duration_hours = (datetime.fromisoformat(period["end"].replace("Z", "+00:00")) - datetime.fromisoformat(period["start"].replace("Z", "+00:00"))).total_seconds() / 3600
                encounters[organization].append({
                    "class": resource.get("class", {}).get("code"), "durationHours": duration_hours,
                    "reasonCodes": sorted(code for _, code in _codes(resource.get("reasonCode"))),
                })
            if cohort and resource.get("resourceType") == "Procedure" and any(
                system == "http://snomed.info/sct" and code == "71651007" for system, code in _codes(resource.get("code"))
            ):
                mammography_cohorts.add(cohort)
    longitudinal_rows = []
    for patient in crosswalk["patients"]:
        values = sorted(dates_by_cohort[patient["cohortId"]])
        longitudinal_rows.append({
            "cohortId": patient["cohortId"], "primaryArchetype": patient["primaryArchetype"],
            "distinctClinicalDates": len(values), "spanDays": (values[-1] - values[0]).days if values else 0,
        })
    identical = 0
    compared = 0
    for cohort in hmo_patients:
        graphs = source_graphs[cohort]
        if graphs.get("hospital-a") and graphs.get("hospital-b"):
            compared += 1
            identical += sorted(graphs["hospital-a"]) == sorted(graphs["hospital-b"])
    archetype_content_failures = []
    for patient in crosswalk["patients"]:
        cohort = patient["cohortId"]
        archetype = patient["primaryArchetype"]
        graph = archetype_graphs[cohort]
        types = {item[0] for item in graph}
        codes = {(item[0], code) for item in graph for code in item[2]}
        required_types = {
            "chronic-care": {"Condition", "Observation", "MedicationRequest"},
            "pediatric": {"RelatedPerson", "Encounter", "Observation", "Immunization"},
            "medication-intensive": {"MedicationRequest", "MedicationStatement", "Observation"},
            # Screening is age/risk driven; archetype membership alone must not
            # force mammography outside the routine policy window.
            "womens-health": {"Observation"},
            "perioperative-surgical": {"ServiceRequest", "Procedure", "Encounter"},
            "oncology": {"Condition", "Procedure", "Observation"},
        }.get(archetype, set())
        if not required_types.issubset(types):
            archetype_content_failures.append(f"{cohort}:{archetype}:missing-{','.join(sorted(required_types - types))}")
        if archetype == "perioperative-surgical" and ("ServiceRequest", "80146002") not in codes:
            archetype_content_failures.append(f"{cohort}:{archetype}:missing-surgical-order")
        if archetype == "oncology" and not {("Condition", "363346000"), ("Procedure", "367336001")}.issubset(codes):
            archetype_content_failures.append(f"{cohort}:{archetype}:missing-oncology-diagnosis-or-treatment")
        if archetype == "rare-complex" and len(types) < 10:
            archetype_content_failures.append(f"{cohort}:{archetype}:insufficient-profile-dispersion-{len(types)}")
    distributions = {}
    guards = _distribution_guards(spec) if spec.get("schemaVersion") == "2.0.0" else {}
    for loinc, rows in sorted(numeric.items()):
        values = [row["value"] for row in rows]
        ordinary_rows = [row for row in rows if not row["intentional"]]
        applicable_rows = [
            row for row in ordinary_rows
            if row["ageApplicability"].get("ordinaryDistributionGuardApplicable", True) and not row.get("supportedRenal")
        ]
        age_excluded_rows = [row for row in ordinary_rows if not row["ageApplicability"].get("ordinaryDistributionGuardApplicable", True)]
        ordinary = [row["value"] for row in applicable_rows]
        age_excluded_values = [row["value"] for row in age_excluded_rows]
        guard = guards.get(loinc, {})
        in_range = [value for value in ordinary if guard.get("referenceLow", float("-inf")) <= value <= guard.get("referenceHigh", float("inf"))]
        unexplained_extremes = [value for value in ordinary if "maximumUntaggedValue" in guard and value > guard["maximumUntaggedValue"]]
        age_excluded_metadata_violations = sum(
            (
                (row["ageApplicability"].get("referenceRangeHandling") == "omit"
                 and (row["referenceLow"] is not None or row["referenceHigh"] is not None))
                or (row["ageApplicability"].get("referenceRangeHandling") == "required"
                    and (row["referenceLow"] is None or row["referenceHigh"] is None))
                or (row["ageApplicability"].get("interpretationHandling") == "omit" and row["hasInterpretation"])
                or (row["ageApplicability"].get("interpretationHandling") == "required" and not row["hasInterpretation"])
            )
            for row in age_excluded_rows
        )
        age_excluded_safety_violations = sum(
            row["ageApplicability"].get("requiresPositiveFiniteValue")
            and (not isinstance(row["value"], (int, float)) or not isfinite(row["value"]) or row["value"] <= 0)
            for row in age_excluded_rows
        )
        distributions[loinc] = {
            "count": len(values),
            "ordinaryCount": len(ordinary_rows),
            "applicableOrdinaryCount": len(applicable_rows),
            "ageExcludedOrdinaryCount": len(age_excluded_rows),
            "ageApplicabilityCounts": dict(sorted(Counter(row["ageApplicability"].get("id", "unknown") for row in ordinary_rows).items())),
            "unitCodes": sorted({row["unit"] for row in rows}),
            "ordinaryUnitCodes": sorted({row["unit"] for row in ordinary_rows}),
            "ordinaryReferenceRanges": sorted(
                {(row["referenceLow"], row["referenceHigh"]) for row in applicable_rows},
                key=lambda pair: tuple(float("-inf") if value is None else value for value in pair),
            ),
            "ordinaryStatuses": sorted({row["status"] for row in ordinary_rows}),
            "ordinaryInterpretationRate": round(sum(row["hasInterpretation"] for row in applicable_rows) / len(applicable_rows), 6) if applicable_rows else None,
            "ordinaryEventTimeRate": round(sum(row["hasEventTime"] for row in ordinary_rows) / len(ordinary_rows), 6) if ordinary_rows else None,
            "min": min(values), "p05": _percentile(values, .05), "median": _percentile(values, .5),
            "p95": _percentile(values, .95), "max": max(values),
            "ordinaryMin": min(ordinary) if ordinary else None, "ordinaryMax": max(ordinary) if ordinary else None,
            "ageExcludedOrdinaryMin": min(age_excluded_values) if age_excluded_values else None,
            "ageExcludedOrdinaryMax": max(age_excluded_values) if age_excluded_values else None,
            "ordinaryInRangeRate": round(len(in_range) / len(ordinary), 6) if ordinary else None,
            "ordinaryAbnormalRate": round(1 - len(in_range) / len(ordinary), 6) if ordinary else None,
            "conditionSpecificCount": sum(row.get("supportedRenal", False) for row in ordinary_rows),
            "ordinaryInterpretationMix": dict(sorted(Counter(code for row in applicable_rows for code in row["interpretations"]).items())),
            "problemTagRate": round(sum(row["problemTagged"] for row in rows) / len(rows), 6),
            "unexplainedExtremeCount": len(unexplained_extremes),
            "ageExcludedReferenceMetadataViolationCount": age_excluded_metadata_violations,
            "ageExcludedDerivedSafetyViolationCount": age_excluded_safety_violations,
            "unknownAgeCount": sum(row["ageApplicability"].get("ageResolutionRequired", False) for row in ordinary_rows),
        }
    anchor = date.fromisoformat(spec["timelineAnchor"][:10])
    pediatric = [patient for patient in crosswalk["patients"] if patient["primaryArchetype"] == "pediatric"]
    pediatric_ages = []
    for row in pediatric:
        birth = patient_birth_dates.get(row["cohortId"]) or birth_date_or_none(hmo_patients[row["cohortId"]])
        if birth is not None:
            pediatric_ages.append(completed_age_years(birth, anchor))
    name_combinations = {
        (resource["name"][0]["given"][0], resource["name"][0]["family"])
        for resource in hmo_patients.values()
    }
    mammography_policy_violations = []
    for cohort in mammography_cohorts:
        birth = patient_birth_dates.get(cohort) or birth_date_or_none(hmo_patients[cohort])
        age = completed_age_years(birth, anchor) if birth is not None else None
        if age is None or not 50 <= age <= 74:
            mammography_policy_violations.append({"cohortId": cohort, "age": age, "reason": "unresolved age" if age is None else "outside routine range without high-risk rationale"})
    geography_catalog = load_catalog(spec["geography"]["catalog"])
    locality_by_code = {row["code"]: row for row in geography_catalog["localities"]}
    locality_counts = Counter(row["geographyStratum"]["localityCode"] for row in crosswalk["patients"])
    district_weights = {
        district: sum(float(row["weight"]) for row in geography_catalog["localities"] if row["district"] == district)
        for district in geography_catalog.get("districtDistribution", {})
    }
    locality_fit = {
        code: {
            "actualCount": locality_counts[code],
            "actualRate": round(locality_counts[code] / len(crosswalk["patients"]), 6),
            "targetRate": round(
                geography_catalog.get("districtDistribution", {}).get(locality_by_code[code]["district"], 1.0)
                * float(locality_by_code[code]["weight"]) / district_weights.get(locality_by_code[code]["district"], 1.0), 6
            ),
            "absoluteDeviation": round(abs(
                locality_counts[code] / len(crosswalk["patients"])
                - geography_catalog.get("districtDistribution", {}).get(locality_by_code[code]["district"], 1.0)
                * float(locality_by_code[code]["weight"]) / district_weights.get(locality_by_code[code]["district"], 1.0)
            ), 6),
        }
        for code in sorted(locality_by_code)
    }
    district_counts = Counter(locality_by_code[code]["district"] for code in locality_counts.elements())
    district_fit = {
        district: {
            "actualCount": district_counts[district],
            "actualRate": round(district_counts[district] / len(crosswalk["patients"]), 6),
            "targetRate": rate,
            "absoluteDeviation": round(abs(district_counts[district] / len(crosswalk["patients"]) - rate), 6),
        }
        for district, rate in geography_catalog.get("districtDistribution", {}).items()
    }
    utilization = {
        cohort: {organization for organization in ("hmo", "hospital-a", "hospital-b") if source_graphs[cohort].get(organization)}
        for cohort in hmo_patients
    }
    encounter_summary = {
        organization: {
            "count": len(rows), "classCounts": dict(sorted(Counter(row["class"] for row in rows).items())),
            "minimumDurationHours": min((row["durationHours"] for row in rows if row["durationHours"] is not None), default=None),
            "maximumDurationHours": max((row["durationHours"] for row in rows if row["durationHours"] is not None), default=None),
            "distinctDurationCount": len({row["durationHours"] for row in rows if row["durationHours"] is not None}),
            "reasonCodeCounts": dict(sorted(Counter(code for row in rows for code in row["reasonCodes"]).items())),
        }
        for organization, rows in encounters.items()
    }
    return {
        "schemaVersion": "2.0.0",
        **({"realizedProblemAllocation": realized_problem_allocation(bundles, spec)} if spec.get("remediation") else {}),
        "archetypePatientCounts": dict(sorted(Counter(row["primaryArchetype"] for row in crosswalk["patients"]).items())),
        "longitudinality": {
            "everyPatientTwoDates": all(row["distinctClinicalDates"] >= 2 for row in longitudinal_rows),
            "patientsSpanning180Days": sum(row["spanDays"] >= 180 for row in longitudinal_rows),
            "chronicThreeDatesOneYear": all(row["distinctClinicalDates"] >= 3 and row["spanDays"] >= 365 for row in longitudinal_rows if row["primaryArchetype"] == "chronic-care"),
            "distinctNonZeroSpanCount": len({row["spanDays"] for row in longitudinal_rows if row["spanDays"] > 0}),
            "rows": longitudinal_rows,
        },
        "sourceDifferentiation": {
            "heldClinicalDataPatientCounts": {source: sum(source in sources for sources in utilization.values()) for source in bundles},
            "performedCarePatientCounts": {source: sum(source in sources for sources in performed_care.values()) for source in bundles},
            "comparedHospitalPairs": compared, "identicalHospitalPairs": identical,
            "identicalRate": round(identical / compared, 6) if compared else 0.0,
            "hmoOnlyPatientCount": sum(sources == {"hmo"} for sources in utilization.values()),
            "exactlyOneHospitalPatientCount": sum(len(sources & {"hospital-a", "hospital-b"}) == 1 for sources in utilization.values()),
            "patientsWithoutClinicalByHospital": {
                hospital: sum(hospital not in sources for sources in utilization.values())
                for hospital in ("hospital-a", "hospital-b")
            },
            "encounters": encounter_summary,
        },
        "archetypeContent": ({
            "status": "superseded-by-r29-tag-stripped-executable-grammar",
            "failureCount": 0,
            "failures": [],
            "legacyPresenceOnlyFailureCount": len(archetype_content_failures),
            "legacyPresenceOnlyFailures": archetype_content_failures,
        } if spec.get("clinicalRealism") else {
            "status": "active",
            "failureCount": len(archetype_content_failures),
            "failures": archetype_content_failures,
        }),
        "clinicalDistributions": distributions,
        "branchQuantitativeFindings": population_branch_quantitative_findings(bundles, content),
        "demographics": {
            **({"lifecycle": build_lifecycle_audit(bundles, spec)} if spec["demographics"].get("lifecyclePolicyCatalog") else {}),
            "distinctHmoFullNames": len(name_combinations),
            "genderCounts": dict(sorted(Counter(resource["gender"] for resource in hmo_patients.values()).items())),
            "pediatricCount": len(pediatric),
            "infantCount": sum(age < 1 for age in pediatric_ages),
            "childCount": sum(1 <= age < 13 for age in pediatric_ages),
            "adolescentCount": sum(13 <= age < 18 for age in pediatric_ages),
            "preventiveCare": {"mammographyCount": len(mammography_cohorts), "policyViolations": mammography_policy_violations},
        },
        "geography": {
            "districtCounts": dict(sorted(district_counts.items())),
            "districtFit": district_fit,
            "localityCount": len(locality_counts),
            "localityClassCounts": dict(sorted(Counter(locality_by_code[code]["class"] for code in locality_counts.elements()).items())),
            "localityFit": locality_fit,
            "maximumAbsoluteWeightDeviation": max(row["absoluteDeviation"] for row in locality_fit.values()),
        },
    }


def metric_failures(metrics: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    if spec.get("schemaVersion") != "2.0.0":
        return []
    errors = []
    if spec.get("remediation"):
        realized = metrics.get("realizedProblemAllocation", {})
        for key in ("patientsWithOne", "patientsWithSecond", "crossSourceClinicalPatients"):
            if realized.get(key) != spec["problems"][key]:
                errors.append(f"realized problem allocation differs: {key}")
        if realized.get("problemIdCount") != spec["problems"]["patientsWithOne"] + spec["problems"]["patientsWithSecond"]:
            errors.append("realized problem ID total differs")
    lifecycle = metrics["demographics"].get("lifecycle")
    if spec["demographics"].get("lifecyclePolicyCatalog") and (not lifecycle or lifecycle["status"] != "pass"):
        errors.append("demographic lifecycle audit failed: " + "; ".join((lifecycle or {}).get("failures", [])[:5]))
    long = metrics["longitudinality"]
    if not long["everyPatientTwoDates"]:
        errors.append("not every patient has two clinical dates")
    minimum_180 = min(spec["populationSize"], round(spec["populationSize"] * .6))
    if long["patientsSpanning180Days"] < minimum_180:
        errors.append("fewer than 60% of patients span 180 days")
    if not long["chronicThreeDatesOneYear"]:
        errors.append("chronic-care longitudinal gate failed")
    source = metrics["sourceDifferentiation"]
    # v4 replaces this lossy legacy graph comparison with the explicit
    # tag-blind, multi-grain variability audit. The old signature discards
    # capture disposition and therefore treats two honest source views as
    # identical.
    if not spec.get("remediation") and source["identicalRate"] > .05:
        errors.append("normalized Hospital A/Hospital B graph identity exceeds 5%")
    if spec["populationSize"] >= 100:
        if source["hmoOnlyPatientCount"] / spec["populationSize"] < .20:
            errors.append("fewer than 20% of canonical persons have HMO-only clinical history")
        if source["exactlyOneHospitalPatientCount"] / spec["populationSize"] < .10:
            errors.append("fewer than 10% of canonical persons use exactly one hospital")
        if any(count / spec["populationSize"] < .20 for count in source["patientsWithoutClinicalByHospital"].values()):
            errors.append("a hospital has fewer than 20% demographics-only persons")
        for hospital in ("hospital-a", "hospital-b"):
            classes = source["encounters"].get(hospital, {}).get("classCounts", {})
            if classes and set(classes) == {"IMP"}:
                errors.append(f"{hospital} classifies every encounter as IMP")
    if metrics["archetypeContent"]["failureCount"]:
        errors.append("archetype event-grammar content gate failed")
    errors.extend(row["reason"] for row in metrics.get("branchQuantitativeFindings", []))
    for code, row in metrics["clinicalDistributions"].items():
        guard = _distribution_guards(spec).get(code) if spec.get("schemaVersion") == "2.0.0" else None
        if not guard:
            errors.append(f"quantitative observation {code} has no reviewed distribution guardrail")
            continue
        applicable_count = row.get("applicableOrdinaryCount", row["ordinaryCount"])
        if row["ordinaryCount"] and row["ordinaryUnitCodes"] != [guard["unit"]]:
            errors.append(f"clinical distribution {code} has incompatible ordinary units")
        if row["ordinaryCount"] and guard.get("allowedStatuses") and any(status not in guard["allowedStatuses"] for status in row["ordinaryStatuses"]):
            errors.append(f"clinical distribution {code} has an unreviewed status")
        if row["ordinaryCount"] and guard.get("requiresEventTime") and row["ordinaryEventTimeRate"] != 1.0:
            errors.append(f"clinical distribution {code} lacks ordinary event timing")
        if applicable_count:
            if row["ordinaryReferenceRanges"] != [[guard["referenceLow"], guard["referenceHigh"]]] and row["ordinaryReferenceRanges"] != [(guard["referenceLow"], guard["referenceHigh"])]:
                errors.append(f"clinical distribution {code} has incompatible ordinary reference ranges")
        # Rate bands are population-distribution gates, not tiny-pilot gates.
        # Small samples still enforce units, bounds, status and interpretation.
        if applicable_count >= 100 and "maximumOrdinaryAbnormalRate" in guard and row["ordinaryInRangeRate"] is not None and row["ordinaryInRangeRate"] < 1 - guard["maximumOrdinaryAbnormalRate"]:
            errors.append(f"clinical distribution {code} exceeds ordinary abnormal-rate guard")
        if applicable_count >= 500 and row["ordinaryAbnormalRate"] < guard.get("minimumOrdinaryAbnormalRate", 0):
            errors.append(f"clinical distribution {code} is implausibly all-normal")
        if applicable_count and (row["ordinaryMin"] < guard.get("abnormalLow", guard["ordinaryMin"]) or row["ordinaryMax"] > guard.get("abnormalHigh", guard["ordinaryMax"])):
            errors.append(f"clinical distribution {code} exceeds reviewed ordinary bounds")
        if applicable_count and guard.get("requiresInterpretation") and row["ordinaryInterpretationRate"] != 1.0:
            errors.append(f"clinical distribution {code} lacks ordinary interpretation")
        if applicable_count and row["unexplainedExtremeCount"]:
            errors.append(f"clinical distribution {code} has unexplained extremes")
        if row.get("ageExcludedReferenceMetadataViolationCount", 0):
            errors.append(f"clinical distribution {code} has age-inapplicable reference metadata")
        if row.get("ageExcludedDerivedSafetyViolationCount", 0):
            errors.append(f"clinical distribution {code} has an invalid age-excluded derived value")
        if row.get("unknownAgeCount", 0):
            errors.append(f"clinical distribution {code} has unresolved age applicability")
    if spec["populationSize"] == 1000:
        if long["distinctNonZeroSpanCount"] < 100:
            errors.append("fewer than 100 distinct non-zero combined-history spans")
        expected = {row["id"]: row["count"] for row in load_catalog(spec["archetypes"]["catalog"])["primaryDistribution"]}
        if metrics["archetypePatientCounts"] != dict(sorted(expected.items())):
            errors.append("canonical archetype counts differ from the catalog")
        demographics = metrics["demographics"]
        if demographics["distinctHmoFullNames"] < 250:
            errors.append("fewer than 250 HMO full-name combinations")
        if demographics["pediatricCount"] != 150 or demographics["infantCount"] < 1:
            errors.append("canonical pediatric/infant gate failed")
        if demographics["preventiveCare"]["policyViolations"]:
            errors.append("mammography age/risk policy violation")
        expected_female = round(spec["populationSize"] * spec["demographics"]["genderDistribution"]["female"])
        if demographics["genderCounts"].get("female") != expected_female:
            errors.append("canonical sex distribution drifted after joint archetype assignment")
        distinct_spans = len({row["spanDays"] for row in long["rows"] if row["spanDays"] > 0})
        if distinct_spans < 100:
            errors.append("fewer than 100 distinct non-zero combined-history spans")
        geography = metrics["geography"]
        if set(geography["districtCounts"]) != set(spec["geography"]["requiredDistricts"]):
            errors.append("canonical geography does not cover every required district")
        if set(geography["localityClassCounts"]) != {"urban", "peripheral", "rural"}:
            errors.append("canonical geography does not cover urban/peripheral/rural strata")
        if geography["localityCount"] < 30:
            errors.append("canonical geography uses fewer than 30 localities")
        if max(row["absoluteDeviation"] for row in geography["districtFit"].values()) > .005:
            errors.append("canonical district shares exceed seeded tolerance")
    return errors
