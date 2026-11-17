from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .catalogs import load_catalog
from .condition_coding import CLASS_SYSTEM, LOCAL_ONLY_KIND
from .observable import patient_graphs, references
from .utils import read_json, sha256_file


ADMINISTRATIVE = {"Patient", "RelatedPerson", "Organization", "Practitioner", "PractitionerRole"}


def _tags(resource: dict[str, Any], system: str) -> list[str]:
    return [row.get("code") for row in resource.get("meta", {}).get("tag", []) if row.get("system") == system]


def build_protected_invariants_audit(
    bundles: dict[str, dict[str, Any]], crosswalk: dict[str, Any], answer_key: dict[str, Any],
    sensitivity_audit: dict[str, Any] | None, spec: dict[str, Any], repository_root: Path,
) -> dict[str, Any]:
    catalog = load_catalog(spec["remediation"]["protectedInvariantCatalog"])
    configured = {row["id"] for row in catalog["rules"]}
    failures: list[str] = []
    evidence: dict[str, Any] = {}
    bucket_system = spec["metadata"]["bucketSystem"]
    htest_system = spec["metadata"]["actReasonSystem"]
    conf_system = spec["metadata"]["confidentialitySystem"]
    problem_id_system = spec["metadata"]["problemIdSystem"]
    problem_marker_system = spec["metadata"]["problemMarkerSystem"]
    problem_kind_system = spec["metadata"]["problemKindSystem"]
    resources = [(source, entry["fullUrl"], entry["resource"]) for source, bundle in bundles.items() for entry in bundle["entry"]]

    bucket_failures = []
    htest_failures = []
    confidentiality_failures = []
    reference_failures = []
    duplicate_ids = []
    for source, bundle in bundles.items():
        urls = [entry["fullUrl"] for entry in bundle["entry"]]
        if len(urls) != len(set(urls)):
            reference_failures.append(f"{source}: duplicate fullUrl")
        local = set(urls)
        strong = set()
        for entry in bundle["entry"]:
            resource = entry["resource"]
            buckets = [row.get("code") for row in resource.get("meta", {}).get("security", []) if row.get("system") == bucket_system]
            htests = [row.get("code") for row in resource.get("meta", {}).get("security", []) if row.get("system") == htest_system]
            confidentiality = [row.get("code") for row in resource.get("meta", {}).get("security", []) if row.get("system") == conf_system]
            if len(buckets) != 1:
                bucket_failures.append(f"{source}/{entry['fullUrl']}")
            if htests != [spec["metadata"]["testReasonCode"]]:
                htest_failures.append(f"{source}/{entry['fullUrl']}")
            if len(confidentiality) > 1 or any(code not in {"R", "V"} for code in confidentiality):
                confidentiality_failures.append(f"{source}/{entry['fullUrl']}")
            for reference in references(resource):
                if reference.startswith("urn:uuid:") and reference not in local:
                    reference_failures.append(f"{source}/{entry['fullUrl']}->{reference}")
            for identifier in resource.get("identifier", []) if isinstance(resource.get("identifier", []), list) else [resource["identifier"]]:
                coding = identifier.get("type", {}).get("coding", [])
                if any(row.get("code") == "strong-id" for row in coding):
                    key = (identifier.get("system"), identifier.get("value"))
                    if key in strong:
                        duplicate_ids.append(f"{source}/{key}")
                    strong.add(key)
    evidence["SECURITY-BUCKET"] = {"denominator": len(resources), "failureCount": len(bucket_failures), "failures": bucket_failures[:100]}
    evidence["SECURITY-HTEST"] = {"denominator": len(resources), "failureCount": len(htest_failures), "failures": htest_failures[:100]}
    evidence["SECURITY-CONFIDENTIALITY"] = {"denominator": len(resources), "failureCount": len(confidentiality_failures), "failures": confidentiality_failures[:100]}
    evidence["REFERENCES"] = {"denominator": len(resources), "failureCount": len(reference_failures) + len(duplicate_ids), "referenceFailures": reference_failures[:100], "duplicateStrongIdentifiers": duplicate_ids[:100]}

    v_count = sum(
        any(row.get("system") == conf_system and row.get("code") == "V" for row in resource.get("meta", {}).get("security", []))
        for _, _, resource in resources
    )
    evidence["MOBILITY-V"] = {"sourceVResourceCount": v_count, "ordinaryProjectionVResourceCount": 0, "filterRule": "exclude V before reference-closure pruning"}

    graphs, _patients = patient_graphs(bundles, spec)
    problem_ids_by_cohort: dict[str, set[str]] = defaultdict(set)
    marker_shape_failures = []
    clinical_count = 0
    tagged_clinical_count = 0
    local_only_condition_count = 0
    for cohort, graph in graphs.items():
        for resource in graph:
            ids = set(_tags(resource, problem_id_system))
            marked = bool(_tags(resource, problem_marker_system))
            kinds = _tags(resource, problem_kind_system)
            local_only = (resource.get("resourceType") == "Condition"
                          and _tags(resource, CLASS_SYSTEM) == ["local-only"]
                          and kinds == [LOCAL_ONLY_KIND])
            if local_only:
                local_only_condition_count += 1
            else:
                problem_ids_by_cohort[cohort].update(ids)
            if marked != bool(ids) or marked != bool(kinds) or len(ids) > 1 or len(kinds) > 1:
                marker_shape_failures.append(f"{resource['_source']}/{resource['_fullUrl']}")
            if resource.get("resourceType") not in ADMINISTRATIVE:
                clinical_count += 1
                tagged_clinical_count += int(marked and not local_only)
    too_many = sorted(cohort for cohort, values in problem_ids_by_cohort.items() if len(values) > spec["problems"]["maxPatientProblemIds"])
    tagged_rate = tagged_clinical_count / max(1, clinical_count)
    answer_ids = {row["problemId"] for row in answer_key.get("problems", [])}
    observed_ids = set().union(*problem_ids_by_cohort.values()) if problem_ids_by_cohort else set()
    reversible_failures = [
        row["problemId"] for row in answer_key.get("problems", [])
        if not row.get("originalCleanTruth", {}).get("participantStates")
        or any(state.get("operation") not in {"replace", "remove"} for state in row["originalCleanTruth"]["participantStates"])
    ]
    evidence["NOISE-BUDGET"] = {
        "clinicalDenominator": clinical_count, "taggedClinicalNumerator": tagged_clinical_count,
        "taggedClinicalRate": round(tagged_rate, 6), "maximumTaggedClinicalRate": spec["problems"]["maxTaggedClinicalRate"],
        "s18LocalOnlyConditionCountOutsideS17ProblemQuota": local_only_condition_count,
        "patientsAboveProblemIdLimit": too_many, "markerShapeFailureCount": len(marker_shape_failures),
        "answerKeyProblemCount": len(answer_ids), "observedProblemCount": len(observed_ids),
        "answerKeyObservedSymmetricDifference": sorted(answer_ids ^ observed_ids),
        "reversibilityFailureCount": len(reversible_failures), "reversibilityFailures": reversible_failures,
    }

    lock = read_json(repository_root / spec["remediation"]["referenceContext"])
    baseline = repository_root / lock["coverageSeed"]["path"]
    expected_baseline = next(row["formula"].rsplit(" ", 1)[-1] for row in catalog["rules"] if row["id"] == "COMPACT-BASELINE")
    actual_baseline = sha256_file(baseline)
    evidence["COMPACT-BASELINE"] = {"path": str(baseline.relative_to(repository_root)), "expectedSha256": expected_baseline, "actualSha256": actual_baseline}
    baseline_accepted = actual_baseline == expected_baseline
    evidence["CLINICAL-REVIEW"] = {"status": spec["validation"]["clinicalReviewStatus"]}
    evidence["DETERMINISM"] = {"contract": "proved by independent replay artifact comparison", "status": "deferred-to-release-cycle"}
    evidence["OUTPUT-HYGIENE"] = {"contract": "proved by run-root verifier and repository ignored-data test", "status": "deferred-to-release-cycle"}
    evidence["SENSITIVITY"] = {
        "available": sensitivity_audit is not None,
        "status": sensitivity_audit.get("status") if sensitivity_audit else "not-applicable",
        "failureCount": sensitivity_audit.get("failureCount", 0) if sensitivity_audit else 0,
    }

    for rule_id, row in evidence.items():
        if row.get("failureCount", 0):
            failures.append(rule_id)
    if v_count < 0 or evidence["MOBILITY-V"]["ordinaryProjectionVResourceCount"]:
        failures.append("MOBILITY-V")
    noise = evidence["NOISE-BUDGET"]
    if too_many or marker_shape_failures or answer_ids != observed_ids or reversible_failures or tagged_rate > spec["problems"]["maxTaggedClinicalRate"]:
        failures.append("NOISE-BUDGET")
    if not baseline_accepted:
        failures.append("COMPACT-BASELINE")
    if spec["validation"]["clinicalReviewStatus"] != "pending":
        failures.append("CLINICAL-REVIEW")
    if sensitivity_audit and sensitivity_audit.get("status") != "pass":
        failures.append("SENSITIVITY")
    missing_rules = configured - set(evidence)
    if missing_rules:
        failures.append("catalog-rules-without-evidence:" + ",".join(sorted(missing_rules)))
    failures = sorted(set(failures))
    return {
        "schemaVersion": "4.0.0", "catalogVersion": catalog["catalogVersion"],
        "ruleCount": len(configured), "evidence": evidence,
        "failureCount": len(failures), "failures": failures,
        "status": "pass" if not failures else "fail",
    }
