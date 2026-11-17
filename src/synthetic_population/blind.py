from __future__ import annotations

import copy
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable

from .detectors import (
    ANALYTE_DISCORDANCE,
    _codes,
    detect_group,
    detects,
    duplicate_linkage_score,
    blocked_duplicate_pairs,
    normalized_quantity,
    stale_snapshot_evidence,
    quality_variant_evidence,
)
from .predicates import evaluate, evaluate_variant
from .problems import DATA_QUALITY_KINDS, SCENARIO_VARIANTS, additional_fixture_conclusions, validate_fixture_conclusions


def _references(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        if isinstance(value.get("reference"), str):
            yield value["reference"]
        for child in value.values():
            yield from _references(child)
    elif isinstance(value, list):
        for child in value:
            yield from _references(child)


def blind_resource(resource: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(resource)
    copied.get("meta", {})["tag"] = [
        tag for tag in copied.get("meta", {}).get("tag", [])
        if not str(tag.get("system", "")).startswith("urn:hdp-test-bundle:synthetic")
    ]

    def scrub(value: Any) -> None:
        if isinstance(value, dict):
            identifiers = value.get("identifier")
            if isinstance(identifiers, list):
                value["identifier"] = [row for row in identifiers if row.get("system") not in {
                    spec["identity"]["cohortIdentifierSystem"], "urn:hdp-test-bundle:synthetic:linkage-key",
                }]
            elif isinstance(identifiers, dict) and identifiers.get("system") in {
                spec["identity"]["cohortIdentifierSystem"], "urn:hdp-test-bundle:synthetic:linkage-key",
            }:
                value.pop("identifier")
            for child in list(value.values()):
                scrub(child)
        elif isinstance(value, list):
            for child in value:
                scrub(child)

    scrub(copied)
    return copied


def _identity_key(patient: dict[str, Any], spec: dict[str, Any] | None = None) -> tuple[str, ...]:
    name = patient.get("name", [{}])[0]
    normalize = lambda value: re.sub(r"[^a-z0-9]", "", str(value).lower())
    return (normalize(name.get("family", "")), normalize("".join(name.get("given", []))).rstrip("h"),
            patient.get("birthDate", ""), patient.get("gender", ""))


def detect_corpus(bundles: dict[str, dict[str, Any]], spec: dict[str, Any], *, work_stats: dict | None = None) -> list[dict[str, Any]]:
    """Discover candidates from the complete tag-stripped corpus without truth inputs."""
    rows = []
    patients_by_org_ref: dict[tuple[str, str], tuple[str, str, str]] = {}
    patients_by_identity: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for organization, bundle in bundles.items():
        for entry in bundle["entry"]:
            resource = blind_resource(entry["resource"], spec)
            row = {"organization": organization, "fullUrl": entry["fullUrl"], "resource": resource}
            rows.append(row)
            if resource.get("resourceType") == "Patient":
                key = _identity_key(resource, spec)
                if not all(key):
                    key = (*key, organization, entry["fullUrl"])
                patients_by_org_ref[(organization, entry["fullUrl"])] = key
                patients_by_identity[key].append(row)
    uncertain_linkage = []
    for key, group in list(patients_by_identity.items()):
        source_counts = defaultdict(int)
        for row in group:
            source_counts[row["organization"]] += 1
        if max(source_counts.values(), default=0) > 1:
            uncertain_linkage.append({"kind": "patient-linkage", "conclusionLevel": "uncertain-linkage", "patientKey": list(key), "resourceFullUrls": sorted(row["fullUrl"] for row in group), "reason": "Multiple source records share the independent identity features; clinical histories remain separate."})
            del patients_by_identity[key]
            for row in group:
                local_key = (*key, row["organization"], row["fullUrl"])
                patients_by_identity[local_key] = [row]
                patients_by_org_ref[(row["organization"], row["fullUrl"])] = local_key
    clinical_by_identity: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["resource"].get("resourceType") == "Patient":
            continue
        keys = {
            patients_by_org_ref[(row["organization"], reference)]
            for reference in _references(row["resource"])
            if (row["organization"], reference) in patients_by_org_ref
        }
        for key in keys:
            clinical_by_identity[key].append(row)

    candidates: list[dict[str, Any]] = list(uncertain_linkage)
    for row in rows:
        resource = row["resource"]
        for kind in ("semantically-wrong-code",):
            if detects(resource, kind):
                candidates.append({"kind": kind, "variant": {"119297000": "specimen-as-diagnosis", "368208006": "body-site-as-diagnosis", "396550006": "procedure-as-diagnosis"}.get(next((code for system, code in _codes(resource.get("code")) if system == "http://snomed.info/sct"), "")), "patientKey": None, "resourceFullUrls": [row["fullUrl"]]})

    for key, patient_rows in patients_by_identity.items():
        clinical_rows = clinical_by_identity.get(key, [])
        resources = [row["resource"] for row in clinical_rows]
        observable = [{**row["resource"], "_fullUrl": row["fullUrl"]} for row in clinical_rows]
        for row in clinical_rows:
            if row["resource"].get("resourceType") != "Observation":
                continue
            context = [row["resource"], *[patient["resource"] for patient in patient_rows]]
            for variant in sorted(quality_variant_evidence(context, "timestamp-order", anchor=spec["timelineAnchor"])):
                candidates.append({"kind": "timestamp-order", "variant": variant, "patientKey": list(key), "resourceFullUrls": [row["fullUrl"]]})
        for kind, types in (("conflicting-diagnosis", {"Condition"}), ("discordant-lab", {"Observation"}), ("unit-mismatch", {"Observation"})):
            relevant = [row for row in observable if row.get("resourceType") in types]
            for variant in sorted(quality_variant_evidence(relevant, kind)):
                candidates.append({"kind": kind, "variant": variant, "patientKey": list(key), "resourceFullUrls": sorted(row["_fullUrl"] for row in relevant)})

        for row in patient_rows:
            evidence = stale_snapshot_evidence(
                row["resource"],
                [other["resource"] for other in patient_rows if other is not row],
            )
            if evidence["stale"]:
                candidates.append({
                    "kind": "stale-demographics", "patientKey": list(key),
                    "resourceFullUrls": [row["fullUrl"]], "variant": "stale-" + next((field for field in ("address", "telecom", "name") if any(reason.startswith(field) for reason in evidence["reasons"])), "address") + "-snapshot", "staleEvidence": evidence["reasons"],
                })

        for scenario, variants in SCENARIO_VARIANTS.items():
            evidence = sorted(row["fullUrl"] for row in clinical_rows)
            if evaluate(scenario, resources):
                candidates.append({"kind": "cross-source-clinical-risk", "scenario": scenario, "conclusionLevel": "family-alert" if spec.get("remediation") else "legacy-family", "patientKey": list(key), "resourceFullUrls": evidence})
            for variant in variants if spec.get("remediation") else ():
                if evaluate_variant(scenario, variant, resources, require_three_source_context=True):
                    candidates.append({"kind": "cross-source-clinical-risk", "scenario": scenario, "variant": variant, "conclusionLevel": "strict-variant", "patientKey": list(key), "resourceFullUrls": evidence})
    # Stale-demographic discovery cannot depend on an exact phone-based identity
    # key: changed contact data is itself one of the intended positive variants.
    # Link source snapshots with independent name/birth/sex evidence, then compare
    # provenance timestamps and element periods/values.
    all_patient_rows = [row for values in patients_by_identity.values() for row in values]
    stale_link_index: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)

    def stale_link_key(patient: dict[str, Any]) -> tuple[str, str, str, str]:
        name = patient.get("name", [{}])[0]
        normalize = lambda value: re.sub(r"[^a-z0-9]", "", str(value).lower())
        return (
            normalize(name.get("family", "")),
            normalize("".join(name.get("given", []))).rstrip("h"),
            patient.get("birthDate", ""), patient.get("gender", ""),
        )

    for row in all_patient_rows:
        stale_link_index[stale_link_key(row["resource"])].append(row)
    existing_stale_urls = {
        row["resourceFullUrls"][0] for row in candidates if row["kind"] == "stale-demographics"
    }
    for row in all_patient_rows:
        compatible = [
            other["resource"] for other in stale_link_index[stale_link_key(row["resource"])]
            if other is not row and other["organization"] != row["organization"]
        ]
        evidence = stale_snapshot_evidence(row["resource"], compatible)
        if evidence["stale"] and row["fullUrl"] not in existing_stale_urls:
            candidates.append({
                "kind": "stale-demographics", "patientKey": None,
                "resourceFullUrls": [row["fullUrl"]], "variant": "stale-" + next((field for field in ("address", "telecom", "name") if any(reason.startswith(field) for reason in evidence["reasons"])), "address") + "-snapshot", "staleEvidence": evidence["reasons"],
            })
            existing_stale_urls.add(row["fullUrl"])
    # Duplicate discovery is source-local and uses observable weighted linkage;
    # it is intentionally independent of the exact cross-source identity key.
    for organization in sorted(bundles):
        source_patients = [row for rows in patients_by_identity.values() for row in rows if row["organization"] == organization]
        for left, right, score, evidence in blocked_duplicate_pairs([row["resource"] for row in source_patients], work_stats):
            candidates.append({
                "kind": "probable-duplicate", "variant": "minor-dob-discrepancy" if source_patients[left]["resource"].get("birthDate") != source_patients[right]["resource"].get("birthDate") else "changed-phone" if not evidence["phone"] else "transliteration-spelling", "patientKey": None,
                "linkageScore": score, "linkageEvidence": evidence,
                "resourceFullUrls": sorted([source_patients[left]["fullUrl"], source_patients[right]["fullUrl"]]),
            })
    return sorted(candidates, key=lambda row: (row["kind"], row.get("scenario", ""), row["resourceFullUrls"]))


def evaluate_candidates(candidates: list[dict[str, Any]], answer_key: dict[str, Any], scenario_evidence: dict[str, Any], *, work_stats: dict | None = None) -> dict[str, Any]:
    broad_alerts = [row for row in candidates if row.get("conclusionLevel") == "family-alert"]
    uncertain_linkage = [row for row in candidates if row.get("conclusionLevel") == "uncertain-linkage"]
    candidates = [row for row in candidates if row.get("conclusionLevel") not in {"family-alert", "uncertain-linkage"}]
    expected_conclusions = []
    for problem in answer_key["problems"]:
        if "expectedConclusions" in problem:
            validate_fixture_conclusions(problem["variant"], problem["expectedConclusions"],
                                         problem.get("conditionalExpectedConclusions", []))
        for variant in problem.get("expectedConclusions", [problem.get("variant")]):
            expected_conclusions.append({**problem, **({"variant": variant} if variant else {})})
    for control in scenario_evidence.get("scenarios", []):
        if control.get("matchedControl"):
            if "additionalExpectedConclusions" in control:
                validate_fixture_conclusions(control["structuralVariant"], control["additionalExpectedConclusions"],
                                             control.get("conditionalExpectedConclusions", []), control=True)
            for variant in control.get("additionalExpectedConclusions", []):
                expected_conclusions.append({"problemId": control["scenarioId"] + ":" + variant, "kind": "cross-source-clinical-risk", "scenario": control["name"], "variant": variant, "participants": [{"fullUrl": url} for url in control["aggregateEvidence"]], "expectedControlConclusion": True})
    results = {}
    variant_results = {}
    for kind in sorted(set(DATA_QUALITY_KINDS) | {"cross-source-clinical-risk"} | {row["kind"] for row in answer_key["problems"]} | {row["kind"] for row in candidates}):
        expected = [row for row in expected_conclusions if row["kind"] == kind]
        actual = [row for row in candidates if row["kind"] == kind]
        # Deterministic maximum one-to-one matching. A candidate is consumable by
        # exactly one truth row; this prevents precision > 1 when participant sets
        # overlap or one broad candidate is a superset of several expected cases.
        adjacency: list[list[int]] = []
        for problem in expected:
            expected_urls = {row["fullUrl"] for row in problem["participants"]}
            ranked = []
            for position, row in enumerate(actual):
                actual_urls = set(row["resourceFullUrls"])
                if (expected_urls.issubset(actual_urls)
                    and (not problem.get("scenario") or row.get("scenario") == problem["scenario"])
                    and (not problem.get("variant") or row.get("conclusionLevel") == "legacy-family" or problem["variant"] == row.get("variant"))):
                    surplus = len(actual_urls - expected_urls)
                    ranked.append((surplus, position))
            adjacency.append([position for _surplus, position in sorted(ranked)])

        candidate_to_expected: dict[int, int] = {}

        def augment(expected_index: int, seen: set[int]) -> bool:
            for candidate_index in adjacency[expected_index]:
                if candidate_index in seen:
                    continue
                seen.add(candidate_index)
                previous = candidate_to_expected.get(candidate_index)
                if previous is None or augment(previous, seen):
                    candidate_to_expected[candidate_index] = expected_index
                    return True
            return False

        for expected_index in range(len(expected)):
            augment(expected_index, set())
        matched_expected_indexes = set(candidate_to_expected.values())
        matched_candidate_indexes = set(candidate_to_expected)
        true_positives = len(matched_candidate_indexes)
        false_negatives = [
            problem["problemId"] for position, problem in enumerate(expected)
            if position not in matched_expected_indexes
        ]
        false_positive_indexes = [position for position in range(len(actual)) if position not in matched_candidate_indexes]
        recall = true_positives / len(expected) if expected else 1.0
        precision = true_positives / len(actual) if actual else (1.0 if not expected else 0.0)
        if not (0.0 <= precision <= 1.0 and 0.0 <= recall <= 1.0):
            raise AssertionError("blind-evaluation metrics escaped [0,1]")
        variants = {row.get("variant", "unspecified") for row in expected + actual}
        if kind == "cross-source-clinical-risk":
            variants |= {variant for values in SCENARIO_VARIANTS.values() for variant in values}
        else:
            from .problems import PROBLEM_VARIANTS
            variants |= set(PROBLEM_VARIANTS.get(kind, ()))
        for variant in sorted(variants):
            expected_indexes = {i for i, row in enumerate(expected) if row.get("variant", "unspecified") == variant}
            actual_indexes = {i for i, row in enumerate(actual) if row.get("variant", "unspecified") == variant}
            tp = len(expected_indexes & matched_expected_indexes)
            fp = len(actual_indexes - matched_candidate_indexes)
            fn = len(expected_indexes - matched_expected_indexes)
            variant_results[f"{kind}:{variant}"] = {"expectedCount": len(expected_indexes), "candidateCount": len(actual_indexes), "truePositiveCount": tp, "falsePositiveCount": fp, "falseNegativeCount": fn,
                "precision": round(tp / (tp + fp), 6) if tp + fp else (0.0 if fn else 1.0), "recall": round(tp / (tp + fn), 6) if tp + fn else 1.0}
        results[kind] = {
            "expectedCount": len(expected), "candidateCount": len(actual), "truePositiveCount": true_positives,
            "falsePositiveCount": len(false_positive_indexes), "falseNegativeCount": len(false_negatives),
            "precision": round(precision, 6), "recall": round(recall, 6),
            "falseNegativeProblemIds": false_negatives,
            "incidentalCandidateIndexes": false_positive_indexes,
        }
    controls = [row for row in scenario_evidence.get("scenarios", []) if row.get("matchedControl")]
    cross_candidates = [row for row in candidates if row["kind"] == "cross-source-clinical-risk"]
    control_false_positives = [
        control["scenarioId"] for control in controls
        if any(
            candidate.get("scenario") == control["name"]
            and (candidate.get("conclusionLevel") == "legacy-family" or candidate.get("variant") == control.get("structuralVariant"))
            and set(control.get("aggregateEvidence", [])).issubset(set(candidate["resourceFullUrls"]))
            for candidate in cross_candidates
        )
    ]
    unexplained = [row for kind, result in results.items() for index, row in enumerate([item for item in candidates if item["kind"] == kind]) if index in result["incidentalCandidateIndexes"]]
    totals = {name: sum(row[name] for row in results.values()) for name in ("truePositiveCount", "falsePositiveCount", "falseNegativeCount")}
    return {
        **totals,
        "zeroUnexplainedCandidates": totals["falsePositiveCount"] == 0, "unexplainedCandidates": unexplained,
        "broadFamilyAlertCount": len(broad_alerts), "broadFamilyAlerts": broad_alerts,
        "uncertainLinkageCount": len(uncertain_linkage), "uncertainLinkageCandidates": uncertain_linkage,
        "zeroDenominatorPolicy": "Recall is 1 with no expected cases; precision is 1 only with no candidates and no expected cases.",
        "schemaVersion": "2.1.0" if work_stats is not None else "2.0.0",
        **({"work": dict(work_stats)} if work_stats is not None else {}), "evaluationMode": "complete-tag-stripped-corpus",
        "candidateCount": len(candidates), "expectedProblemCount": len(answer_key["problems"]), "expectedConclusionCount": len(expected_conclusions), "evaluationUnit": "observable conclusion; explicitly declared overlapping fixture conclusions are scored separately", "byProblemKind": results, "byProblemVariant": variant_results,
        "allDeclaredRecallComplete": all(row["recall"] == 1.0 for row in results.values()),
        "namedNearMissControlCount": len(controls),
        "namedNearMissFalsePositiveCount": len(control_false_positives),
        "namedNearMissFalsePositiveScenarioIds": control_false_positives,
        "detectorTruthBoundary": "candidate generation receives no answer key, problem tags, cohort identifiers, linkage keys, or archetype tags",
        "candidates": candidates,
    }
