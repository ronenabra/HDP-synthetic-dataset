from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from .bundle import build_population
from .blind import detect_corpus, evaluate_candidates
from .buckets import CATALOG_PATH as BUCKET_CATALOG_PATH
from .rehydrate import require_references
from .metrics import build_quality_metrics, metric_failures
from .terminology import audit_terminology
from .sensitivity import realization_audit
from .realism import build_realism_audit
from .grammar import build_grammar_audit
from .story import build_story_coverage_audit
from .utils import (
    StorageError, assert_storage_safe, atomic_target, checksum_lines, directory_size,
    publish_atomic, read_json, sha256_file, sha256_json, write_json,
)
from .artifacts import bundle_relative_path, is_v4_spec
from .history import build_history_audit
from .typed_grammar import build_typed_grammar_audit
from .variability_v4 import build_variability_audit
from .latent_audit import build_latent_mapping_audit
from .scenario_audit import build_scenario_causality_audit
from .clinical_depth import build_clinical_depth_audit
from .variant_audit import build_variant_audit
from .protected import build_protected_invariants_audit




def _write_ndjson(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _validate_local_contracts(repository_root: Path, built: dict[str, Any], spec: dict) -> dict[str, Any]:
    if spec.get("remediation"):
        lock = read_json(repository_root / spec["remediation"]["referenceContext"])
        profile_directories = [repository_root / row["path"] for row in lock["profiles"]]
    else:
        profile_directories = [repository_root / "reference/ilcore", repository_root / "reference/hdp-il"]
    profile_urls = {
        json.loads(path.read_text(encoding="utf-8")).get("url")
        for directory in profile_directories
        for path in directory.glob("StructureDefinition-*.json")
    }
    declared = {
        profile
        for bundle in built["bundles"].values()
        for entry in bundle["entry"]
        for profile in entry["resource"].get("meta", {}).get("profile", [])
    }
    missing_profiles = sorted(profile for profile in declared if profile not in profile_urls and not profile.startswith("http://hl7.org/fhir/StructureDefinition/"))
    if missing_profiles:
        raise ValueError(f"declared profiles do not resolve in the pinned local trees: {missing_profiles}")
    bucket_artifact = next(
        path for directory in profile_directories
        for path in [directory / "CodeSystem-il-hdp-information-buckets.json"] if path.is_file()
    )
    bucket_resource = read_json(bucket_artifact)
    bucket_codes = {item["code"] for item in bucket_resource["concept"]}
    used_buckets = {
        coding["code"]
        for bundle in built["bundles"].values()
        for entry in bundle["entry"]
        for coding in entry["resource"].get("meta", {}).get("security", [])
        if coding.get("system") == spec["metadata"]["bucketSystem"]
    }
    unknown_buckets = sorted(used_buckets - bucket_codes)
    if unknown_buckets:
        raise ValueError(f"unknown active information-bucket codes: {unknown_buckets}")
    return {
        "resolvedDeclaredProfileCount": len(declared), "bucketCodeSystemSha256": sha256_file(bucket_artifact),
        "usedBucketCodes": sorted(used_buckets),
    }


def generate(spec: dict, out: Path, repository_root: Path, force: bool = False) -> Path:
    require_references(repository_root)
    return _materialize_population(spec, out, repository_root, build_population, force)


def _materialize_population(spec: dict, out: Path, repository_root: Path, builder,
                            force: bool = False) -> Path:
    """Shared audit/serialization path; builders must supply real input resources."""
    storage = assert_storage_safe(spec, out)
    target, temp = atomic_target(out, force)
    temp.mkdir(parents=True)
    write_json(temp / "run-state.json", {"status": "partial", "datasetVersion": spec["datasetVersion"]})
    try:
        built = builder(spec, repository_root, monitor=lambda: assert_storage_safe(spec, temp))
        all_output_bundles = dict(built["bundles"])
        all_output_bundles.update({f"coverage-{key}": value for key, value in built.get("coverageBundles", {}).items()})
        _validate_local_contracts(repository_root, {**built, "bundles": all_output_bundles}, spec)
        terminology_audit = audit_terminology(all_output_bundles, spec, repository_root)
        if terminology_audit["status"] != "pass":
            raise ValueError("terminology provenance failed: " + "; ".join(terminology_audit["errors"][:20]))
        condition_aggregate = None
        if built.get("conditionAllocation"):
            from .condition_aggregate import build_condition_aggregate
            condition_aggregate = build_condition_aggregate(
                built["bundles"], built["crosswalk"], built["latentTruthLedger"], spec, repository_root,
            )
            if condition_aggregate["status"] != "pass":
                raise ValueError("S18 Condition aggregate failed: " + "; ".join(condition_aggregate["errors"][:20]))
        detector_work = {}
        detector_evidence = evaluate_candidates(
            detect_corpus(built["bundles"], spec, work_stats=detector_work), built["answerKey"], built["scenarioEvidence"],
            work_stats=detector_work,
        )
        if not detector_evidence["allDeclaredRecallComplete"] or detector_evidence["namedNearMissFalsePositiveCount"] or not detector_evidence["zeroUnexplainedCandidates"]:
            raise ValueError("blind whole-corpus detector gates failed")
        write_json(temp / "dataset-spec.json", spec)
        for org_id, bundle in built["bundles"].items():
            write_json(temp / bundle_relative_path(spec, org_id), bundle)
        for org_id, bundle in built.get("coverageBundles", {}).items():
            write_json(temp / f"{org_id}.conformance-coverage.json", bundle)
        write_json(temp / "cohort-crosswalk.json", built["crosswalk"])
        write_json(temp / "problem-answer-key.json", built["answerKey"])
        write_json(temp / "scenario-evidence.json", built["scenarioEvidence"])
        if spec.get("remediation"):
            _write_ndjson(temp / "latent-truth-ledger.ndjson", built["latentTruthLedger"])
        if built.get("conditionAllocation"):
            write_json(temp / "condition-allocation.json", built["conditionAllocation"])
            write_json(temp / "condition-aggregate.json", condition_aggregate)
        write_json(temp / "terminology-audit.json", terminology_audit)
        write_json(temp / "challenge-detector-results.json", detector_evidence)
        sensitivity_accounting = realization_audit(built["bundles"], spec) if spec.get("schemaVersion") == "2.0.0" else None
        if sensitivity_accounting and sensitivity_accounting["status"] != "pass":
            raise ValueError("sensitivity realization failed: " + "; ".join(sensitivity_accounting["errors"]))
        if sensitivity_accounting:
            write_json(temp / "sensitivity-accounting.json", sensitivity_accounting)
        quality_metrics = build_quality_metrics(built["bundles"], built["crosswalk"], spec)
        quality_errors = metric_failures(quality_metrics, spec)
        quality_metrics["status"] = "pass" if not quality_errors else "fail"
        quality_metrics["diagnostics"] = quality_errors
        write_json(temp / "quality-metrics.json", quality_metrics)
        if quality_errors:
            raise ValueError("quality metrics failed: " + "; ".join(quality_errors))
        grammar_audit = build_grammar_audit(built["bundles"], built["crosswalk"], spec) if spec.get("clinicalRealism") else None
        if grammar_audit:
            write_json(temp / "archetype-grammar-audit.json", grammar_audit)
            if grammar_audit["status"] != "pass" and not spec.get("remediation"):
                failures = [f"{row['cohortId']}:{row['clause']}" for row in grammar_audit["failures"][:20]]
                raise ValueError("archetype grammar audit failed: " + "; ".join(failures))
        story_audit = build_story_coverage_audit(
            built["bundles"], built["crosswalk"], spec, built.get("coverageBundles", {})
        ) if spec.get("clinicalRealism") else None
        if story_audit:
            write_json(temp / "clinical-story-coverage-audit.json", story_audit)
            if story_audit["status"] != "pass":
                raise ValueError("clinical story/coverage separation failed: " + "; ".join(story_audit["diagnostics"]))
        realism_audit = build_realism_audit(built["bundles"], built["crosswalk"], spec) if spec.get("clinicalRealism") else None
        if realism_audit:
            write_json(temp / "clinical-realism-audit.json", realism_audit)
            if realism_audit["status"] != "pass" and not spec.get("remediation"):
                raise ValueError("clinical realism audit failed: " + "; ".join(realism_audit["diagnostics"]))
        history_audit = None
        typed_grammar_audit = None
        variability_audit = None
        latent_mapping_audit = None
        scenario_causality_audit = None
        clinical_depth_audit = None
        variant_audit = None
        protected_audit = None
        if spec.get("remediation"):
            history_audit = build_history_audit(
                built["bundles"], built["crosswalk"], spec, built["latentTruthLedger"],
            )
            typed_grammar_audit = build_typed_grammar_audit(
                built["bundles"], built["crosswalk"], spec,
                spec["remediation"]["pathwayTemplateCatalog"],
            )
            variability_audit = build_variability_audit(
                built["bundles"], built["crosswalk"], spec, built["latentTruthLedger"],
            )
            latent_mapping_audit = build_latent_mapping_audit(
                built["bundles"], built["crosswalk"], built["latentTruthLedger"], spec,
            )
            scenario_causality_audit = build_scenario_causality_audit(
                built["bundles"], built["crosswalk"], built["scenarioEvidence"], spec,
            )
            clinical_depth_audit = build_clinical_depth_audit(
                built["bundles"], built["crosswalk"], spec,
            )
            variant_audit = build_variant_audit(built["bundles"], built["crosswalk"], spec)
            protected_audit = build_protected_invariants_audit(
                built["bundles"], built["crosswalk"], built["answerKey"],
                sensitivity_accounting, spec, repository_root,
            )
            write_json(temp / "clinical-history-audit.json", history_audit)
            write_json(temp / "typed-grammar-audit.json", typed_grammar_audit)
            write_json(temp / "anti-template-audit.json", variability_audit)
            write_json(temp / "latent-mapping-audit.json", latent_mapping_audit)
            write_json(temp / "temporal-causality-audit.json", scenario_causality_audit)
            write_json(temp / "clinical-depth-audit.json", clinical_depth_audit)
            write_json(temp / "material-variant-audit.json", variant_audit)
            write_json(temp / "protected-invariants-audit.json", protected_audit)
            remediation_failures = [
                name for name, audit in (
                    ("clinical-history", history_audit),
                    ("typed-grammar", typed_grammar_audit),
                    ("anti-template", variability_audit),
                    ("latent-mapping", latent_mapping_audit),
                    ("temporal-causality", scenario_causality_audit),
                    ("clinical-depth", clinical_depth_audit),
                    ("material-variant", variant_audit),
                    ("protected-invariants", protected_audit),
                ) if audit["status"] != "pass"
            ]
            if remediation_failures:
                diagnostics = []
                for name, audit in (
                    ("clinical-history", history_audit), ("typed-grammar", typed_grammar_audit),
                    ("anti-template", variability_audit), ("latent-mapping", latent_mapping_audit),
                    ("temporal-causality", scenario_causality_audit), ("clinical-depth", clinical_depth_audit),
                    ("material-variant", variant_audit), ("protected-invariants", protected_audit),
                ):
                    if audit["status"] != "pass":
                        diagnostics.extend(f"{name}: {item}" for item in audit.get("failures", [])[:10])
                raise ValueError(
                    "v4 remediation audits failed: " + ", ".join(remediation_failures)
                    + ("; " + "; ".join(diagnostics) if diagnostics else "")
                )
        index = []
        type_counts: Counter[str] = Counter()
        bucket_counts: Counter[str] = Counter()
        confidentiality_counts: Counter[str] = Counter()
        sensitivity_counts: Counter[str] = Counter()
        profile_counts: Counter[str] = Counter()
        problem_kind_counts: Counter[str] = Counter()
        archetype_counts: Counter[str] = Counter()
        clinical_timestamps: list[str] = []
        for org_id, bundle in built["bundles"].items():
            for entry in bundle["entry"]:
                resource = entry["resource"]
                type_counts[resource["resourceType"]] += 1
                bucket_system = spec["metadata"]["bucketSystem"]
                conf_system = spec["metadata"]["confidentialitySystem"]
                bucket = next(item["code"] for item in resource["meta"]["security"] if item.get("system") == bucket_system)
                bucket_counts[bucket] += 1
                confidentiality = [item["code"] for item in resource["meta"]["security"] if item.get("system") == conf_system]
                confidentiality_counts.update(confidentiality or ["ordinary-unlabeled"])
                sensitivity_counts.update(item["code"] for item in resource["meta"]["security"] if item.get("system") == spec["metadata"]["clinicalSensitivitySystem"])
                profile_counts.update(resource["meta"].get("profile", []))
                problem_kind_counts.update(item["code"] for item in resource["meta"].get("tag", []) if item.get("system") == spec["metadata"]["problemKindSystem"])
                archetype = next((item["code"] for item in resource["meta"].get("tag", []) if item.get("system") == "urn:hdp-test-bundle:synthetic-archetype"), "strict-patient-clinical")
                archetype_counts[archetype] += 1
                for key in ("authoredOn", "recordedDate", "effectiveDateTime", "issued", "receivedTime", "occurrenceDateTime"):
                    if isinstance(resource.get(key), str):
                        clinical_timestamps.append(resource[key])
                clinical_timestamps.extend(value for value in resource.get("period", {}).values() if isinstance(value, str))
                raw_identifiers = resource.get("identifier", [])
                identifiers = raw_identifiers if isinstance(raw_identifiers, list) else [raw_identifiers]
                strong_identifier = next((identifier.get("value") for identifier in identifiers if identifier.get("type", {}).get("coding", [{}])[0].get("code") == "strong-id"), None)
                index.append({
                    "organization": org_id, "fullUrl": entry["fullUrl"], "resourceType": resource["resourceType"],
                    "strongIdentifier": strong_identifier, "bucket": bucket,
                    "profiles": resource["meta"].get("profile", []), "archetype": archetype,
                    "confidentiality": confidentiality, "sha256": sha256_json(resource),
                })
        index.sort(key=lambda row: (row["organization"], row["resourceType"], row["fullUrl"]))
        _write_ndjson(temp / "resource-index.ndjson", index)
        write_json(temp / "validation-summary.json", {
            "status": "not-run", "coverage": "none", "eligibleResources": len(index), "validatedResources": 0,
            "cacheReusedResources": 0, "bundleEnvelopes": 3, "validatedBundleEnvelopes": 0,
            "clinicalReviewStatus": "pending",
        })
        (temp / "validation-ledger.ndjson").write_text("", encoding="utf-8")
        catalog_paths = {
            "sensitivity": spec["sensitivity"]["catalog"],
            "terminology": spec["terminology"]["catalog"],
            "geography": spec["geography"]["catalog"],
        }
        if spec.get("schemaVersion") == "2.0.0":
            catalog_paths.update({
                "archetypes": spec["archetypes"]["catalog"],
                "bucketSemantics": BUCKET_CATALOG_PATH,
                "clinicalDistributions": spec["clinicalDistributions"]["catalog"],
                "identityPolicy": spec["identityPolicy"]["catalog"],
                "names": spec["demographics"]["namesCatalog"],
                **({"lifecyclePolicy": spec["demographics"]["lifecyclePolicyCatalog"]}
                   if spec["demographics"].get("lifecyclePolicyCatalog") else {}),
                "sourceModels": spec["sourceModels"]["catalog"],
            })
            if spec.get("clinicalRealism"):
                catalog_paths["clinicalRealism"] = spec["clinicalRealism"]["catalog"]
                realism_catalog = read_json(repository_root / spec["clinicalRealism"]["catalog"])
                for name, key in (
                    ("sourceCapture", "sourceCaptureCatalog"),
                    ("detectorSemantics", "detectorSemanticsCatalog"),
                ):
                    if realism_catalog.get(key):
                        catalog_paths[name] = realism_catalog[key]
            if spec.get("remediation"):
                catalog_paths.update({
                    "pathwayTemplates": spec["remediation"]["pathwayTemplateCatalog"],
                    "capturePolicies": spec["remediation"]["capturePolicyCatalog"],
                    "clinicalContent": spec["remediation"]["clinicalContentCatalog"],
                    "observableClinicalContract": "data/observable-clinical-contract.v1.0.0.json",
                    "medicationSafetyContract": "data/medication-safety-contract.v1.0.0.json",
                    "clinicalStateContract": "data/clinical-state-contract.v1.0.0.json",
                    "protectedInvariants": spec["remediation"]["protectedInvariantCatalog"],
                    "variabilityPolicy": spec["remediation"]["variabilityPolicyCatalog"],
                    "scenarioTiming": spec["remediation"]["scenarioTimingCatalog"],
                    "referenceContext": spec["remediation"]["referenceContext"],
                })
        input_catalogs = {
            name: {"path": path, "sha256": sha256_file(repository_root / path)}
            for name, path in sorted(catalog_paths.items())
        }
        hmo_anchors = {
            next(identifier["value"] for identifier in entry["resource"].get("identifier", []) if identifier.get("system") == spec["identity"]["cohortIdentifierSystem"]): entry["resource"]
            for entry in built["bundles"]["hmo"]["entry"]
            if entry["resource"].get("resourceType") == "Patient" and any(
                identifier.get("system") == spec["identity"]["cohortIdentifierSystem"]
                for identifier in entry["resource"].get("identifier", [])
            )
        }
        crosswalk_by_cohort = {row["cohortId"]: row for row in built["crosswalk"]["patients"]}
        anchor_date = date.fromisoformat(spec["timelineAnchor"][:10])

        def problem_age_band(cohort_id: str) -> str:
            born = date.fromisoformat(hmo_anchors[cohort_id]["birthDate"])
            age = anchor_date.year - born.year - ((anchor_date.month, anchor_date.day) < (born.month, born.day))
            return "infant" if age < 1 else "child" if age < 13 else "adolescent" if age < 18 else "adult-18-39" if age < 40 else "adult-40-64" if age < 65 else "older-adult"

        problem_distributions = {
            "byKind": dict(sorted(Counter(row["kind"] for row in built["answerKey"]["problems"]).items())),
            "bySeverity": dict(sorted(Counter(row["severity"] for row in built["answerKey"]["problems"]).items())),
            "byArchetype": dict(sorted(Counter(crosswalk_by_cohort[row["cohortId"]]["primaryArchetype"] for row in built["answerKey"]["problems"]).items())),
            "byAgeBand": dict(sorted(Counter(problem_age_band(row["cohortId"]) for row in built["answerKey"]["problems"]).items())),
            "bySex": dict(sorted(Counter(hmo_anchors[row["cohortId"]]["gender"] for row in built["answerKey"]["problems"]).items())),
            "byDistrict": dict(sorted(Counter(crosswalk_by_cohort[row["cohortId"]]["geographyStratum"]["district"] for row in built["answerKey"]["problems"]).items())),
            "bySource": dict(sorted(Counter(
                participant["organization"]
                for row in built["answerKey"]["problems"]
                for participant in {item["organization"]: item for item in row["participants"]}.values()
            ).items())),
        }
        release_blockers = []
        for blocker in [
            *(story_audit or {}).get("releaseBlockers", []),
            *(grammar_audit or {}).get("releaseBlockers", []),
        ]:
            if blocker not in release_blockers:
                release_blockers.append(blocker)
        manifest = {
            "legacyCoveragePolicyVersion": "1.1.0",
            "datasetFamily": spec["datasetFamily"], "datasetVersion": spec["datasetVersion"],
            "profileConformanceClaim": "pending-official-validation",
            "releaseEligible": bool(
                (story_audit or {}).get("releaseEligible", True)
                and (grammar_audit or {}).get("releaseEligible", True)
            ),
            "releaseBlockers": release_blockers,
            "semanticClaim": "intentional-labeled-quality-gaps",
            "populationSize": spec["populationSize"], "organizations": [item["id"] for item in spec["organizations"]],
            "resourceCount": len(index), "resourceCountsByType": dict(sorted(type_counts.items())),
            "canonicalPersonCount": spec["populationSize"],
            "anchorPatientCountBySource": {
                org_id: sum(
                    entry["resource"].get("resourceType") == "Patient" and any(
                        identifier.get("system") == spec["identity"]["cohortIdentifierSystem"]
                        for identifier in entry["resource"].get("identifier", [])
                    ) for entry in built["bundles"][org_id]["entry"]
                ) for org_id in sorted(built["bundles"])
            },
            "duplicatePatientRecordCountBySource": {
                org_id: sum(
                    entry["resource"].get("resourceType") == "Patient" and not any(
                        identifier.get("system") == spec["identity"]["cohortIdentifierSystem"]
                        for identifier in entry["resource"].get("identifier", [])
                    ) for entry in built["bundles"][org_id]["entry"]
                ) for org_id in sorted(built["bundles"])
            },
            "totalPatientResourcesBySource": {
                org_id: sum(entry["resource"].get("resourceType") == "Patient" for entry in built["bundles"][org_id]["entry"])
                for org_id in sorted(built["bundles"])
            },
            "bucketCounts": dict(sorted(bucket_counts.items())), "confidentialityCounts": dict(sorted(confidentiality_counts.items())),
            "clinicalSensitivityCounts": dict(sorted(sensitivity_counts.items())),
            "profileCounts": dict(sorted(profile_counts.items())),
            "problemKindResourceCounts": dict(sorted(problem_kind_counts.items())),
            "archetypeCounts": quality_metrics["archetypePatientCounts"],
            "resourceArchetypeCounts": dict(sorted(archetype_counts.items())),
            "geographicDistrictCounts": dict(sorted(Counter(row["geographyStratum"]["district"] for row in built["crosswalk"]["patients"]).items())),
            "clinicalHistoryBounds": {"earliest": min(clinical_timestamps), "latest": max(clinical_timestamps)},
            "problemCount": built["answerKey"]["count"], "scenarioCount": built["scenarioEvidence"]["count"],
            "problemDistributions": problem_distributions,
            "specSha256": sha256_json(spec),
            "geographyProvenanceSha256": sha256_json(built["geography"]["provenance"]),
            "qualityMetricsSha256": sha256_json(quality_metrics),
            "terminologyAuditSha256": sha256_json(terminology_audit),
            "challengeDetectorResultsSha256": sha256_json(detector_evidence),
            "sensitivityAccountingSha256": sha256_json(sensitivity_accounting) if sensitivity_accounting else None,
            "clinicalRealismAuditSha256": sha256_json(realism_audit) if realism_audit else None,
            "archetypeGrammarAuditSha256": sha256_json(grammar_audit) if grammar_audit else None,
            "clinicalStoryCoverageAuditSha256": sha256_json(story_audit) if story_audit else None,
            "clinicalHistoryAuditSha256": sha256_json(history_audit) if history_audit else None,
            "typedGrammarAuditSha256": sha256_json(typed_grammar_audit) if typed_grammar_audit else None,
            "antiTemplateAuditSha256": sha256_json(variability_audit) if variability_audit else None,
            "latentMappingAuditSha256": sha256_json(latent_mapping_audit) if latent_mapping_audit else None,
            "temporalCausalityAuditSha256": sha256_json(scenario_causality_audit) if scenario_causality_audit else None,
            "clinicalDepthAuditSha256": sha256_json(clinical_depth_audit) if clinical_depth_audit else None,
            "materialVariantAuditSha256": sha256_json(variant_audit) if variant_audit else None,
            "protectedInvariantsAuditSha256": sha256_json(protected_audit) if protected_audit else None,
            "sensitivityAccounting": sensitivity_accounting,
            "clinicalRealismSummary": ({
                key: realism_audit[key] for key in (
                    "patientStructuralSignatures", "archetypes", "sources",
                    "variabilityGrains", "timingConcentration",
                    "clinicalDistributions", "episodes",
                )
            } if realism_audit else None),
            "archetypeGrammarSummary": ({
                "clauseCounts": grammar_audit["clauseCounts"],
                "archetypes": grammar_audit["archetypes"],
                "failureCount": grammar_audit["failureCount"],
            } if grammar_audit else None),
            "clinicalStoryCoverageSummary": ({
                key: story_audit[key] for key in (
                    "patientStoryPlaceholderOccurrenceCount",
                    "coverageOnlyResourcesInPatientStoryBundles",
                    "integratedProfileCoverage", "isolatedCoverageFixtureProfiles",
                    "pediatricImmunization", "releaseEligible", "releaseBlockers",
                )
            } if story_audit else None),
            "terminologyProvenance": {
                "codingOccurrenceCount": terminology_audit["codingOccurrenceCount"],
                "counts": terminology_audit["counts"],
            },
            "conditionAggregateSha256": sha256_json(condition_aggregate) if condition_aggregate else None,
            "inputCatalogs": input_catalogs,
            "sensitivityCatalogSha256": sha256_file(repository_root / spec["sensitivity"]["catalog"]),
            "terminologyCatalogSha256": sha256_file(repository_root / spec["terminology"]["catalog"]),
            "activeCompactBundleSha256": sha256_file(repository_root / "reference/coverage-seed.json"),
            "storagePreflight": {key: value for key, value in storage.items() if key != "availableBytes"},
            "generator": "native-python", "clinicalReviewStatus": "pending",
            "outOfScope": ["live API authorization", "consent lifecycle", "audit logging", "cyber controls", "legal certification"],
        }
        if is_v4_spec(spec):
            manifest["realizedProblemAllocation"] = quality_metrics["realizedProblemAllocation"]
            manifest["positiveScenarioCount"] = sum(not row.get("matchedControl") for row in built["scenarioEvidence"]["scenarios"])
            manifest["matchedControlScenarioCount"] = sum(bool(row.get("matchedControl")) for row in built["scenarioEvidence"]["scenarios"])
            manifest["loadableSourceBundles"] = [
                {
                    "source": org_id,
                    "path": bundle_relative_path(spec, org_id).as_posix(),
                    "bundleType": bundle["type"],
                    "entryCount": len(bundle["entry"]),
                    "resourceCount": len(bundle["entry"]),
                    "bytes": (temp / bundle_relative_path(spec, org_id)).stat().st_size,
                    "sha256": sha256_file(temp / bundle_relative_path(spec, org_id)),
                }
                for org_id, bundle in sorted(built["bundles"].items())
            ]
            manifest["remediationEvidence"] = {
                "latentTruthRows": len(built["latentTruthLedger"]),
                "clinicalHistoryStatus": history_audit["status"],
                "typedGrammarStatus": typed_grammar_audit["status"],
                "antiTemplateStatus": variability_audit["status"],
                "antiTemplateNativeExecutionStatus": variability_audit["nativeExecutionStatus"],
                "antiTemplatePopulationEvidenceStatus": variability_audit["populationEvidenceStatus"],
                "antiTemplateCanonicalQualificationEligible": variability_audit["canonicalQualificationEligible"],
                "antiTemplatePopulationEvidence": {
                    key: variability_audit["populationEvidence"][key]
                    for key in (
                        "scope", "gateView", "grain", "archetypeDenominatorMinimum",
                        "minimumSignaturesUnchanged", "status", "aggregateFailureCount",
                        "insufficientSampleCount", "demonstrableFailureCount",
                        "canonicalQualificationEligible",
                    )
                },
                "latentMappingStatus": latent_mapping_audit["status"],
                "temporalCausalityStatus": scenario_causality_audit["status"],
                "clinicalDepthStatus": clinical_depth_audit["status"],
                "materialVariantStatus": variant_audit["status"],
                "protectedInvariantsStatus": protected_audit["status"],
                "technicalGateStatus": "pass" if all(
                    row["status"] == "pass"
                    for row in (history_audit, typed_grammar_audit, variability_audit, latent_mapping_audit, scenario_causality_audit, clinical_depth_audit, variant_audit, protected_audit)
                ) else "fail",
            }
        write_json(temp / "dataset-manifest.json", manifest)
        write_json(temp / "run-state.json", {"status": "complete", "datasetVersion": spec["datasetVersion"]})
        (temp / "checksums.sha256").write_text("\n".join(checksum_lines(temp)) + "\n", encoding="utf-8")
        assert_storage_safe(spec, temp)
        publish_atomic(target, temp, force)
        return target
    except StorageError as exc:
        write_json(temp / "storage-pause.json", {"status": "paused", **exc.report})
        raise
    except Exception:
        raise


def refresh_checksums(out: Path) -> None:
    (out / "checksums.sha256").write_text("\n".join(checksum_lines(out)) + "\n", encoding="utf-8")


def artifact_summary(out: Path) -> dict[str, Any]:
    manifest = json.loads((out / "dataset-manifest.json").read_text(encoding="utf-8"))
    return {**manifest, "artifactBytes": directory_size(out), "artifactPath": str(out.resolve()), "checksumsSha256": sha256_file(out / "checksums.sha256")}
