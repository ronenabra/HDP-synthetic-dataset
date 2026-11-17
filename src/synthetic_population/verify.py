from __future__ import annotations

from .observable import with_graph_index

import json
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .blind import detect_corpus, evaluate_candidates
from .rehydrate import require_references
from .problems import SCENARIO_VARIANTS
from .metrics import build_quality_metrics, metric_failures
from .grammar import build_grammar_audit
from .realism import build_realism_audit
from .safety import scan
from .sensitivity import classify as classify_sensitivity, realization_audit, fact_for_patient
from .nested_allocation import nested
from .story import build_story_coverage_audit
from .terminology import audit_terminology
from .utils import read_json, sha256_file, sha256_json
from .patient_identity import patient_index, identifiers as patient_identifiers, load_duplicate_evidence
from .business_identifiers import bundle_errors as business_identifier_errors
from .artifacts import SOURCES, bundle_relative_path, is_v4_spec, load_source_bundles, source_bundle_path
from .history import build_history_audit
from .typed_grammar import build_typed_grammar_audit
from .variability_v4 import build_variability_audit
from .latent_audit import build_latent_mapping_audit
from .scenario_audit import build_scenario_causality_audit
from .clinical_depth import build_clinical_depth_audit
from .variant_audit import build_variant_audit
from .protected import build_protected_invariants_audit






def _references(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "reference" and isinstance(item, str):
                yield item
            else:
                yield from _references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _references(item)


def _empty_array_paths(value: Any, path: str = "") -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else key
            if isinstance(item, list) and not item:
                yield child_path
            else:
                yield from _empty_array_paths(item, child_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _empty_array_paths(item, f"{path}[{index}]")


def _tags(resource: dict, system: str) -> list[str]:
    return [item.get("code") for item in resource.get("meta", {}).get("tag", []) if item.get("system") == system]


def _instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)




def sensitivity_fact_errors(resource: dict, spec: dict, patient_by_url: dict, births: dict) -> list[str]:
    """Classify a fact using the complete real Patient lookup, irrespective of order."""
    fact_tags = _tags(resource, "urn:hdp-test-bundle:synthetic-sensitivity-fact")
    fact_roles = _tags(resource, "urn:hdp-test-bundle:synthetic-fact-role")
    security = resource.get("meta", {}).get("security", [])
    confidences = [row.get("code") for row in security if row.get("system") == spec["metadata"]["confidentialitySystem"]]
    domains = [row.get("code") for row in security if row.get("system") == spec["metadata"]["clinicalSensitivitySystem"]]
    if not fact_tags:
        return ["sensitivity/confidentiality lacks a catalogued fact"] if confidences or domains else []
    if len(fact_tags) != 1 or len(fact_roles) != 1:
        return ["sensitivity fact/role cardinality"]
    subject_ref = next((ref for ref in _references(resource) if ref in patient_by_url), None)
    birth = births.get(subject_ref) if subject_ref else None
    event = next((resource.get(key) for key in ("effectiveDateTime", "recordedDate", "onsetDateTime") if resource.get(key)), None)
    if not birth or not event:
        return ["sensitivity fact lacks age/event evidence"]
    expected_conf, expected_domain, expected_nopat = classify_sensitivity(spec, fact_tags[0], fact_roles[0], birth, event)
    expected_domains = sorted([*expected_domain, *(["NOPAT"] if expected_nopat else [])])
    if confidences != ([expected_conf] if expected_conf else []) or sorted(domains) != expected_domains:
        return ["fact-driven sensitivity classification mismatch"]
    return []


@with_graph_index
def verify(out: Path, *, _resource_checks_only: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    repository_root = Path(__file__).resolve().parent
    manifest_path = out / "dataset-manifest.json"
    if manifest_path.is_file() and read_json(manifest_path).get("acceptedRelease"):
        from .release_repairs import verify_release
        return verify_release(out, repository_root)
    required = {
        "dataset-spec.json", "dataset-manifest.json", "cohort-crosswalk.json", "problem-answer-key.json",
        "scenario-evidence.json", "validation-summary.json",
        "quality-metrics.json", "terminology-audit.json", "challenge-detector-results.json",
        "validation-ledger.ndjson", "resource-index.ndjson", "checksums.sha256", "run-state.json",
    }
    missing = sorted(name for name in required if not (out / name).is_file())
    if missing:
        return {"ok": False, "errors": [f"missing artifacts: {missing}"]}
    require_references(repository_root)
    spec = read_json(out / "dataset-spec.json")
    answer = read_json(out / "problem-answer-key.json")
    timestamp_variants = {
        (participant["organization"], participant["fullUrl"]): item["variant"]
        for item in answer.get("problems", []) if item.get("kind") == "timestamp-order"
        for participant in item.get("participants", [])
    }
    bundle_missing = [
        bundle_relative_path(spec, source).as_posix() for source in SOURCES
        if not (out / bundle_relative_path(spec, source)).is_file()
    ]
    if bundle_missing:
        return {"ok": False, "errors": [f"missing source Bundles: {bundle_missing}"]}
    if spec.get("clinicalRealism"):
        audit_missing = sorted(
            name for name in (
                "sensitivity-accounting.json", "clinical-realism-audit.json",
                "archetype-grammar-audit.json", "clinical-story-coverage-audit.json",
            )
            if not (out / name).is_file()
        )
        if audit_missing:
            errors.append(f"missing R25-R36 audit artifacts: {audit_missing}")
        coverage_missing = sorted(
            f"{org_id}.conformance-coverage.json" for org_id in ("hmo", "hospital-a", "hospital-b")
            if not (out / f"{org_id}.conformance-coverage.json").is_file()
        )
        if coverage_missing:
            errors.append(f"missing isolated conformance coverage artifacts: {coverage_missing}")
    if spec.get("remediation"):
        v4_scenario_evidence = read_json(out / "scenario-evidence.json")
        remediation_missing = sorted(
            name for name in (
                "latent-truth-ledger.ndjson", "clinical-history-audit.json",
                "typed-grammar-audit.json", "anti-template-audit.json",
                "latent-mapping-audit.json",
                "temporal-causality-audit.json",
                "clinical-depth-audit.json",
                "material-variant-audit.json",
                "protected-invariants-audit.json",
            ) if not (out / name).is_file()
        )
        if remediation_missing:
            errors.append(f"missing R37-R45 audit artifacts: {remediation_missing}")
    manifest = read_json(out / "dataset-manifest.json")
    if spec.get("schemaVersion") == "2.0.0":
        required_catalogs = {
            "archetypes", "bucketSemantics", "clinicalDistributions",
            "geography", "identityPolicy", "names", "sensitivity", "sourceModels",
            "terminology",
        }
        if spec.get("clinicalRealism"):
            required_catalogs.update({
                "clinicalRealism", "sourceCapture", "detectorSemantics",
            })
        if spec.get("remediation"):
            required_catalogs.update({
                "pathwayTemplates", "capturePolicies", "clinicalContent", "clinicalStateContract", "medicationSafetyContract", "observableClinicalContract", "protectedInvariants",
                "variabilityPolicy", "scenarioTiming", "referenceContext",
            })
        if spec["demographics"].get("lifecyclePolicyCatalog"):
            required_catalogs.add("lifecyclePolicy")
        if set(manifest.get("inputCatalogs", {})) != required_catalogs:
            errors.append("v2 manifest input-catalog inventory is incomplete or unexpected")
    for name, record in manifest.get("inputCatalogs", {}).items():
        catalog_path = repository_root / record.get("path", "")
        if not catalog_path.is_file() or sha256_file(catalog_path) != record.get("sha256"):
            errors.append(f"input catalog drift: {name}")
    if read_json(out / "run-state.json").get("status") != "complete":
        errors.append("run-state is not complete")
    expected_checksums = {}
    for line in (out / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        expected_checksums[name] = digest
    for name, digest in expected_checksums.items():
        path = out / name
        if not path.is_file() or sha256_file(path) != digest:
            errors.append(f"checksum mismatch: {name}")
    bucket_system = spec["metadata"]["bucketSystem"]
    htest_system = spec["metadata"]["actReasonSystem"]
    conf_system = spec["metadata"]["confidentialitySystem"]
    problem_system = spec["metadata"]["problemIdSystem"]
    all_problem_tags: Counter[str] = Counter()
    resource_count = 0
    clinical_count = 0
    tagged_clinical = 0
    cohort_clinical: dict[tuple[str, str], int] = Counter()
    strict_security: dict[tuple[str, str], list[tuple[str, set[tuple[str, str]]]]] = {}
    resources_by_url: dict[tuple[str, str], dict] = {}
    birth_by_patient_url: dict[tuple[str, str], str] = {}
    strong_ids: set[tuple[str, str]] = set()
    patient_totals_by_source: dict[str, int] = {}
    anchor_counts_by_source: dict[str, int] = {}
    duplicate_counts_by_source: dict[str, int] = {}
    index_rows = [json.loads(line) for line in (out / "resource-index.ndjson").read_text(encoding="utf-8").splitlines() if line]
    index_by_url = {(row["organization"], row["fullUrl"]): row for row in index_rows}
    if is_v4_spec(spec):
        if list(out.glob("*.source.transaction.json")):
            errors.append("v4 source Bundles must not be duplicated at the run root")
        rows = manifest.get("loadableSourceBundles", [])
        if {row.get("source") for row in rows} != set(SOURCES) or len(rows) != 3:
            errors.append("v4 manifest must contain exactly three loadable source Bundles")
        for row in rows:
            try:
                path = source_bundle_path(out, row["source"])
            except (ValueError, KeyError) as exc:
                errors.append(str(exc))
                continue
            if path.is_symlink() or not path.is_file():
                errors.append(f"loadable source Bundle is not a regular file: {row.get('path')}")
                continue
            bundle = read_json(path)
            if row.get("sha256") != sha256_file(path) or row.get("bytes") != path.stat().st_size:
                errors.append(f"loadable source Bundle digest/size mismatch: {row['source']}")
            if row.get("entryCount") != len(bundle.get("entry", [])) or row.get("resourceCount") != len(bundle.get("entry", [])):
                errors.append(f"loadable source Bundle count mismatch: {row['source']}")
    identifier_bundles = {}
    for org_id in SOURCES:
        bundle = read_json(source_bundle_path(out, org_id))
        identifier_bundles[org_id] = bundle
        if bundle.get("resourceType") != "Bundle" or bundle.get("type") != "transaction":
            errors.append(f"{org_id}: not a transaction Bundle")
            continue
        urls = {entry.get("fullUrl") for entry in bundle.get("entry", [])}
        if None in urls or len(urls) != len(bundle.get("entry", [])):
            errors.append(f"{org_id}: duplicate or absent fullUrl")
        patient_by_url, patient_births = patient_index(bundle, spec)
        birth_by_patient_url.update({(org_id, url): birth for url, birth in patient_births.items()})
        patient_count = 0
        patient_cohorts: set[str] = set()
        duplicate_patient_urls: set[str] = set()
        for entry in bundle.get("entry", []):
            resource_count += 1
            resource = entry["resource"]
            for safety_error in scan(resource, spec):
                errors.append(f"{org_id}:{entry['fullUrl']}: {safety_error}")
            for empty_path in _empty_array_paths(resource):
                errors.append(f"{org_id}:{entry['fullUrl']}: empty FHIR array at {empty_path}")
            resources_by_url[(org_id, entry["fullUrl"])] = resource
            rtype = resource.get("resourceType")
            buckets = [item.get("code") for item in resource.get("meta", {}).get("security", []) if item.get("system") == bucket_system]
            htests = [item.get("code") for item in resource.get("meta", {}).get("security", []) if item.get("system") == htest_system]
            confidences = [item.get("code") for item in resource.get("meta", {}).get("security", []) if item.get("system") == conf_system]
            if len(buckets) != 1:
                errors.append(f"{org_id}:{entry['fullUrl']}: bucket count {len(buckets)}")
            if htests != ["HTEST"]:
                errors.append(f"{org_id}:{entry['fullUrl']}: missing/duplicate HTEST")
            if _tags(resource, spec["metadata"]["datasetTagSystem"]) != [spec["datasetVersion"]]:
                errors.append(f"{org_id}:{entry['fullUrl']}: dataset version tag mismatch")
            if _tags(resource, spec["metadata"]["sourceTagSystem"]) != [org_id]:
                errors.append(f"{org_id}:{entry['fullUrl']}: source tag mismatch")
            if len(_tags(resource, "urn:hdp-test-bundle:synthetic-archetype")) != 1:
                errors.append(f"{org_id}:{entry['fullUrl']}: archetype tag mismatch")
            if "N" in confidences or any(code not in {"R", "V"} for code in confidences):
                errors.append(f"{org_id}:{entry['fullUrl']}: forbidden confidentiality {confidences}")
            raw_identifiers = resource.get("identifier", [])
            identifiers = raw_identifiers if isinstance(raw_identifiers, list) else [raw_identifiers]
            if not any(identifier.get("type", {}).get("coding", [{}])[0].get("code") == "strong-id" for identifier in identifiers):
                errors.append(f"{org_id}:{entry['fullUrl']}: missing generator strong identifier")
            for identifier in identifiers:
                if identifier.get("type", {}).get("coding", [{}])[0].get("code") == "strong-id":
                    key = (identifier.get("system"), identifier.get("value"))
                    if key in strong_ids:
                        errors.append(f"{org_id}:{entry['fullUrl']}: strong identifier collision")
                    strong_ids.add(key)
            profiles = resource.get("meta", {}).get("profile", [])
            if not profiles:
                errors.append(f"{org_id}:{entry['fullUrl']}: missing declared profile")
            if rtype == "Encounter" and _instant(resource["period"]["start"]) > _instant(resource["period"]["end"]):
                errors.append(f"{org_id}:{entry['fullUrl']}: encounter chronology")
            if rtype == "Observation" and resource.get("issued") and resource.get("effectiveDateTime"):
                reversed_time = _instant(resource["issued"]) < _instant(resource["effectiveDateTime"])
                timestamp_tagged = "timestamp-order" in _tags(resource, spec["metadata"]["problemKindSystem"])
                variant = timestamp_variants.get((org_id, entry["fullUrl"]))
                if (timestamp_tagged != (variant is not None)
                        or reversed_time != (variant == "issued-before-effective")):
                    errors.append(f"{org_id}:{entry['fullUrl']}: observation chronology/problem mismatch")
            if rtype == "Specimen" and resource.get("receivedTime") and resource.get("collection", {}).get("collectedDateTime"):
                if _instant(resource["receivedTime"]) < _instant(resource["collection"]["collectedDateTime"]):
                    errors.append(f"{org_id}:{entry['fullUrl']}: specimen chronology")
            if rtype == "ServiceRequest" and resource.get("authoredOn") and resource.get("occurrenceDateTime"):
                authored = _instant(resource["authoredOn"])
                occurrence = _instant(resource["occurrenceDateTime"])
                if authored.date() > occurrence.date():
                    errors.append(f"{org_id}:{entry['fullUrl']}: request chronology")
            if spec.get("schemaVersion") == "2.0.0":
                errors.extend(f"{org_id}:{entry['fullUrl']}: {message}" for message in
                              sensitivity_fact_errors(resource, spec, patient_by_url, patient_births))
            for reference in _references(resource):
                if reference.startswith("urn:uuid:") and reference not in urls:
                    errors.append(f"{org_id}:{entry['fullUrl']}: unresolved {reference}")
            row = index_by_url.get((org_id, entry["fullUrl"]))
            if not row or row["sha256"] != sha256_json(resource):
                errors.append(f"{org_id}:{entry['fullUrl']}: index mismatch")
            pids = _tags(resource, problem_system)
            markers = _tags(resource, spec["metadata"]["problemMarkerSystem"])
            local_only = (rtype == "Condition" and
                          "local-only-condition-coding" in _tags(resource, spec["metadata"]["problemKindSystem"]))
            if bool(pids) != (markers == ["intentional-problem"]):
                errors.append(f"{org_id}:{entry['fullUrl']}: problem marker/ID contract mismatch")
            if not local_only:
                all_problem_tags.update(pids)
            if rtype == "Patient":
                patient_count += 1
                cohort = next((item.get("value") for item in identifiers if item.get("system") == spec["identity"]["cohortIdentifierSystem"]), None)
                if cohort:
                    patient_cohorts.add(cohort)
                    patient_by_url[entry["fullUrl"]] = cohort
                    birth_by_patient_url[(org_id, entry["fullUrl"])] = resource.get("birthDate")
                else:
                    duplicate_patient_urls.add(entry["fullUrl"])
            elif rtype not in {"Organization", "Practitioner", "PractitionerRole", "RelatedPerson", "Location", "Questionnaire", "Measure", "ResearchStudy", "Medication", "Substance"}:
                clinical_count += 1
                if pids and not local_only:
                    tagged_clinical += 1
                subject_refs = [ref for ref in _references(resource) if ref in patient_by_url]
                for patient_ref in subject_refs:
                    cohort_clinical[(org_id, patient_by_url[patient_ref])] += 1
                    if "compact-coverage" not in _tags(resource, "urn:hdp-test-bundle:synthetic-archetype"):
                        strict_security.setdefault((org_id, patient_by_url[patient_ref]), []).append((rtype, {
                            (item.get("system"), item.get("code")) for item in resource.get("meta", {}).get("security", [])
                            if item.get("system") in {conf_system, spec["metadata"]["clinicalSensitivitySystem"]}
                        }))
                event_dates = [
                    value for key, value in resource.items()
                    if key in {"authoredOn", "recordedDate", "effectiveDateTime", "issued", "receivedTime", "occurrenceDateTime"}
                    and isinstance(value, str)
                ]
                event_dates.extend(value for value in resource.get("period", {}).values() if isinstance(value, str))
                for patient_ref in subject_refs:
                    birth = birth_by_patient_url.get((org_id, patient_ref))
                    if birth and any(_instant(value).date() < _instant(birth).date() for value in event_dates) \
                            and "timestamp-order" not in _tags(resource, spec["metadata"]["problemKindSystem"]):
                        errors.append(f"{org_id}:{entry['fullUrl']}: event before patient birth")
        if len(patient_cohorts) != spec["populationSize"]:
            errors.append(f"{org_id}: distinct patient cohort count {len(patient_cohorts)} != {spec['populationSize']}")
        patient_totals_by_source[org_id] = patient_count
        anchor_counts_by_source[org_id] = len(patient_cohorts)
        duplicate_counts_by_source[org_id] = len(duplicate_patient_urls)
    answer_ids = {item["problemId"] for item in answer["problems"]}
    errors.extend(business_identifier_errors(identifier_bundles, spec))
    if set(all_problem_tags) != answer_ids:
        errors.append(f"problem tag/key mismatch: tags={len(all_problem_tags)}, key={len(answer_ids)}")
    if len(answer_ids) != answer.get("count"):
        errors.append("problem answer-key count mismatch")
    expected_problem_count = spec["problems"]["patientsWithOne"] + spec["problems"]["patientsWithSecond"]
    if len(answer_ids) != expected_problem_count:
        errors.append(f"problem allocation mismatch: {len(answer_ids)} != {expected_problem_count}")
    problems_by_cohort: Counter[str] = Counter(item["cohortId"] for item in answer["problems"])
    if problems_by_cohort and max(problems_by_cohort.values()) > spec["problems"]["maxPatientProblemIds"]:
        errors.append("maximum patient problem-ID budget exceeded")
    for item in answer["problems"]:
        expected_participants = ((4 if item.get("scenario") == "allergy-treatment-risk" else 3) if item["class"] == "cross-source-clinical" else (2 if item["kind"] in {"conflicting-diagnosis", "discordant-lab", "probable-duplicate"} else 1))
        if not spec.get("clinicalRealism") and len(item.get("participants", [])) != expected_participants:
            errors.append(f"{item['problemId']}: participant count mismatch")
        if spec.get("clinicalRealism") and not item.get("participants"):
            errors.append(f"{item['problemId']}: problem has no exact participants")
        clean_states = item.get("originalCleanTruth", {}).get("participantStates", [])
        if spec.get("clinicalRealism"):
            participant_keys = {
                (row["organization"], row["fullUrl"]) for row in item.get("participants", [])
            }
            state_keys = {
                (row.get("organization"), row.get("fullUrl")) for row in clean_states
            }
            if state_keys != participant_keys:
                errors.append(f"{item['problemId']}: exact clean-state coverage differs from participants")
            if not item.get("expectedTruth", {}).get("hiddenCorrectionNotEmitted"):
                errors.append(f"{item['problemId']}: hidden correction contract is missing")
            changed_state = False
            for state in clean_states:
                operation = state.get("operation")
                observed = resources_by_url.get((state.get("organization"), state.get("fullUrl")))
                if operation == "remove":
                    changed_state = changed_state or observed is not None
                    if "resource" in state:
                        errors.append(f"{item['problemId']}: remove correction unexpectedly embeds a resource")
                elif operation == "replace" and isinstance(state.get("resource"), dict):
                    clean_resource = state["resource"]
                    if item["problemId"] in _tags(clean_resource, problem_system):
                        errors.append(f"{item['problemId']}: hidden clean state still carries its problem tag")
                    changed_state = changed_state or observed != clean_resource
                else:
                    errors.append(f"{item['problemId']}: invalid exact correction operation")
            if not changed_state:
                errors.append(f"{item['problemId']}: exact correction does not change observed data")
        for participant in item.get("participants", []):
            resource = resources_by_url.get((participant["organization"], participant["fullUrl"]))
            if not resource or item["problemId"] not in _tags(resource, problem_system):
                errors.append(f"{item['problemId']}: participant resource/tag mismatch")
    rate_limit = spec["problems"]["maxTaggedClinicalRate"]
    # Tiny integration pilots preserve one case of each three-source scenario. That
    # minimum coverage necessarily exceeds the release-density cap below 100 people.
    if spec["populationSize"] < 100:
        cross_count = spec["problems"]["crossSourceClinicalPatients"]
        third = max(1, cross_count // 3) if cross_count else 0
        allergy_count = max(0, cross_count - 2 * third)
        minimum_cross_records = 3 * cross_count + allergy_count
        rate_limit = max(rate_limit, minimum_cross_records / clinical_count)
    if spec["populationSize"] >= 100 and clinical_count and tagged_clinical / clinical_count > rate_limit:
        errors.append(f"tagged clinical rate exceeded: {tagged_clinical}/{clinical_count}")
    crosswalk = read_json(out / "cohort-crosswalk.json")
    if crosswalk.get("count") != spec["populationSize"]:
        errors.append("crosswalk population mismatch")
    expected_patient_accounting = {
        "canonicalPersonCount": spec["populationSize"],
        "anchorPatientCountBySource": anchor_counts_by_source,
        "duplicatePatientRecordCountBySource": duplicate_counts_by_source,
        "totalPatientResourcesBySource": patient_totals_by_source,
    }
    if crosswalk.get("patientAccounting") != expected_patient_accounting:
        errors.append("crosswalk Patient accounting mismatch")
    for key in expected_patient_accounting:
        if manifest.get(key) != expected_patient_accounting[key]:
            errors.append(f"manifest Patient accounting mismatch: {key}")
    declared_duplicate_urls = {
        (org_id, candidate.get("patientFullUrl"))
        for patient in crosswalk.get("patients", [])
        for org_id, source in patient.get("sources", {}).items()
        for candidate in source.get("duplicateCandidates", [])
    }
    observed_duplicate_urls = {
        (org_id, full_url)
        for (org_id, full_url), resource in resources_by_url.items()
        if resource.get("resourceType") == "Patient" and not any(
            identifier.get("system") == spec["identity"]["cohortIdentifierSystem"]
            for identifier in patient_identifiers(resource)
        )
    }
    if declared_duplicate_urls != observed_duplicate_urls:
        errors.append("duplicate Patient crosswalk reconciliation mismatch")
    if spec.get("schemaVersion") == "2.0.0":
        crosswalk_by_cohort = {row["cohortId"]: row for row in crosswalk.get("patients", [])}
        if nested(spec) and spec.get("remediation"):
            expected_problems: dict[str, list[dict]] = {}
            for item in answer["problems"]:
                expected_problems.setdefault(item["cohortId"], []).append({
                    key: item.get(key) for key in ("problemId", "class", "kind", "scenario", "variant")
                })
            expected_controls = {
                item["cohortId"]: (item["name"], item["structuralVariant"])
                for item in v4_scenario_evidence["scenarios"] if item.get("matchedControl")
            }
            if len(crosswalk_by_cohort) != spec["populationSize"]:
                errors.append("nested crosswalk has duplicate or missing cohorts")
            for index in range(1, spec["populationSize"] + 1):
                cohort = f"cohort-{index:06d}"
                row = crosswalk_by_cohort.get(cohort, {})
                control_name, control_variant = expected_controls.get(cohort, (None, None))
                source_use = sorted(org_id for org_id, view in row.get("sources", {}).items()
                                    if view.get("clinicalCapture", {}).get("active"))
                if (row.get("problemAssignments") != expected_problems.get(cohort, [])
                        or row.get("matchedControlScenario") != control_name
                        or row.get("matchedControlVariant") != control_variant
                        or row.get("sensitivityFact") != fact_for_patient(spec, index)
                        or row.get("sourceUse") != source_use):
                    errors.append(f"{cohort}: nested shared assignment differs from native evidence")
                    break
        hmo_anchors = {
            cohort: resources_by_url[("hmo", row["sources"]["hmo"]["patientFullUrl"])]
            for cohort, row in crosswalk_by_cohort.items()
        }
        anchor_date = date.fromisoformat(spec["timelineAnchor"][:10])

        def age_band(cohort_id: str) -> str:
            born = date.fromisoformat(hmo_anchors[cohort_id]["birthDate"])
            age = anchor_date.year - born.year - ((anchor_date.month, anchor_date.day) < (born.month, born.day))
            return "infant" if age < 1 else "child" if age < 13 else "adolescent" if age < 18 else "adult-18-39" if age < 40 else "adult-40-64" if age < 65 else "older-adult"

        computed_problem_distributions = {
            "byKind": dict(sorted(Counter(row["kind"] for row in answer["problems"]).items())),
            "bySeverity": dict(sorted(Counter(row["severity"] for row in answer["problems"]).items())),
            "byArchetype": dict(sorted(Counter(crosswalk_by_cohort[row["cohortId"]]["primaryArchetype"] for row in answer["problems"]).items())),
            "byAgeBand": dict(sorted(Counter(age_band(row["cohortId"]) for row in answer["problems"]).items())),
            "bySex": dict(sorted(Counter(hmo_anchors[row["cohortId"]]["gender"] for row in answer["problems"]).items())),
            "byDistrict": dict(sorted(Counter(crosswalk_by_cohort[row["cohortId"]]["geographyStratum"]["district"] for row in answer["problems"]).items())),
            "bySource": dict(sorted(Counter(
                participant["organization"]
                for row in answer["problems"]
                for participant in {item["organization"]: item for item in row["participants"]}.values()
            ).items())),
        }
        if manifest.get("problemDistributions") != computed_problem_distributions:
            errors.append("manifest problem distributions differ from answer-key/Bundle evidence")
        for row in answer["problems"]:
            prerequisites = row.get("prerequisites", {})
            archetype = crosswalk_by_cohort[row["cohortId"]]["primaryArchetype"]
            if archetype not in prerequisites.get("compatibleArchetypes", []):
                errors.append(f"{row['problemId']}: incompatible archetype prerequisite")
            if prerequisites.get("requiresAllSources") and {item["organization"] for item in row["participants"]} != {"hmo", "hospital-a", "hospital-b"}:
                errors.append(f"{row['problemId']}: required source participation missing")
        if spec["populationSize"] == 1000 and answer["problems"]:
            archetype_counts = Counter(crosswalk_by_cohort[row["cohortId"]]["primaryArchetype"] for row in answer["problems"])
            if max(archetype_counts.values()) / len(answer["problems"]) > .40:
                errors.append("more than 40% of problem IDs occupy one archetype")
            for kind in {row["kind"] for row in answer["problems"]}:
                kind_archetypes = {crosswalk_by_cohort[row["cohortId"]]["primaryArchetype"] for row in answer["problems"] if row["kind"] == kind}
                if len(kind_archetypes) < 2:
                    errors.append(f"problem kind does not span two archetypes: {kind}")
            cross_archetypes = {crosswalk_by_cohort[row["cohortId"]]["primaryArchetype"] for row in answer["problems"] if row["class"] == "cross-source-clinical"}
            if len(cross_archetypes) < 4:
                errors.append("cross-source positives span fewer than four compatible archetypes")
            ordinals = {int(row["cohortId"].split("-")[-1]) for row in answer["problems"]}
            if ordinals == set(range(1, len(ordinals) + 1)) or max(ordinals) <= spec["problems"]["patientsWithOne"] * 2:
                errors.append("problem allocation remains ordinally predictable")
    hmo_only = 0
    exactly_one_hospital = 0
    hospital_without = {"hospital-a": 0, "hospital-b": 0}
    for patient in crosswalk.get("patients", []):
        anchor_digests = set()
        active_hospitals = sum(cohort_clinical[(org_id, patient["cohortId"])] > 0 for org_id in ("hospital-a", "hospital-b"))
        if active_hospitals == 0:
            hmo_only += 1
        if active_hospitals == 1:
            exactly_one_hospital += 1
        for hospital in hospital_without:
            hospital_without[hospital] += cohort_clinical[(hospital, patient["cohortId"])] == 0
        for org_id in ("hmo", "hospital-a", "hospital-b"):
            if org_id == "hmo" and cohort_clinical[(org_id, patient["cohortId"])] < 1:
                errors.append(f"community longitudinal coverage missing: {org_id}:{patient['cohortId']}")
            source = patient.get("sources", {}).get(org_id, {})
            patient_resource = resources_by_url.get((org_id, source.get("patientFullUrl")))
            if not patient_resource:
                errors.append(f"crosswalk patient missing: {org_id}:{patient['cohortId']}")
            else:
                anchor = sha256_json({"cohortId": patient["cohortId"], "birthDate": patient_resource.get("birthDate"), "gender": patient_resource.get("gender")})
                anchor_digests.add(anchor)
                view = sha256_json({"name": patient_resource.get("name"), "address": patient_resource.get("address"), "telecom": patient_resource.get("telecom", [])})
                if view != source.get("demographicDigest"):
                    errors.append(f"crosswalk demographic digest mismatch: {org_id}:{patient['cohortId']}")
            labels = strict_security.get((org_id, patient["cohortId"]), [])
            non_request_labels = [label for rtype, label in labels if rtype != "ServiceRequest"]
            if spec.get("schemaVersion") != "2.0.0" and non_request_labels and any(label != non_request_labels[0] for label in non_request_labels[1:]):
                errors.append(f"sensitivity graph propagation mismatch: {org_id}:{patient['cohortId']}")
            expected_conf = {item for item in (non_request_labels[0] if non_request_labels else set()) if item[0] == conf_system}
            for rtype, label in labels:
                codes = {code for _, code in label}
                if "NOPAT" in codes and not {"V", "SDV", "NOPAT"}.issubset(codes):
                    errors.append(f"NOPAT contract mismatch: {org_id}:{patient['cohortId']}")
                if spec.get("schemaVersion") != "2.0.0" and rtype == "ServiceRequest":
                    if {item for item in label if item[0] == conf_system} != expected_conf:
                        errors.append(f"ServiceRequest confidentiality exception mismatch: {org_id}:{patient['cohortId']}")
                    if any(system == spec["metadata"]["clinicalSensitivitySystem"] for system, _ in label):
                        errors.append(f"ServiceRequest closed-slice exception mismatch: {org_id}:{patient['cohortId']}")
        if anchor_digests != {patient.get("identityAnchorSha256")}:
            errors.append(f"crosswalk identity anchor mismatch: {patient['cohortId']}")
    population = max(1, spec["populationSize"])
    if spec.get("schemaVersion") == "2.0.0" and spec["populationSize"] >= 100:
        if hmo_only / population < .20:
            errors.append("source utilization guardrail failed: fewer than 20% HMO-only clinical histories")
        if exactly_one_hospital / population < .10:
            errors.append("source utilization guardrail failed: fewer than 10% use exactly one hospital")
        for hospital, count in hospital_without.items():
            if count / population < .20:
                errors.append(f"source utilization guardrail failed: fewer than 20% without {hospital} clinical events")
    if resource_count != manifest.get("resourceCount") or resource_count != len(index_rows):
        errors.append("resource count mismatch")
    scenario = read_json(out / "scenario-evidence.json")
    positive_scenarios = [item for item in scenario.get("scenarios", []) if item.get("problemId")]
    if len(positive_scenarios) != spec["problems"]["crossSourceClinicalPatients"]:
        errors.append("cross-source scenario count mismatch")
    for item in scenario.get("scenarios", []):
        expected = False if item.get("matchedControl") else True
        if any(item["perSource"].values()) or item["aggregate"] is not expected:
            errors.append(f"scenario predicate mismatch: {item['scenarioId']}")
        if item.get("subsetTruthTable") != item.get("expectedSubsetTruthTable") or len(item.get("subsetTruthTable", {})) != 7:
            errors.append(f"scenario seven-subset truth table mismatch: {item['scenarioId']}")
        for evidence_url in item.get("aggregateEvidence", []):
            if not any((org_id, evidence_url) in resources_by_url for org_id in ("hmo", "hospital-a", "hospital-b")):
                errors.append(f"scenario evidence resource missing: {item['scenarioId']}:{evidence_url}")
        evidence_resources = [
            resources_by_url[(org_id, evidence_url)]
            for evidence_url in item.get("aggregateEvidence", [])
            for org_id in ("hmo", "hospital-a", "hospital-b")
            if (org_id, evidence_url) in resources_by_url
        ]
        observations = [resource for resource in evidence_resources if resource["resourceType"] == "Observation"]
        medications = [resource for resource in evidence_resources if resource["resourceType"] == "MedicationRequest"]
        variant = item.get("structuralVariant")
        if spec.get("clinicalRealism"):
            if variant not in SCENARIO_VARIANTS[item["name"]]:
                errors.append(f"scenario structural variant is missing or unknown: {item['scenarioId']}")
        elif variant == "core-exposure-arc" and not any(resource.get("method") for resource in observations):
            errors.append(f"scenario core structural variant is missing method evidence: {item['scenarioId']}")
        elif variant == "monitored-course" and (
            not any(resource.get("note") and resource.get("bodySite") for resource in observations)
            or not any(resource.get("dispenseRequest") for resource in medications)
        ):
            errors.append(f"scenario monitored structural variant is incomplete: {item['scenarioId']}")
        elif variant == "reconciliation-enriched" and (
            not any(resource.get("hasMember") for resource in observations)
            or not any(resource.get("note") for resource in medications)
        ):
            errors.append(f"scenario reconciliation structural variant is incomplete: {item['scenarioId']}")
        elif variant not in {"core-exposure-arc", "monitored-course", "reconciliation-enriched"}:
            errors.append(f"scenario structural variant is missing or unknown: {item['scenarioId']}")
        if not item.get("matchedControl"):
            evidence = item.get("sourceEvidence", {})
            if set(evidence) != {"hmo", "hospital-a", "hospital-b"} or any(not refs for refs in evidence.values()):
                errors.append(f"scenario evidence mismatch: {item['scenarioId']}")
            if len(item.get("aggregateEvidence", [])) != sum(len(refs) for refs in evidence.values()):
                errors.append(f"scenario aggregate evidence mismatch: {item['scenarioId']}")
        else:
            evidence = item.get("sourceEvidence", {})
            if sum(bool(refs) for refs in evidence.values()) < 2 or len(item.get("aggregateEvidence", [])) < 2:
                errors.append(f"matched-control evidence is not sufficiently similar: {item['scenarioId']}")
    if spec["problems"]["crossSourceClinicalPatients"] >= 9:
        for name in {item["name"] for item in positive_scenarios}:
            required_variants = set(SCENARIO_VARIANTS[name]) if spec.get("clinicalRealism") else {"core-exposure-arc", "monitored-course", "reconciliation-enriched"}
            positive_variants = {item.get("structuralVariant") for item in positive_scenarios if item["name"] == name}
            control_variants = {
                item.get("structuralVariant") for item in scenario.get("scenarios", [])
                if item.get("matchedControl") and item["name"] == name
            }
            if positive_variants != required_variants or control_variants != required_variants:
                errors.append(f"scenario structural variants/controls are incomplete: {name}")
    if sha256_json(spec) != manifest.get("specSha256"):
        errors.append("manifest/spec digest mismatch")
    if _resource_checks_only:
        return {"ok": not errors, "errors": errors, "scope": "resource-integrity-only",
                "resourceCount": resource_count, "problemCount": len(answer_ids),
                "clinicalResourceCount": clinical_count, "taggedClinicalResourceCount": tagged_clinical,
                "patientCounts": patient_totals_by_source, "canonicalCounts": anchor_counts_by_source,
                "cohortClinicalSha256": sha256_json(sorted(cohort_clinical.items())),
                "strictSecuritySha256": sha256_json([
                    [list(key), [[kind, sorted(labels)] for kind, labels in rows]]
                    for key, rows in sorted(strict_security.items())])}
    bundles = load_source_bundles(out)
    if spec.get("clinicalRealism"):
        coverage_bundles = {
            org_id: read_json(out / f"{org_id}.conformance-coverage.json")
            for org_id in ("hmo", "hospital-a", "hospital-b")
            if (out / f"{org_id}.conformance-coverage.json").is_file()
        }
        audit_contracts = (
            (
                "sensitivity-accounting.json", "sensitivityAccountingSha256",
                realization_audit(bundles, spec), "sensitivity realization",
            ),
            (
                "archetype-grammar-audit.json", "archetypeGrammarAuditSha256",
                build_grammar_audit(bundles, crosswalk, spec), "archetype grammar",
            ),
            (
                "clinical-story-coverage-audit.json", "clinicalStoryCoverageAuditSha256",
                build_story_coverage_audit(bundles, crosswalk, spec, coverage_bundles),
                "clinical story/coverage",
            ),
            (
                "clinical-realism-audit.json", "clinicalRealismAuditSha256",
                build_realism_audit(bundles, crosswalk, spec), "clinical realism",
            ),
        )
        for filename, digest_field, recomputed_audit, label in audit_contracts:
            path = out / filename
            if not path.is_file():
                continue
            recorded_audit = read_json(path)
            if sha256_json(recorded_audit) != sha256_json(recomputed_audit):
                errors.append(f"{label} audit differs from recomputed Bundle evidence")
            if recorded_audit.get("status") != "pass" and not (
                spec.get("remediation") and label in {"archetype grammar", "clinical realism"}
            ):
                errors.append(f"{label} audit status is not pass")
            if sha256_json(recorded_audit) != manifest.get(digest_field):
                errors.append(f"{label} audit manifest digest mismatch")
        if manifest.get("sensitivityAccounting") != audit_contracts[0][2]:
            errors.append("manifest sensitivity accounting summary differs from recomputed evidence")
        expected_realism = {
            key: audit_contracts[3][2][key] for key in (
                "patientStructuralSignatures", "archetypes", "sources",
                "variabilityGrains", "timingConcentration",
                "clinicalDistributions", "episodes",
            )
        }
        if manifest.get("clinicalRealismSummary") != expected_realism:
            errors.append("manifest clinical realism summary differs from recomputed evidence")
        expected_grammar = {
            "clauseCounts": audit_contracts[1][2]["clauseCounts"],
            "archetypes": audit_contracts[1][2]["archetypes"],
            "failureCount": audit_contracts[1][2]["failureCount"],
        }
        if manifest.get("archetypeGrammarSummary") != expected_grammar:
            errors.append("manifest archetype grammar summary differs from recomputed evidence")
        expected_story = {
            key: audit_contracts[2][2][key] for key in (
                "patientStoryPlaceholderOccurrenceCount",
                "coverageOnlyResourcesInPatientStoryBundles",
                "integratedProfileCoverage", "isolatedCoverageFixtureProfiles",
                "pediatricImmunization", "releaseEligible", "releaseBlockers",
            )
        }
        if manifest.get("clinicalStoryCoverageSummary") != expected_story:
            errors.append("manifest clinical story/coverage summary differs from recomputed evidence")
        expected_release_eligible = bool(
            audit_contracts[1][2].get("releaseEligible", True)
            and audit_contracts[2][2].get("releaseEligible", True)
        )
        if manifest.get("releaseEligible") is not expected_release_eligible:
            errors.append("manifest release eligibility differs from grammar/story evidence")
    if spec.get("remediation"):
        latent_ledger = [
            json.loads(line) for line in (out / "latent-truth-ledger.ndjson").read_text(encoding="utf-8").splitlines()
            if line
        ]
        if spec["terminology"]["catalog"] == "data/terminology.v2.4.0.json":
            from .condition_aggregate import build_condition_aggregate
            allocation_path = out / "condition-allocation.json"
            aggregate_path = out / "condition-aggregate.json"
            if not allocation_path.is_file() or not aggregate_path.is_file():
                errors.append("missing S18 Condition allocation or aggregate artifact")
            else:
                condition_aggregate = read_json(aggregate_path)
                recomputed_condition_aggregate = build_condition_aggregate(
                    bundles, crosswalk, latent_ledger, spec, repository_root,
                )
                allocation = read_json(allocation_path)
                if sha256_json(condition_aggregate) != sha256_json(recomputed_condition_aggregate):
                    errors.append("S18 Condition aggregate differs from recomputed source evidence")
                if condition_aggregate.get("status") != "pass":
                    errors.append("S18 Condition aggregate status is not pass")
                if condition_aggregate.get("targetClasses") != allocation.get("target") or condition_aggregate.get("actualClasses") != allocation.get("actual"):
                    errors.append("S18 Condition allocation differs from aggregate classes")
                if condition_aggregate.get("denominators", {}).get("sourceConditionsN") != allocation.get("denominator"):
                    errors.append("S18 Condition source denominator differs from allocation")
                if sha256_json(condition_aggregate) != manifest.get("conditionAggregateSha256"):
                    errors.append("S18 Condition aggregate manifest digest mismatch")
        v4_audits = (
            ("clinical-history-audit.json", "clinicalHistoryAuditSha256", build_history_audit(
                bundles, crosswalk, spec, latent_ledger,
            ), "clinical history"),
            ("typed-grammar-audit.json", "typedGrammarAuditSha256", build_typed_grammar_audit(
                bundles, crosswalk, spec, spec["remediation"]["pathwayTemplateCatalog"]
            ), "typed grammar"),
            ("anti-template-audit.json", "antiTemplateAuditSha256", build_variability_audit(
                bundles, crosswalk, spec, latent_ledger,
            ), "anti-template"),
            ("latent-mapping-audit.json", "latentMappingAuditSha256", build_latent_mapping_audit(
                bundles, crosswalk, latent_ledger, spec
            ), "latent mapping"),
            ("temporal-causality-audit.json", "temporalCausalityAuditSha256", build_scenario_causality_audit(
                bundles, crosswalk, v4_scenario_evidence, spec
            ), "temporal causality"),
            ("clinical-depth-audit.json", "clinicalDepthAuditSha256", build_clinical_depth_audit(
                bundles, crosswalk, spec
            ), "clinical depth"),
            ("material-variant-audit.json", "materialVariantAuditSha256", build_variant_audit(
                bundles, crosswalk, spec
            ), "material variant"),
            ("protected-invariants-audit.json", "protectedInvariantsAuditSha256", build_protected_invariants_audit(
                bundles, crosswalk, answer,
                realization_audit(bundles, spec) if spec.get("clinicalRealism") else None,
                spec, repository_root,
            ), "protected invariants"),
        )
        for filename, digest_field, recomputed_audit, label in v4_audits:
            recorded_audit = read_json(out / filename)
            if sha256_json(recorded_audit) != sha256_json(recomputed_audit):
                errors.append(f"{label} audit differs from recomputed Bundle evidence")
            if recorded_audit.get("status") != "pass":
                errors.append(f"{label} audit status is not pass")
            if sha256_json(recorded_audit) != manifest.get(digest_field):
                errors.append(f"{label} audit manifest digest mismatch")
        variability = next(audit for _filename, _digest, audit, label in v4_audits if label == "anti-template")
        expected_remediation_evidence = {
            "latentTruthRows": len(latent_ledger),
            "clinicalHistoryStatus": next(audit for _f, _d, audit, label in v4_audits if label == "clinical history")["status"],
            "typedGrammarStatus": next(audit for _f, _d, audit, label in v4_audits if label == "typed grammar")["status"],
            "antiTemplateStatus": variability["status"],
            "antiTemplateNativeExecutionStatus": variability["nativeExecutionStatus"],
            "antiTemplatePopulationEvidenceStatus": variability["populationEvidenceStatus"],
            "antiTemplateCanonicalQualificationEligible": variability["canonicalQualificationEligible"],
            "antiTemplatePopulationEvidence": {
                key: variability["populationEvidence"][key]
                for key in (
                    "scope", "gateView", "grain", "archetypeDenominatorMinimum",
                    "minimumSignaturesUnchanged", "status", "aggregateFailureCount",
                    "insufficientSampleCount", "demonstrableFailureCount",
                    "canonicalQualificationEligible",
                )
            },
            "latentMappingStatus": next(audit for _f, _d, audit, label in v4_audits if label == "latent mapping")["status"],
            "temporalCausalityStatus": next(audit for _f, _d, audit, label in v4_audits if label == "temporal causality")["status"],
            "clinicalDepthStatus": next(audit for _f, _d, audit, label in v4_audits if label == "clinical depth")["status"],
            "materialVariantStatus": next(audit for _f, _d, audit, label in v4_audits if label == "material variant")["status"],
            "protectedInvariantsStatus": next(audit for _f, _d, audit, label in v4_audits if label == "protected invariants")["status"],
        }
        expected_remediation_evidence["technicalGateStatus"] = "pass" if all(
            expected_remediation_evidence[key] == "pass"
            for key in (
                "clinicalHistoryStatus", "typedGrammarStatus", "antiTemplateStatus",
                "latentMappingStatus", "temporalCausalityStatus", "clinicalDepthStatus",
                "materialVariantStatus", "protectedInvariantsStatus",
            )
        ) else "fail"
        if manifest.get("remediationEvidence") != expected_remediation_evidence:
            errors.append("manifest remediation evidence differs from recomputed audits")
    detector_results = read_json(out / "challenge-detector-results.json")
    detector_work = {} if detector_results.get("schemaVersion") == "2.1.0" else None
    recomputed_detectors = evaluate_candidates(detect_corpus(bundles, spec, work_stats=detector_work), answer, scenario, work_stats=detector_work)
    if sha256_json(detector_results) != sha256_json(recomputed_detectors):
        errors.append("challenge detector results differ from complete tag-stripped corpus recomputation")
    if not detector_results.get("allDeclaredRecallComplete") or detector_results.get("namedNearMissFalsePositiveCount") or not detector_results.get("zeroUnexplainedCandidates"):
        errors.append("blind detector recall/control gate failed")
    if sha256_json(detector_results) != manifest.get("challengeDetectorResultsSha256"):
        errors.append("challenge detector results manifest digest mismatch")
    terminology = read_json(out / "terminology-audit.json")
    terminology_bundles = dict(bundles)
    if spec.get("clinicalRealism"):
        terminology_bundles.update({
            f"coverage-{org_id}": read_json(out / f"{org_id}.conformance-coverage.json")
            for org_id in ("hmo", "hospital-a", "hospital-b")
        })
    recomputed_terminology = audit_terminology(terminology_bundles, spec, repository_root)
    if sha256_json(terminology) != sha256_json(recomputed_terminology):
        errors.append("terminology audit differs from recomputed Bundle evidence")
    if terminology.get("status") != "pass":
        errors.append("terminology provenance status is not pass")
    if sha256_json(terminology) != manifest.get("terminologyAuditSha256"):
        errors.append("terminology audit manifest digest mismatch")
    recomputed_metrics = build_quality_metrics(bundles, crosswalk, spec)
    quality = read_json(out / "quality-metrics.json")
    if spec.get("remediation") and ("allocationPolicyVersion" in spec["problems"] or "realizedProblemAllocation" in manifest):
        if manifest.get("realizedProblemAllocation") != recomputed_metrics.get("realizedProblemAllocation"):
            errors.append("manifest realized problem allocation differs from emitted resources")
        scenarios = read_json(out / "scenario-evidence.json")["scenarios"]
        if (manifest.get("positiveScenarioCount") != sum(not row.get("matchedControl") for row in scenarios)
                or manifest.get("matchedControlScenarioCount") != sum(bool(row.get("matchedControl")) for row in scenarios)):
            errors.append("manifest positive/control scenario counts differ from evidence")
    metric_errors = metric_failures(recomputed_metrics, spec)
    recomputed_metrics["status"] = "pass" if not metric_errors else "fail"
    recomputed_metrics["diagnostics"] = metric_errors
    if sha256_json(recomputed_metrics) != sha256_json(quality):
        errors.append("quality metrics differ from recomputed Bundle evidence")
    if quality.get("status") != "pass":
        errors.append("quality metrics status is not pass")
    if sha256_json(quality) != manifest.get("qualityMetricsSha256"):
        errors.append("quality metrics manifest digest mismatch")
    return {
        "ok": not errors, "errors": errors[:100], "resourceCount": resource_count,
        "problemCount": len(answer_ids), "clinicalResourceCount": clinical_count,
        "taggedClinicalResourceCount": tagged_clinical,
        "taggedClinicalRate": tagged_clinical / clinical_count if clinical_count else 0,
        "taggedClinicalRateLimit": rate_limit,
    }


def verify_resource_integrity(out: Path) -> dict[str, Any]:
    """Targeted resource/Patient-dependent checks; excludes expensive audit gates."""
    return verify(out, _resource_checks_only=True)
