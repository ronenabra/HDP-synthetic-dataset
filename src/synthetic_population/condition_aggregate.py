"""S18 Condition concept and capture audit at explicitly different grains."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .condition_allocation import CLASS_ORDER, hamilton_counts
from .condition_coding import CLASS_SYSTEM, LOCAL_ONLY_KIND, SNOMED, condition_code_inventory
from .utils import read_json, sha256_file, sha256_json


POLICY = "data/condition-policy.json"
LOCAL_SYSTEM = "urn:hdp-test-bundle:synthetic:condition"
ARCHETYPE_SYSTEM = "urn:hdp-test-bundle:synthetic-archetype"
SENSITIVITY_SYSTEM = "urn:hdp-test-bundle:synthetic-sensitivity-fact"
NONCLINICAL_Q = {"119297000", "368208006", "396550006"}
COMMON = {"195662009", "25374005", "38341003"}


def policy_codes(root: Path) -> set[str]:
    codes = read_json(root / POLICY)["snomedCodes"]
    if len(codes) != 32 or len(set(codes)) != 32:
        raise ValueError("Condition code policy must contain 32 distinct concepts")
    return set(codes)


def _tag(resource: dict, system: str) -> list[str]:
    return [row.get("code", "") for row in resource.get("meta", {}).get("tag", []) if row.get("system") == system]


def _cohort(resource: dict, spec: dict) -> str | None:
    identifier = resource.get("subject", {}).get("identifier", {})
    return identifier.get("value") if identifier.get("system") == spec["identity"]["cohortIdentifierSystem"] else None


def _counts(counter: Counter) -> list[dict]:
    return [{"key": list(key) if isinstance(key, tuple) else key, "count": count}
            for key, count in sorted(counter.items())]


def build_condition_aggregate(
    bundles: dict[str, dict[str, Any]], crosswalk: dict[str, Any],
    ledger: list[dict[str, Any]], spec: dict[str, Any], root: Path,
) -> dict[str, Any]:
    """Audit source Conditions; one coding pair is one clinical concept.

    Typed facts use the latent fact ID. The source episode, code and date are
    the documented reconciliation key for supplemental/legacy rows. Protected
    support pairs use their explicit sensitivity-fact tag. Unkeyed rows are
    counted separately and cannot quietly improve distinct-fact breadth.
    """
    allowed = policy_codes(root)
    from .condition_expectation import condition_expectation
    frozen = condition_expectation(spec, root)
    local = condition_code_inventory()
    people = {row["cohortId"]: row for row in crosswalk["patients"]}
    latent_eligible: dict[str, set[str]] = defaultdict(set)
    for row in ledger:
        if (row.get("rowType") == "latent-event" and row.get("eventRole") == "condition"
                and people.get(row.get("cohortId"), {}).get("evaluationClass") == "ordinary"):
            latent_eligible[row["fact"]["code"]].add(row["cohortId"])
    active_source_patients = Counter(source for person in people.values() for source in person.get("sourceUse", []))
    mapped = {(row["source"], row["resourceFullUrl"]): row for row in ledger
              if row.get("rowType") == "fhir-mapping" and row.get("resourceType") == "Condition"}
    supplemental = {(row["source"], row["resourceFullUrl"]): row for row in ledger
                    if row.get("rowType") == "supplemental-source-capture" and row.get("resourceType") == "Condition"}
    errors: list[str] = []
    class_count = Counter()
    stratum_count = Counter()
    concept_occurrences = Counter()
    concept_strata = Counter()
    concept_patients: dict[str, set[str]] = defaultdict(set)
    concept_patients_by_source: dict[tuple[str, str], set[str]] = defaultdict(set)
    ordinary_concept_patients: dict[str, set[str]] = defaultdict(set)
    concept_facts: dict[str, set[tuple]] = defaultdict(set)
    ordinary_facts: dict[str, set[tuple]] = defaultdict(set)
    ordinary_occurrences = Counter()
    source_patients: dict[str, set[str]] = defaultdict(set)
    all_facts: set[tuple] = set()
    all_patients: set[str] = set()
    reconciliation = Counter()
    dispositions = Counter()
    unsupported = Counter()
    local_only_urls: list[dict[str, str]] = []
    declared_fixture_count = 0
    raw_coding_tokens = 0
    rows_seen: set[tuple[str, str]] = set()
    observed_ordinary_typed = Counter()
    observed_ordinary_typed_strata = Counter()
    declared_overlay_changes = Counter()
    for source, bundle in sorted(bundles.items()):
        for entry in bundle["entry"]:
            resource = entry["resource"]
            if resource.get("resourceType") != "Condition":
                continue
            url = entry["fullUrl"]
            identity = (source, url)
            if identity in rows_seen:
                errors.append(f"duplicate Condition identity {source}/{url}")
                continue
            rows_seen.add(identity)
            cohort = _cohort(resource, spec)
            person = people.get(cohort)
            if not cohort or person is None:
                errors.append(f"Condition has no canonical cohort {source}/{url}")
                continue
            all_patients.add(cohort)
            source_patients[source].add(cohort)
            archetype = _tag(resource, ARCHETYPE_SYSTEM)
            if archetype != [person["primaryArchetype"]]:
                errors.append(f"Condition archetype differs from person {source}/{url}")
            evaluation = person["evaluationClass"]
            variant = person.get("pathwayVariant", "unknown")
            classes = _tag(resource, CLASS_SYSTEM)
            if len(classes) != 1 or classes[0] not in CLASS_ORDER:
                errors.append(f"Condition class missing/invalid {source}/{url}")
                continue
            kind = classes[0]
            class_count[kind] += 1
            stratum_count[(source, archetype[0] if archetype else "unknown", variant, evaluation, kind)] += 1
            codings = resource.get("code", {}).get("coding", [])
            raw_coding_tokens += len(codings)
            standards = [row for row in codings if row.get("system") == SNOMED]
            locals_ = [row for row in codings if row.get("system") == LOCAL_SYSTEM]
            expected_shape = {"standard-only": (1, 0), "standard-plus-local": (1, 1), "local-only": (0, 1)}[kind]
            if len(codings) != sum(expected_shape) or (len(standards), len(locals_)) != expected_shape:
                errors.append(f"Condition coding shape differs from class {source}/{url}")
                continue
            standard = standards[0] if standards else None
            local_coding = locals_[0] if locals_ else None
            concept = None
            if local_coding:
                declaration = local.get(local_coding.get("code"))
                if (declaration is None or local_coding.get("version") != "1.0.0"
                        or local_coding.get("display") != declaration["display"]):
                    errors.append(f"undeclared/drifted local Condition coding {source}/{url}")
                    continue
                mapping = declaration["mapping"]
                concept = mapping["code"]
                if standard and (standard.get("code"), standard.get("display")) != (mapping["code"], mapping["display"]):
                    errors.append(f"local/standard pair changed clinical meaning {source}/{url}")
                if source not in declaration["eligibility"]["sources"] or person["primaryArchetype"] not in declaration["eligibility"]["archetypes"]:
                    errors.append(f"local Condition outside approved eligibility {source}/{url}")
                if not set(resource.get("meta", {}).get("profile", [])) & set(declaration["profiles"]):
                    errors.append(f"local Condition outside approved profile {source}/{url}")
            else:
                concept = standard.get("code") if standard else None
            if standard and standard.get("code") != concept:
                errors.append(f"Condition concept mapping mismatch {source}/{url}")
            if concept not in allowed:
                unsupported[concept or "missing-code"] += 1
                errors.append(f"unsupported Condition concept {source}/{url}:{concept}")
            marker = _tag(resource, spec["metadata"]["problemMarkerSystem"])
            problem_kinds = _tag(resource, spec["metadata"]["problemKindSystem"])
            ids = _tag(resource, spec["metadata"]["problemIdSystem"])
            if kind == "local-only":
                if marker != ["intentional-problem"] or problem_kinds != [LOCAL_ONLY_KIND] or len(ids) != 1:
                    errors.append(f"local-only finding attribution invalid {source}/{url}")
                if concept in NONCLINICAL_Q:
                    errors.append(f"local-only Condition has wrong-code fixture {source}/{url}")
                local_only_urls.append({"source": source, "fullUrl": url, "problemId": ids[0] if ids else ""})
            elif local_coding and marker:
                errors.append(f"ordinary local alias marked as unrelated problem {source}/{url}")
            if concept in NONCLINICAL_Q and not (marker and "semantically-wrong-code" in problem_kinds):
                errors.append(f"nonclinical code outside declared wrong-code fixture {source}/{url}")
            if concept == "46635009" and not (marker and "conflicting-diagnosis" in problem_kinds):
                errors.append(f"type 1 diagnosis outside declared conflict fixture {source}/{url}")
            declared_fixture_count += int(bool(marker) and kind != "local-only")
            clinical = concept not in NONCLINICAL_Q and not (marker and kind != "local-only")
            ordinary = evaluation == "ordinary" and (not marker or kind == "local-only") and clinical
            # Source record/capture identity always counts once, including Q and local-only.
            if clinical:
                concept_occurrences[concept] += 1
                concept_strata[(source, archetype[0] if archetype else "unknown", variant, evaluation, concept)] += 1
                concept_patients[concept].add(cohort)
                concept_patients_by_source[(source, concept)].add(cohort)
            if ordinary:
                ordinary_occurrences[concept] += 1
                ordinary_concept_patients[concept].add(cohort)
            if identity in mapped:
                link = mapped[identity]
                fact = ("typed", cohort, link["factId"])
                reconciliation["typed-fact-id"] += 1
                dispositions[(source, link["disposition"])] += 1
                if ordinary:
                    observed_ordinary_typed[(source, concept)] += 1
                    observed_ordinary_typed_strata[(source, person["primaryArchetype"], variant)] += 1
                    if concept in {"125643001", "44465007"}:
                        from .care_pathway import INJURY_BRANCH_CONCEPTS
                        if (person["primaryArchetype"], variant) not in {
                            ("generally-healthy", "minor-injury"),
                            ("pediatric", "minor-injury"),
                            ("womens-health", "ordinary-injury"),
                        } or INJURY_BRANCH_CONCEPTS.get(person.get("secondaryBranch"), (None,))[0] != concept:
                            errors.append(f"ordinary injury concept contradicts selected pathway branch {source}/{url}")
                if link["observableCode"]["code"] != concept:
                    scenario_roles = _tag(resource, "urn:hdp-test-bundle:synthetic-scenario-role")
                    if evaluation != "ordinary" and (marker or scenario_roles):
                        declared_overlay_changes[(evaluation, link["observableCode"]["code"], concept)] += 1
                    else:
                        errors.append(f"typed Condition code differs from latent clinical fact {source}/{url}")
            elif (sensitivity := _tag(resource, SENSITIVITY_SYSTEM)):
                fact = ("sensitive", cohort, sensitivity[0], concept)
                reconciliation["protected-fact-tag"] += 1
            elif identity in supplemental:
                link = supplemental[identity]
                fact = ("supplemental", cohort, source, link["episodeId"], concept,
                        (link.get("occurrence") or {}).get("start"))
                reconciliation["supplemental-episode-code-time"] += 1
                dispositions[(source, "source-supplemental")] += 1
            else:
                timestamp = resource.get("onsetDateTime") or resource.get("recordedDate")
                episode = resource.get("encounter", {}).get("reference", "")
                if not timestamp and not episode:
                    reconciliation["unresolved-record"] += 1
                    fact = ("unresolved", source, url)
                else:
                    reconciliation["legacy-cohort-source-episode-code-time"] += 1
                    fact = ("legacy", cohort, source, episode, concept, timestamp)
                dispositions[(source, "legacy-or-fixture")] += 1
            if clinical:
                all_facts.add(fact)
                concept_facts[concept].add(fact)
                if ordinary:
                    ordinary_facts[concept].add(fact)
    total = len(rows_seen)
    target = dict(zip(CLASS_ORDER, hamilton_counts(total)))
    if any(class_count[kind] != target[kind] for kind in CLASS_ORDER):
        errors.append("Condition class counts differ from deterministic Hamilton target")
    expected_typed = Counter({(row["source"], row["snomedCode"]): row["count"]
                              for row in frozen["typedSourceConditionByConcept"] if row["evaluation"] == "ordinary"})
    if observed_ordinary_typed != expected_typed:
        errors.append("ordinary typed source concept captures differ from frozen conditional expectation")
    expected_typed_strata = Counter({(row["source"], row["archetype"], row["variant"]): row["count"]
                                     for row in frozen["typedSourceConditionByStratum"] if row["evaluation"] == "ordinary"})
    if observed_ordinary_typed_strata != expected_typed_strata:
        errors.append("ordinary typed source strata differ from frozen conditional expectation")
    for code in COMMON:
        if not ordinary_facts.get(code):
            errors.append(f"missing common ordinary Condition concept {code}")
    breadth_floor = 14 if spec["populationSize"] >= 10000 else 12 if spec["populationSize"] >= 1000 else 0
    if len(ordinary_facts) < breadth_floor:
        errors.append(f"ordinary source breadth {len(ordinary_facts)} below reviewed population floor {breadth_floor}")
    if frozen["staticTemplateTopConceptShare"] > .61:
        errors.append("frozen static template concept concentration exceeds reviewed 61% ceiling")
    ordinary_fact_count = len(set().union(*ordinary_facts.values())) if ordinary_facts else 0
    top_fact_share = max((len(values) for values in ordinary_facts.values()), default=0) / ordinary_fact_count if ordinary_fact_count else 0
    ordinary_occurrence_total = sum(ordinary_occurrences.values())
    top_occurrence_share = max(ordinary_occurrences.values(), default=0) / ordinary_occurrence_total if ordinary_occurrence_total else 0
    rows = [{
        "snomedCode": code, "sourceOccurrences": concept_occurrences[code],
        "ordinarySourceOccurrences": ordinary_occurrences[code],
        "patients": len(concept_patients[code]), "facts": len(concept_facts[code]),
        "ordinaryPatients": len(ordinary_concept_patients[code]), "ordinaryFacts": len(ordinary_facts[code]),
        "wholeCohortPeople": spec["populationSize"],
        "patientShareOfWholeCohort": len(concept_patients[code]) / spec["populationSize"],
        "ordinaryLatentEligiblePatients": len(latent_eligible[code]),
        "capturedOrdinaryLatentEligiblePatients": len(ordinary_concept_patients[code] & latent_eligible[code]),
        "captureRateAmongLatentEligible": (len(ordinary_concept_patients[code] & latent_eligible[code]) / len(latent_eligible[code])
                                          if latent_eligible[code] else None),
        "latentEligibilityBasis": "ordinary pre-capture Condition events only; conditional primary/scenario paths have separate denominators",
        "ordinaryOccurrenceRate": ordinary_occurrences[code] / ordinary_occurrence_total if ordinary_occurrence_total else 0,
    } for code in sorted(concept_occurrences)]
    return {
        "schemaVersion": "1.0.0", "status": "pass" if not errors else "fail", "errors": errors,
        "policySha256": sha256_file(root / POLICY),
        "conditionExpectationSha256": sha256_json(frozen), "conditionExpectationSpecSha256": frozen["specSha256"],
        "denominators": {"sourceConditionsN": total, "canonicalCohortPeople": spec["populationSize"],
                         "conditionPatientsP": len(all_patients), "clinicalFactsF": len(all_facts),
                         "sourceCapturesC": total, "ordinaryClinicalFacts": ordinary_fact_count,
                         "ordinarySourceConditions": ordinary_occurrence_total,
                         "rawConditionCodingTokens": raw_coding_tokens},
        "targetClasses": target, "actualClasses": {kind: class_count[kind] for kind in CLASS_ORDER},
        "classBySourceArchetypeVariantEvaluation": _counts(stratum_count),
        "conditionPatientsBySource": {source: len(values) for source, values in sorted(source_patients.items())},
        "activeSourcePatientsBySource": dict(sorted(active_source_patients.items())),
        "declaredFixtureSourceConditions": declared_fixture_count,
        "factReconciliation": dict(sorted(reconciliation.items())),
        "captureDispositions": _counts(dispositions),
        "distinctClinicalConcepts": len(concept_occurrences),
        "distinctOrdinaryConcepts": len(ordinary_facts),
        "localAliasCodes": len({row["code"] for row in local.values()}),
        "concepts": rows, "unsupportedCodes": _counts(unsupported),
        "conceptPatientsBySource": [
            {"source": source, "snomedCode": code, "patients": len(cohorts),
             "activeSourcePatients": active_source_patients[source],
             "patientShareOfActiveSource": len(cohorts) / active_source_patients[source] if active_source_patients[source] else None}
            for (source, code), cohorts in sorted(concept_patients_by_source.items())
        ],
        "conceptOccurrencesBySourceArchetypeVariantEvaluation": _counts(concept_strata),
        "concentration": {"frozenStaticTemplateTopShare": frozen["staticTemplateTopConceptShare"],
                          "ordinaryFactsTopShare": top_fact_share,
                          "ordinarySourceOccurrencesTopShare": top_occurrence_share},
        "expectedOrdinaryTypedSourceConcepts": _counts(expected_typed),
        "actualOrdinaryTypedSourceConcepts": _counts(observed_ordinary_typed),
        "expectedOrdinaryTypedSourceStrata": _counts(expected_typed_strata),
        "actualOrdinaryTypedSourceStrata": _counts(observed_ordinary_typed_strata),
        "declaredOverlayConceptChanges": _counts(declared_overlay_changes),
        "localOnlyWarningCandidates": local_only_urls,
        "factKeyPolicy": "typed cohort/factId; protected cohort/sensitivity tag/concept; supplemental cohort/source/episode/concept/start; legacy cohort/source/encounter/concept/time; unresolved record identity reported",
    }
