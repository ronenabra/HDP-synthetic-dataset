from __future__ import annotations

import copy
import hashlib
import json
import math
from functools import lru_cache
from collections import Counter, defaultdict
from typing import Any

from .catalogs import default_catalog, load_catalog
from .observable import codings, patient_graphs, references
from .temporal import occurrence_from_resource
from .utils import sha256_json


CLINICAL_SYSTEMS = {
    "http://snomed.info/sct", "http://loinc.org",
    "http://www.nlm.nih.gov/research/umls/rxnorm",
}
CAPTURE_SYSTEM = "urn:hdp-test-bundle:synthetic:capture-class"


@lru_cache(maxsize=32)
def _family_map(catalog_path: str) -> dict[tuple[str, str], tuple[str, ...]]:
    pathway_catalog = load_catalog(catalog_path)
    result: dict[tuple[str, str], set[str]] = defaultdict(set)
    for reason in pathway_catalog.get("historicalEncounterReasons", []):
        result[(reason["system"], reason["code"])].add(reason["family"])
    for templates in pathway_catalog["archetypes"].values():
        for template in templates:
            for event in template["events"]:
                result[(event["system"], event["code"])].add(event["family"])
    return {key: tuple(sorted(value)) for key, value in result.items()}


def _reference_type(reference: str) -> str:
    if reference.startswith("urn:uuid:"):
        return "local-reference"
    return reference.split("/", 1)[0]


def _resource_token(
    resource: dict[str, Any], view: str, order_bucket: int | str = "undated",
    *, concept_families: dict | None = None,
) -> tuple[Any, ...]:
    if concept_families is None:
        concept_families = _family_map(default_catalog("remediation", "pathwayTemplateCatalog"))
    codes = sorted({
        (system, code) for system, code in codings({
            key: resource.get(key) for key in (
                "code", "reasonCode", "medicationCodeableConcept", "vaccineCode",
                "category", "interpretation", "clinicalStatus", "type", "bodySite",
            )
    }) if system in CLINICAL_SYSTEMS
    })
    if view == "role-topology":
        codes = [
            (system.rsplit("/", 1)[-1], concept_families.get((system, code), ("unmapped-clinical-concept",)))
            for system, code in codes
        ]
    return (
        resource.get("resourceType"),
        tuple(profile.rsplit("/", 1)[-1] for profile in resource.get("meta", {}).get("profile", [])),
        resource.get("status"), tuple(codes),
        bool(resource.get("basedOn")), bool(resource.get("specimen")),
        bool(resource.get("reasonReference")), bool(resource.get("diagnosis")),
        order_bucket,
    )


def _cosmetic_mutation(resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mutated = copy.deepcopy(resources)
    renamed = {resource["_fullUrl"]: f"urn:uuid:cosmetic-{position}" for position, resource in enumerate(resources) if resource.get("_fullUrl")}
    renamed.update({f"{resource['resourceType']}/{resource['id']}": f"{resource['resourceType']}/cosmetic-{position}" for position, resource in enumerate(resources) if resource.get("id")})

    def visit(value: Any, key: str | None = None) -> None:
        if isinstance(value, dict):
            if value.get("reference") in renamed:
                value["reference"] = renamed[value["reference"]]
            if isinstance(value.get("display"), str):
                value["display"] = "Cosmetic display change"
            if isinstance(value.get("value"), (int, float)) and key in {"valueQuantity", "doseQuantity"}:
                value["value"] = float(value["value"]) + 0.123
            for child_key, child in value.items():
                if child_key in {
                    "effectiveDateTime", "occurrenceDateTime", "performedDateTime",
                    "whenHandedOver", "collectedDateTime", "issued", "authoredOn",
                    "recordedDate", "onsetDateTime", "dateAsserted", "date", "start", "end",
                } and isinstance(child, str) and len(child) >= 10 and child[:4].isdigit():
                    value[child_key] = f"{int(child[:4]) + 28:04d}{child[4:]}"
                else:
                    visit(child, child_key)
        elif isinstance(value, list):
            for child in value:
                visit(child, key)

    for position, resource in enumerate(mutated):
        resource["id"] = f"cosmetic-{position}"
        resource["_fullUrl"] = f"urn:uuid:cosmetic-{position}"
        resource["text"] = {"status": "generated", "div": '<div xmlns="http://www.w3.org/1999/xhtml">Changed narrative.</div>'}
        resource["note"] = [{"text": "Changed note"}]
        resource.setdefault("meta", {}).setdefault("tag", []).append({
            "system": "urn:hdp-test-bundle:synthetic:trajectory", "code": "changed-trajectory-tag",
        })
        resource.setdefault("extension", []).append({
            "url": "urn:hdp-test-bundle:cosmetic-extension", "valueString": "ignored cosmetic value",
        })
        visit(resource)
    return mutated


def _mutation_evidence(
    graphs: dict[str, list[dict[str, Any]]], declarations: dict[str, Any],
    *, pathway_catalog_path: str | None = None,
) -> dict[str, Any]:
    def normalize(resources, view):
        return normalized_signature(resources, view, pathway_catalog_path=pathway_catalog_path)

    representative = next((
        (cohort, resources) for cohort, resources in sorted(graphs.items())
        if cohort in declarations and declarations[cohort].get("ordinaryPathwayEligible", True) and resources
    ), None)
    if representative is None:
        return {"status": "fail", "failures": ["no ordinary representative graph"]}
    cohort, resources = representative
    cosmetic = _cosmetic_mutation(resources)
    invariance = {}
    capture_invariance = {}
    materiality = {}
    optional_branch = {}
    added_patient = {}
    randomized_reason = {}
    for view in ("role-topology", "semantic"):
        original = normalize(resources, view)[0]
        relabeled = copy.deepcopy(resources)
        for position, row in enumerate(relabeled):
            tags = row.setdefault("meta", {}).setdefault("tag", [])
            tags[:] = [item for item in tags if item.get("system") != CAPTURE_SYSTEM]
            tags.append({"system": CAPTURE_SYSTEM, "code": f"random-{position}-received"})
        capture_invariance[view] = original == normalize(relabeled, view)[0]
        invariance[view] = original == normalize(cosmetic, view)[0]
        unused = copy.deepcopy(resources)
        unused.append({
            "resourceType": "Observation", "status": "final",
            "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4"}]},
            "subject": {"reference": "urn:uuid:patient"}, "_fullUrl": "urn:uuid:unused-resource",
            "valueQuantity": {"value": 5.4, "system": "http://unitsofmeasure.org", "code": "%"},
        })
        materiality[view] = original != normalize(unused, view)[0]
        removable_position = next((
            position for position in range(len(resources) - 1, -1, -1)
            if resources[position].get("resourceType") not in {"Patient", "RelatedPerson"}
        ), None)
        reduced = copy.deepcopy(resources)
        if removable_position is not None:
            reduced.pop(removable_position)
        optional_branch[view] = (
            removable_position is not None
            and original != normalize(reduced, view)[0]
            and len(normalize(reduced, view)[1]) < len(normalize(resources, view)[1])
        )
        with_patient = copy.deepcopy(resources) + [{
            "resourceType": "Patient", "id": "new-cosmetic-patient",
            "name": [{"family": "Ignored"}], "_source": "new-source",
        }]
        added_patient[view] = original == normalize(with_patient, view)[0]
        encounter = next((row for row in resources if row.get("resourceType") == "Encounter"), None)
        if encounter:
            first = copy.deepcopy(resources)
            second = copy.deepcopy(resources)
            for collection, code in ((first, "random-a"), (second, "random-b")):
                target = next(row for row in collection if row.get("_fullUrl") == encounter.get("_fullUrl"))
                target["reasonCode"] = [{"coding": [{
                    "system": "urn:hdp-test-bundle:independent-random-reason", "code": code,
                }]}]
            randomized_reason[view] = normalize(first, view)[0] == normalize(second, view)[0]
        else:
            randomized_reason[view] = True
    rewiring = {}
    for candidate_resources in graphs.values():
        clinical_rows = [row for row in candidate_resources if row.get("resourceType") not in {"Patient", "RelatedPerson"}]
        targets = {(row.get("_source"), row.get("_fullUrl")): row for row in clinical_rows if row.get("_fullUrl")}
        changed = copy.deepcopy(candidate_resources)
        redirected = False
        def redirect(value, source):
            nonlocal redirected
            if isinstance(value, dict):
                target = targets.get((source, value.get("reference")))
                if target is not None and not redirected:
                    alternative = next((row for row in clinical_rows if row.get("_source") == source and row.get("_fullUrl") and _resource_token(row, "role-topology", concept_families=_family_map(pathway_catalog_path or default_catalog("remediation", "pathwayTemplateCatalog"))) != _resource_token(target, "role-topology", concept_families=_family_map(pathway_catalog_path or default_catalog("remediation", "pathwayTemplateCatalog")))), None)
                    if alternative:
                        value["reference"] = alternative["_fullUrl"]
                        redirected = True
                for child in value.values():
                    redirect(child, source)
            elif isinstance(value, list):
                for child in value:
                    redirect(child, source)
        for row in changed:
            redirect(row, row.get("_source"))
        if redirected:
            rewiring = {view: normalize(candidate_resources, view)[0] != normalize(changed, view)[0] for view in ("role-topology", "semantic")}
            break
    collapsed_rows = [{"signature": "same", "tokens": [("same",)]} for _ in range(10)]
    collapsed = _metrics(collapsed_rows)
    collapsed_detected = (
        collapsed["largestExactCloneShare"] > .05
        and collapsed["duplicateMembershipRate"] > .30
        and collapsed["effectiveSignatureCount"] < 2
    )
    orphan_detected = all(
        "urn:uuid:unused-resource" not in set(references(resource)) for resource in resources
    )
    failures = [
        label for label, passed in {
            **{f"cosmetic-invariance/{key}": value for key, value in invariance.items()},
            **{f"capture-label-invariance/{key}": value for key, value in capture_invariance.items()},
            **{f"edge-rewiring/{key}": value for key, value in rewiring.items()},
            **{f"material-change/{key}": value for key, value in materiality.items()},
            **{f"optional-branch-removal/{key}": value for key, value in optional_branch.items()},
            **{f"add-patient-preserves-existing/{key}": value for key, value in added_patient.items()},
            **{f"random-reason-exact-code-collapses/{key}": value for key, value in randomized_reason.items()},
            "collapsed-fixture-detected": collapsed_detected,
            "unused-resource-orphan-detected": orphan_detected,
        }.items() if not passed
    ]
    return {
        "representativeCohortId": cohort,
        "cosmeticChanges": ["identifiers", "UUID/fullUrl", "timestamps", "numeric values", "display", "narrative", "notes", "random extension", "trajectory tag"],
        "cosmeticSignatureInvariant": invariance, "captureLabelSignatureInvariant": capture_invariance,
        "rewiredEdgeSignatureChanges": rewiring, "rewiringProbeExercised": bool(rewiring),
        "unusedResourceChangesSignature": materiality,
        "unusedResourceOrphanDetected": orphan_detected,
        "optionalBranchRemovalWorsensStructure": optional_branch,
        "addingPatientPreservesExistingSignature": added_patient,
        "independentRandomReasonCodesCollapse": randomized_reason,
        "independentRandomReasonCoherenceContract": "R37 latent/FHIR fact agreement remains a separate hard gate",
        "collapsedFixture": collapsed,
        "collapsedFixtureDetected": collapsed_detected,
        "failures": failures, "status": "pass" if not failures else "fail",
    }


def normalized_signature(
    resources: list[dict[str, Any]], view: str, *, pathway_catalog_path: str | None = None,
) -> tuple[str, list[tuple[Any, ...]]]:
    concept_families = _family_map(pathway_catalog_path or default_catalog("remediation", "pathwayTemplateCatalog"))
    if view not in {"role-topology", "semantic"}:
        raise ValueError(f"unknown normalization view: {view}")
    from .clinical_semantics import meaningful_result
    clinical = [
        resource for resource in resources
        if meaningful_result(resource) and resource.get("resourceType") not in {"Patient", "RelatedPerson"}
    ]
    occurrences = [occurrence_from_resource(resource) for resource in clinical]
    ordered_instants = sorted({row.interval.start for row in occurrences if row is not None})
    order_buckets = {instant: position for position, instant in enumerate(ordered_instants)}
    base_tokens = [_resource_token(resource, view, order_buckets[occurrence.interval.start] if occurrence else "undated", concept_families=concept_families) for resource, occurrence in zip(clinical, occurrences)]
    local_index = {}
    for position, resource in enumerate(clinical):
        for key in (resource.get("_fullUrl"), f"{resource['resourceType']}/{resource['id']}" if resource.get("id") else None):
            if key:
                local_index[(resource.get("_source"), key)] = position

    def reference_fields(value, path=""):
        if isinstance(value, dict):
            if isinstance(value.get("reference"), str):
                yield path + ".reference", value["reference"], value.get("type")
            for key, child in value.items():
                if key not in {"meta", "text", "note", "identifier", "extension", "reference"}:
                    yield from reference_fields(child, path + "." + key if path else key)
        elif isinstance(value, list):
            for child in value:
                yield from reference_fields(child, path + "[]")

    outgoing = [[] for _ in clinical]
    incoming = [[] for _ in clinical]
    for position, resource in enumerate(clinical):
        for path, reference, explicit_type in reference_fields(resource):
            target = local_index.get((resource.get("_source"), reference))
            if target is None:
                # Unresolved identifiers/UUIDs are not clinical diversity tokens.
                target_type = explicit_type or (reference.split("/", 1)[0] if not reference.startswith("urn:") and "/" in reference else "external-reference")
                outgoing[position].append((path, None, target_type))
            else:
                outgoing[position].append((path, target, None))
                incoming[target].append((path, position))
    labels = [sha256_json(token) for token in base_tokens]
    rows = []
    for _round in range(2):
        rows = [(base_tokens[position],
                 tuple(sorted((path, labels[target] if target is not None else external) for path, target, external in outgoing[position])),
                 tuple(sorted((path, labels[source]) for path, source in incoming[position])))
                for position in range(len(clinical))]
        labels = [sha256_json(row) for row in rows]
    rows = sorted(rows, key=repr)
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(payload.encode()).hexdigest(), rows


def _entropy_effective(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return math.exp(entropy)


def _distance(left: list[tuple[Any, ...]], right: list[tuple[Any, ...]]) -> float:
    one, two = Counter(left), Counter(right)
    shared = sum((one & two).values())
    total = sum((one | two).values())
    return 1 - shared / total if total else 0.0


def _metrics(rows: list[dict[str, Any]], near_threshold: float = .15) -> dict[str, Any]:
    counts = Counter(row["signature"] for row in rows)
    total = len(rows)
    groups = sorted((count, signature) for signature, count in counts.items())
    duplicate_members = sum(count for count in counts.values() if count >= 2)
    representatives = {row["signature"]: Counter(row["tokens"]) for row in rows}
    token_counts = {key: sum(value.values()) for key, value in representatives.items()}
    pair_comparisons = 0
    length_pruned_pairs = 0
    signatures = sorted(counts)
    parent = {signature: signature for signature in signatures}

    def root(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    near_pair_count = sum(count * (count - 1) // 2 for count in counts.values())
    for left_index, left in enumerate(signatures):
        for right in signatures[left_index + 1:]:
            largest_size = max(token_counts[left], token_counts[right])
            # Use the same subtraction as exact distance at inclusive boundaries;
            # comparing the ratio to (1 - threshold) can round the other way.
            lower_bound = 1 - min(token_counts[left], token_counts[right]) / largest_size if largest_size else 0
            if lower_bound > near_threshold:
                length_pruned_pairs += 1
                continue
            pair_comparisons += 1
            one_tokens, two_tokens = representatives[left], representatives[right]
            shared = sum((one_tokens & two_tokens).values())
            total_tokens = token_counts[left] + token_counts[right] - shared
            distance = 1 - shared / total_tokens if total_tokens else 0
            if distance <= near_threshold:
                near_pair_count += counts[left] * counts[right]
                one, two = root(left), root(right)
                if one != two:
                    parent[two] = one
    clusters: Counter[str] = Counter()
    for signature, count in counts.items():
        clusters[root(signature)] += count
    largest_near = max(clusters.values(), default=0)
    return {
        "denominator": total, "signatureCount": len(counts),
        "exactCloneGroups": [
            {"signature": signature, "count": count}
            for count, signature in reversed(groups) if count >= 2
        ],
        "largestExactCloneCount": max(counts.values(), default=0),
        "largestExactCloneShare": round(max(counts.values(), default=0) / total, 6) if total else 0,
        "duplicateMembershipCount": duplicate_members,
        "duplicateMembershipRate": round(duplicate_members / total, 6) if total else 0,
        "topSignatureShare": round(max(counts.values(), default=0) / total, 6) if total else 0,
        "effectiveSignatureCount": round(_entropy_effective(counts), 6),
        "nearCloneThreshold": near_threshold, "nearClonePairCount": near_pair_count,
        "comparisonWork": {"distinctSignatures": len(signatures), "exactDistanceComparisons": pair_comparisons, "lengthBoundPrunedPairs": length_pruned_pairs},
        "largestNearCloneClusterCount": largest_near,
        "largestNearCloneClusterShare": round(largest_near / total, 6) if total else 0,
    }


ARCHETYPE_DIVERSITY_DENOMINATOR = 10


def _classify_variant_evidence(
    *,
    assigned_patient_count: int,
    archetype_eligible_patient_count: int,
    variant_eligible_patient_count: int,
    observed_signature_count: int,
    minimum_signatures: int,
) -> str:
    """Classify one variant without treating an unobserved pilot as a pass.

    The ten-patient archetype denominator is the existing applicability gate for
    stable percentage/signature evidence.  Once that gate applies, the existing
    three-signature (or rare two-signature) rule remains a hard check whenever
    the variant itself has enough eligible patients to exercise it.
    """
    if assigned_patient_count == 0:
        return "absent"
    if variant_eligible_patient_count == 0:
        return "no-eligible-ordinary-history"
    if archetype_eligible_patient_count < ARCHETYPE_DIVERSITY_DENOMINATOR:
        return "insufficient-archetype-sample"
    if variant_eligible_patient_count < minimum_signatures:
        return "insufficient-variant-sample"
    if observed_signature_count < minimum_signatures:
        return "fails-signature-minimum"
    return "meets-signature-minimum"


def _canonical_qualification_eligible(
    *, canonical: bool, population_evidence_status: str, native_execution_status: str,
) -> bool:
    """Require both population evidence and every native diversity gate."""
    return canonical and population_evidence_status == "pass" and native_execution_status == "pass"


def _variant_evidence(
    combined_rows_by_view: dict[str, dict[str, list[dict[str, Any]]]],
    catalog: dict[str, Any],
    thresholds: dict[str, Any],
    assigned_counts: Counter[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Publish every variant denominator and observed signature count.

    ``grain_rows`` contains only ordinary pathway eligible patients, so the
    denominator is intentionally separate from assignment count.  Both the
    role-topology gate and semantic diagnostic view are retained; only the
    former contributes to native population qualification.
    """
    evidence: list[dict[str, Any]] = []
    for view in ("role-topology", "semantic"):
        by_archetype = combined_rows_by_view[view]
        for archetype, variants in catalog["archetypes"].items():
            archetype_rows = by_archetype.get(archetype, [])
            for variant in variants:
                variant_id = variant["id"]
                variant_rows = [row for row in archetype_rows if row["variant"] == variant_id]
                signature_counts = Counter(row["signature"] for row in variant_rows)
                minimum = (
                    thresholds["minimumRareWithinVariantSignatures"]
                    if archetype == "rare-complex"
                    else thresholds["minimumWithinVariantSignatures"]
                )
                assigned = assigned_counts[(archetype, variant_id)]
                disposition = _classify_variant_evidence(
                    assigned_patient_count=assigned,
                    archetype_eligible_patient_count=len(archetype_rows),
                    variant_eligible_patient_count=len(variant_rows),
                    observed_signature_count=len(signature_counts),
                    minimum_signatures=minimum,
                )
                evidence.append({
                    "view": view,
                    "grain": "combined-patient",
                    "gateApplicable": view == "role-topology",
                    "archetype": archetype,
                    "variant": variant_id,
                    "assignedPatientCount": assigned,
                    "archetypeEligiblePatientCount": len(archetype_rows),
                    "variantEligiblePatientCount": len(variant_rows),
                    "observedSignatureCount": len(signature_counts),
                    "observedSignatureCounts": dict(sorted(signature_counts.items())),
                    "minimumSignatures": minimum,
                    "archetypeDenominatorMinimum": ARCHETYPE_DIVERSITY_DENOMINATOR,
                    "disposition": disposition,
                })
    return evidence


def build_variability_audit(
    bundles: dict[str, dict[str, Any]], crosswalk: dict[str, Any], spec: dict[str, Any],
    ledger: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pathway_catalog_path = spec["remediation"]["pathwayTemplateCatalog"]

    def normalize(resources, view):
        return normalized_signature(resources, view, pathway_catalog_path=pathway_catalog_path)

    graphs, _patients = patient_graphs(bundles, spec, include_demographics=False)
    declarations = {row["cohortId"]: row for row in crosswalk["patients"]}
    assigned_counts = Counter(
        (row["primaryArchetype"], row.get("pathwayVariant"))
        for row in crosswalk["patients"]
    )
    policy = load_catalog(spec["remediation"]["variabilityPolicyCatalog"])
    expected_policy_fields = {
        "catalogVersion", "frozenBeforeCanonicalGeneration", "normalizerVersion",
        "includedFields", "excludedFields", "distanceWeights",
        "nearCloneDistanceThreshold", "thresholds",
    }
    expected_threshold_fields = {
        "maximumExactCloneShare", "maximumDuplicateMembershipRate",
        "maximumNearCloneClusterShare", "maximumTopSignatureShare",
        "minimumEffectiveSignatureFraction", "maximumEffectiveSignatureCountRequired",
        "maximumEpisodeTopSignatureShare", "minimumWithinVariantSignatures",
        "minimumRareWithinVariantSignatures",
    }
    if set(policy) != expected_policy_fields or set(policy["thresholds"]) != expected_threshold_fields:
        raise ValueError("variability policy has unknown, missing, or unused fields")
    if policy["distanceWeights"] != {"multisetClinicalRoleToken": 1.0}:
        raise ValueError("unsupported variability distance weights")
    near_threshold = float(policy["nearCloneDistanceThreshold"])
    thresholds = policy["thresholds"]
    mutation_evidence = _mutation_evidence(graphs, declarations, pathway_catalog_path=pathway_catalog_path)
    mappings = {
        (row["source"], row["resourceFullUrl"]): row
        for row in (ledger or []) if row.get("rowType") in {
            "fhir-mapping", "supplemental-source-capture",
        }
    }
    latent_events = {
        row["eventId"]: row for row in (ledger or []) if row.get("rowType") == "latent-event"
    }
    views = {}
    signature_table = []
    failures = []
    aggregate_gate_failures = []
    release_dispositions = []
    raw_episode_failures = []
    combined_rows_by_view = {}
    for view in ("role-topology", "semantic"):
        grain_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
            grain: defaultdict(list) for grain in (
                "combined-patient", "source-local-episode",
                "latent-captured-pathway", "event-role",
            )
        }
        for cohort, resources in graphs.items():
            if cohort not in declarations:
                continue
            if not declarations[cohort].get("ordinaryPathwayEligible", True):
                continue
            signature, tokens = normalize(resources, view)
            base = {
                "cohortId": cohort, "archetype": declarations[cohort]["primaryArchetype"],
                "variant": declarations[cohort].get("pathwayVariant"),
                "secondaryBranch": declarations[cohort].get("secondaryBranch"),
                "signature": signature, "tokens": tokens,
            }
            archetype = base["archetype"]
            grain_rows["combined-patient"][archetype].append(base)
            by_pathway: dict[str, list[dict[str, Any]]] = defaultdict(list)
            by_source_episode: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
            by_event_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for resource in resources:
                mapping = mappings.get((resource["_source"], resource["_fullUrl"]))
                pathway_id = mapping.get("pathwayId", cohort) if mapping else cohort
                event_id = mapping.get("eventId") if mapping else None
                event = latent_events.get(event_id, {})
                episode_id = mapping.get("episodeId") if mapping else None
                episode_id = episode_id or event.get("episodeId") or "observable-unassigned-episode"
                event_role = event.get("eventRole") or resource.get("resourceType", "unknown")
                care_setting = event.get("careSetting") or (mapping or {}).get("careSetting") or "observable-unassigned-setting"
                by_pathway[pathway_id].append(resource)
                by_source_episode[(resource["_source"], episode_id, care_setting)].append(resource)
                by_event_role[event_role].append(resource)
            for pathway_id, pathway_resources in by_pathway.items():
                pathway_signature, pathway_tokens = normalize(pathway_resources, view)
                grain_rows["latent-captured-pathway"][archetype].append({
                    **base, "cohortId": f"{cohort}|{pathway_id}", "grainKey": pathway_id,
                    "signature": pathway_signature, "tokens": pathway_tokens,
                })
            for (source, episode_id, care_setting), source_resources in by_source_episode.items():
                source_signature, source_tokens = normalize(source_resources, view)
                episode_stratum = f"{archetype}|{source}|{care_setting}"
                grain_rows["source-local-episode"][episode_stratum].append({
                    **base, "cohortId": f"{cohort}|{source}|{episode_id}",
                    "grainKey": f"{source}|{episode_id}|{care_setting}", "source": source,
                    "episodeId": episode_id, "careSetting": care_setting,
                    "signature": source_signature,
                    "tokens": source_tokens,
                })
            for event_role, role_resources in by_event_role.items():
                role_signature, role_tokens = normalize(role_resources, view)
                grain_rows["event-role"][archetype].append({
                    **base, "cohortId": f"{cohort}|{event_role}",
                    "grainKey": event_role, "eventRole": event_role,
                    "signature": role_signature, "tokens": role_tokens,
                })
        view_metrics = {}
        for grain, by_archetype in grain_rows.items():
            view_metrics[grain] = {}
            for stratum, rows in sorted(by_archetype.items()):
                metrics = _metrics(rows, near_threshold)
                variant_counts = Counter(row["variant"] for row in rows)
                metrics["materialVariantCounts"] = dict(sorted(variant_counts.items()))
                metrics["withinVariantSignatureCounts"] = {
                    variant: len({row["signature"] for row in rows if row["variant"] == variant})
                    for variant in variant_counts
                }
                view_metrics[grain][stratum] = metrics
                signature_table.extend({
                    key: value for key, value in row.items() if key != "tokens"
                } | {"view": view, "grain": grain} for row in rows)
                if view == "role-topology" and grain == "combined-patient" and len(rows) >= 10:
                    exact_limit = max(2, math.ceil(thresholds["maximumExactCloneShare"] * len(rows)))
                    if metrics["largestExactCloneCount"] > exact_limit:
                        aggregate_gate_failures.append(f"{view}/{grain}/{stratum}: largest exact clone exceeds {exact_limit}")
                    if metrics["duplicateMembershipRate"] > thresholds["maximumDuplicateMembershipRate"]:
                        aggregate_gate_failures.append(f"{view}/{grain}/{stratum}: duplicate membership exceeds policy")
                    if metrics["largestNearCloneClusterShare"] > thresholds["maximumNearCloneClusterShare"]:
                        aggregate_gate_failures.append(f"{view}/{grain}/{stratum}: largest near-clone cluster exceeds policy")
                    if metrics["topSignatureShare"] > thresholds["maximumTopSignatureShare"]:
                        aggregate_gate_failures.append(f"{view}/{grain}/{stratum}: top signature exceeds policy")
                    minimum_effective = min(
                        thresholds["maximumEffectiveSignatureCountRequired"],
                        math.ceil(thresholds["minimumEffectiveSignatureFraction"] * len(rows)),
                    )
                    if metrics["effectiveSignatureCount"] < minimum_effective:
                        aggregate_gate_failures.append(f"{view}/{grain}/{stratum}: effective signatures below {minimum_effective}")
                    if metrics["effectiveSignatureCount"] < len(variant_counts):
                        aggregate_gate_failures.append(f"{view}/{grain}/{stratum}: effective signatures below material variant count")
                if view == "role-topology" and grain == "source-local-episode" and len(rows) >= 10:
                    if metrics["topSignatureShare"] > thresholds["maximumEpisodeTopSignatureShare"]:
                        raw_failure = f"{view}/{grain}/{stratum}: episode top signature exceeds policy"
                        aggregate_gate_failures.append(raw_failure)
        views[view] = view_metrics
        combined_rows_by_view[view] = grain_rows["combined-patient"]

    variant_evidence = _variant_evidence(
        combined_rows_by_view, load_catalog(spec["remediation"]["pathwayTemplateCatalog"]),
        thresholds, assigned_counts,
    )
    native_variant_evidence = [row for row in variant_evidence if row["gateApplicable"]]
    insufficient_sample_rows = [row for row in native_variant_evidence if row["disposition"] in {
        "absent", "no-eligible-ordinary-history", "insufficient-archetype-sample",
        "insufficient-variant-sample",
    }]
    demonstrable_variant_failures = [
        row for row in native_variant_evidence if row["disposition"] == "fails-signature-minimum"
    ]
    population_gate_failures = list(aggregate_gate_failures)
    population_gate_failures.extend(
        f"{row['view']}/{row['grain']}/{row['archetype']}/{row['variant']}: "
        f"within-variant signatures below {row['minimumSignatures']}"
        for row in demonstrable_variant_failures
    )
    population_evidence_status = (
        "fail" if population_gate_failures else "insufficient-sample" if insufficient_sample_rows else "pass"
    )
    canonical = spec["populationSize"] >= 1000
    if canonical and insufficient_sample_rows:
        population_gate_failures.extend(
            f"{row['view']}/{row['grain']}/{row['archetype']}/{row['variant']}: "
            f"canonical evidence denominator is insufficient for the {row['minimumSignatures']}-signature minimum"
            for row in insufficient_sample_rows
        )
        population_evidence_status = "fail"
    failures.extend(population_gate_failures)
    failures.extend(f"mutation/{failure}" for failure in mutation_evidence["failures"])
    native_execution_status = "pass" if not failures else "fail"
    canonical_qualification_eligible = _canonical_qualification_eligible(
        canonical=canonical,
        population_evidence_status=population_evidence_status,
        native_execution_status=native_execution_status,
    )
    population_evidence = {
        "scope": "canonical" if canonical else "pilot",
        "gateView": "role-topology",
        "grain": "combined-patient",
        "archetypeDenominatorMinimum": ARCHETYPE_DIVERSITY_DENOMINATOR,
        "minimumSignaturesUnchanged": {
            "ordinary": thresholds["minimumWithinVariantSignatures"],
            "rare-complex": thresholds["minimumRareWithinVariantSignatures"],
        },
        "status": population_evidence_status,
        "aggregateFailureCount": len(aggregate_gate_failures),
        "aggregateFailures": aggregate_gate_failures,
        "insufficientSampleCount": len(insufficient_sample_rows),
        "demonstrableFailureCount": len(demonstrable_variant_failures),
        "insufficientSamples": insufficient_sample_rows,
        "demonstrableFailures": demonstrable_variant_failures,
        "canonicalQualificationEligible": canonical_qualification_eligible,
        "claim": (
            "canonical qualification may use this evidence"
            if canonical_qualification_eligible else
            "pilot evidence cannot establish canonical population diversity"
        ),
    }
    return {
        "schemaVersion": "4.2.0", "normalizerVersion": policy["normalizerVersion"],
        "policyVersion": policy["catalogVersion"],
        "includedFields": policy["includedFields"],
        "excludedFields": policy["excludedFields"],
        "conceptFamilyMapDigest": sha256_json(load_catalog(pathway_catalog_path)),
        "distanceWeights": policy["distanceWeights"],
        "views": views, "grains": ["combined-patient", "source-local-episode", "latent-captured-pathway", "event-role"],
        "mutationEvidence": mutation_evidence,
        "variantEvidence": variant_evidence,
        "populationEvidence": population_evidence,
        "nativeExecutionStatus": native_execution_status,
        "populationEvidenceStatus": population_evidence_status,
        "canonicalQualificationEligible": canonical_qualification_eligible,
        "signatureTable": sorted(signature_table, key=lambda row: (row["view"], row["grain"], row["archetype"], row["cohortId"])),
        "failures": failures, "status": native_execution_status,
    }
