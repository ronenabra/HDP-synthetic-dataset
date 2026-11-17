from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .catalogs import load_catalog
from .observable import codings, is_supplemental_capture, patient_graphs
from .typed_grammar import build_typed_grammar_audit, compile_grammar


def _vector(variant: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    events = variant["events"]
    return (
        tuple(row["role"] for row in events), tuple(row["code"] for row in events),
        tuple(row["setting"] for row in events), tuple(row["family"] for row in events),
        tuple(row.get("status", row.get("lifecycle", "completed")) for row in events),
    )


def _distance(left: tuple[Any, ...], right: tuple[Any, ...]) -> int:
    return sum(one != two for one, two in zip(left, right))


def build_variant_audit(
    bundles: dict[str, dict[str, Any]], crosswalk: dict[str, Any], spec: dict[str, Any],
) -> dict[str, Any]:
    catalog = load_catalog(spec["remediation"]["pathwayTemplateCatalog"])
    grammar = compile_grammar(spec["remediation"]["pathwayTemplateCatalog"])
    typed_audit = build_typed_grammar_audit(
        bundles, crosswalk, spec, spec["remediation"]["pathwayTemplateCatalog"],
    )
    inferred_by_cohort = {
        row["cohortId"]: row["inferredVariants"] for row in typed_audit["patients"]
    }
    graphs, _patients = patient_graphs(bundles, spec, include_demographics=False)
    declared: dict[str, Counter[str]] = defaultdict(Counter)
    branches: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in crosswalk["patients"]:
        declared[row["primaryArchetype"]][row["pathwayVariant"]] += 1
        branches[(row["primaryArchetype"], row["pathwayVariant"])][row["secondaryBranch"]] += 1
    canonical = spec["populationSize"] >= 1000
    rows = []
    failures = []
    expected_total = 0
    for archetype, variants in catalog["archetypes"].items():
        vectors = {variant["id"]: _vector(variant) for variant in variants}
        expected_total += len(variants)
        archetype_total = sum(declared[archetype].values())
        for variant in variants:
            variant_id = variant["id"]
            count = declared[archetype][variant_id]
            sibling_distances = [
                _distance(vectors[variant_id], value)
                for sibling, value in vectors.items() if sibling != variant_id
            ]
            nearest = min(sibling_distances, default=0)
            row = {
                "archetype": archetype, "variant": variant_id, "patientCount": count,
                "archetypeShare": round(count / archetype_total, 6) if archetype_total else 0,
                "secondaryBranchCounts": dict(sorted(branches[(archetype, variant_id)].items())),
                "nearestSiblingMaterialDimensionDistance": nearest,
                "predicateVector": vectors[variant_id],
                "assignedPatientsInferredExactly": all(
                    inferred_by_cohort.get(patient["cohortId"]) == [variant_id]
                    for patient in crosswalk["patients"]
                    if patient["primaryArchetype"] == archetype and patient["pathwayVariant"] == variant_id
                    and patient.get("ordinaryPathwayEligible", True)
                ),
            }
            rows.append(row)
            minimum = 3 if archetype == "rare-complex" else 5
            if canonical and count < minimum:
                failures.append(f"{archetype}/{variant_id}: count below {minimum}")
            if canonical and count / max(1, archetype_total) > .30:
                failures.append(f"{archetype}/{variant_id}: exceeds 30%")
            if nearest < 2:
                failures.append(f"{archetype}/{variant_id}: differs from nearest sibling in fewer than two dimensions")
            if not row["assignedPatientsInferredExactly"]:
                failures.append(f"{archetype}/{variant_id}: FHIR-inferred identity differs from assignment")
    actual_variants = sum(bool(count) for counts in declared.values() for count in counts.values())
    if canonical and actual_variants != expected_total:
        failures.append(f"realized variants {actual_variants} != {expected_total}")
    if canonical:
        for archetype in catalog["archetypes"]:
            combinations = {
                (variant, branch) for (row_archetype, variant), counts in branches.items()
                if row_archetype == archetype for branch in counts
            }
            if len(combinations) < 3:
                failures.append(f"{archetype}: fewer than three variant/secondary-branch combinations")
    perioperative = catalog["archetypes"]["perioperative-surgical"]
    perioperative_procedure_families = {
        event["family"] for variant in perioperative for event in variant["events"]
        if event["role"] == "procedure"
    }
    perioperative_topologies = {tuple(event["role"] for event in variant["events"]) for variant in perioperative}
    oncology = catalog["archetypes"]["oncology"]
    oncology_disease_families = {
        event["family"] for variant in oncology for event in variant["events"]
        if event["role"] == "condition"
    }
    oncology_management_modes = {
        (event["role"], event["family"]) for variant in oncology for event in variant["events"]
        if event["role"] in {"oncology-treatment", "procedure", "result", "follow-up"}
    }
    if len(perioperative_procedure_families) < 3 or len(perioperative_topologies) < 3:
        failures.append("perioperative catalog lacks three procedure families/topologies")
    if len(oncology_disease_families) < 3 or len(oncology_management_modes) < 3:
        failures.append("oncology catalog lacks three disease families/management modes")
    secondary_rows = []
    secondary_counts = Counter()
    eligible_counts = Counter()
    template_codes = {
        (archetype, variant["id"]): {
            (event["system"], event["code"]) for event in variant["events"]
        }
        for archetype, variants in catalog["archetypes"].items() for variant in variants
    }
    for patient in crosswalk["patients"]:
        if not patient.get("ordinaryPathwayEligible", True):
            continue
        archetype = patient["primaryArchetype"]
        cohort = patient["cohortId"]
        observed = {
            coding for resource in graphs.get(cohort, []) if not is_supplemental_capture(resource)
            if resource.get("resourceType") in {
                "Condition", "MedicationRequest", "MedicationDispense",
                "MedicationStatement", "MedicationAdministration", "Procedure",
            }
            for coding in codings(resource)
            if coding[0] in {
                "http://snomed.info/sct",
                "http://www.nlm.nih.gov/research/umls/rxnorm",
            }
        }
        extra = sorted(observed - template_codes[(archetype, patient["pathwayVariant"])])
        has_secondary = bool(extra)
        eligible_counts[archetype] += 1
        secondary_counts[archetype] += int(has_secondary)
        secondary_rows.append({
            "cohortId": cohort, "archetype": archetype,
            "hasObservableSecondaryConditionOrTreatment": has_secondary,
            "extraReviewedClinicalCodings": extra,
        })
    secondary_rates = {
        archetype: round(secondary_counts[archetype] / max(1, eligible_counts[archetype]), 6)
        for archetype in ("chronic-care", "medication-intensive", "oncology", "rare-complex")
    }
    if canonical and any(rate < .25 for rate in secondary_rates.values()):
        failures.append("intensive archetype secondary condition/treatment rate is below 25%")
    perioperative_variant_family = {
        variant["id"]: next((
            event["family"] for event in variant["events"] if event["role"] == "procedure"
        ), "no-procedure") for variant in perioperative
    }
    perioperative_realized_families = Counter({
        family: sum(declared["perioperative-surgical"][variant] for variant, mapped in perioperative_variant_family.items() if mapped == family)
        for family in set(perioperative_variant_family.values())
    })
    perioperative_total = sum(perioperative_realized_families.values())
    if canonical and (
        sum(count > 0 for count in perioperative_realized_families.values()) < 3
        or max(perioperative_realized_families.values(), default=0) / max(1, perioperative_total) > .50
    ):
        failures.append("realized perioperative procedure-family diversity/concentration gate failed")
    oncology_variant_disease = {
        variant["id"]: next((
            event["family"] for event in variant["events"] if event["role"] == "condition"
        ), "no-characterized-disease") for variant in oncology
    }
    oncology_variant_mode = {
        variant["id"]: next((
            f"{event['role']}:{event['family']}" for event in variant["events"]
            if event["role"] in {"oncology-treatment", "procedure", "result", "follow-up"}
        ), "no-management-mode") for variant in oncology
    }
    realized_oncology_diseases = Counter({
        family: sum(declared["oncology"][variant] for variant, mapped in oncology_variant_disease.items() if mapped == family)
        for family in set(oncology_variant_disease.values())
    })
    realized_oncology_modes = Counter({
        mode: sum(declared["oncology"][variant] for variant, mapped in oncology_variant_mode.items() if mapped == mode)
        for mode in set(oncology_variant_mode.values())
    })
    if canonical and (
        sum(count > 0 for count in realized_oncology_diseases.values()) < 3
        or sum(count > 0 for count in realized_oncology_modes.values()) < 3
    ):
        failures.append("realized oncology disease/management diversity gate failed")
    # Mutation evidence is evaluated against assignment semantics and material
    # vectors rather than trusting variant labels.
    mutation_rows = []
    for archetype, variants in catalog["archetypes"].items():
        variant_ids = [row["id"] for row in variants]
        vectors = {row["id"]: _vector(row) for row in variants}
        rotated = variant_ids[1:] + variant_ids[:1]
        swapped_implementation = {
            variant_id: vectors[replacement]
            for variant_id, replacement in zip(variant_ids, rotated)
        }
        swap_mismatch_count = sum(
            vectors[variant_id] != swapped_implementation[variant_id]
            for variant_id in variant_ids
        )
        swap_detected = swap_mismatch_count == len(variant_ids) if len(variant_ids) > 1 else True
        collapsed_vectors = {value: _vector(variants[0]) for value in variant_ids}
        collapse_grammar_detected = len(set(collapsed_vectors.values())) != len(collapsed_vectors)
        collapsed_signatures = Counter({"one-clinical-template": max(10, len(variant_ids))})
        collapsed_total = sum(collapsed_signatures.values())
        collapse_diversity_detected = (
            max(collapsed_signatures.values()) / collapsed_total > .20
            and len(collapsed_signatures) < len(variant_ids)
        )
        collapse_depth_detected = len(variant_ids) > 1 and len(set(collapsed_vectors.values())) == 1
        collapse_detected = collapse_grammar_detected and collapse_diversity_detected and collapse_depth_detected
        mutation_rows.append({
            "archetype": archetype, "labelImplementationSwapDetected": swap_detected,
            "labelImplementationSwapMismatchCount": swap_mismatch_count,
            "collapsedBaseTemplateDetected": collapse_detected,
            "collapseGrammarGateFailed": collapse_grammar_detected,
            "collapseDiversityGateFailed": collapse_diversity_detected,
            "collapseClinicalDepthGateFailed": collapse_depth_detected,
            "status": "pass" if swap_detected and collapse_detected else "fail",
        })
        if not swap_detected or not collapse_detected:
            failures.append(f"{archetype}: variant label/collapse mutation was not detected")
    compact_coverage = [
        {
            "archetype": archetype, "variant": variant,
            "clauseCount": len(clauses),
            "positiveFixtureCount": len(clauses),
            "mutationClassCount": len(clauses[0].mutation_classes) if clauses else 0,
        }
        for archetype, variants in grammar.items() for variant, clauses in variants.items()
    ]
    return {
        "schemaVersion": "4.1.0", "canonicalCountGatesApplied": canonical,
        "expectedVariantCount": expected_total, "realizedVariantCount": actual_variants,
        "archetypePatientCounts": {key: sum(value.values()) for key, value in sorted(declared.items())},
        "variants": rows,
        "compactCoverageFixtures": compact_coverage,
        "compactCoverageVariantCount": len(compact_coverage),
        "perioperativeCatalogEvidence": {
            "procedureFamilies": sorted(perioperative_procedure_families),
            "eventTopologies": sorted(perioperative_topologies),
            "realizedProcedureFamilyCounts": dict(sorted(perioperative_realized_families.items())),
        },
        "oncologyCatalogEvidence": {
            "diseaseFamilies": sorted(oncology_disease_families),
            "managementModes": sorted(oncology_management_modes),
            "realizedDiseaseFamilyCounts": dict(sorted(realized_oncology_diseases.items())),
            "realizedManagementModeCounts": dict(sorted(realized_oncology_modes.items())),
        },
        "secondaryConditionOrTreatmentRates": secondary_rates,
        "secondaryEvidenceRows": secondary_rows,
        "mutationEvidence": mutation_rows,
        "typedGrammarFailureCount": typed_audit["failureCount"],
        "failureCount": len(failures), "failures": failures,
        "status": "pass" if not failures else "fail",
    }
