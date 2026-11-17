from __future__ import annotations

import copy
from itertools import combinations
from pathlib import Path
from typing import Any

from .utils import read_json
from .nested_allocation import POLICY_VERSION as NESTED_ALLOCATION_POLICY


ROOT_FIELDS = {
    "schemaVersion", "datasetFamily", "datasetVersion", "populationSize", "seed",
    "timelineAnchor", "fhirVersion", "hdpPackage", "ilCorePackage", "generator",
    "organizations", "identity", "metadata", "sensitivity", "terminology", "problems", "demographics", "geography",
    "storage", "validation",
}
V2_ROOT_FIELDS = ROOT_FIELDS | {
    "archetypes", "sourceModels", "clinicalDistributions", "identityPolicy",
}
REMEDIATION_FIELDS = {
    "pathwayTemplateCatalog", "capturePolicyCatalog", "clinicalContentCatalog", "protectedInvariantCatalog",
    "variabilityPolicyCatalog", "scenarioTimingCatalog",
    "referenceContext",
}
NATURAL_ALLOCATION_POLICY = "4.1.0-natural-pilot"

ORGANIZATION_IDS = ("hmo", "hospital-a", "hospital-b")


def _exact_fields(value: dict[str, Any], expected: set[str], label: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise ValueError(f"{label} fields mismatch: unknown={unknown}, missing={missing}")


def load_spec(path: Path) -> dict[str, Any]:
    spec = read_json(path)
    if not isinstance(spec, dict):
        raise ValueError("spec must be a JSON object")
    schema_version = spec.get("schemaVersion")
    if schema_version != "2.0.0" or not spec.get("remediation"):
        raise ValueError("only current native specifications with remediation catalogs are supported")
    expected_root = (
        V2_ROOT_FIELDS
        | ({"clinicalRealism"} if "clinicalRealism" in spec else set())
        | ({"remediation"} if "remediation" in spec else set())
    ) if schema_version == "2.0.0" else ROOT_FIELDS
    unknown = sorted(set(spec) - expected_root)
    missing = sorted(expected_root - set(spec))
    if unknown or missing:
        raise ValueError(f"spec fields mismatch: unknown={unknown}, missing={missing}")
    if spec["schemaVersion"] not in {"1.0.0", "2.0.0"} or spec["datasetFamily"] != "synthetic-population":
        raise ValueError("unsupported schemaVersion or datasetFamily")
    if not isinstance(spec["populationSize"], int) or spec["populationSize"] < 1:
        raise ValueError("populationSize must be a positive integer")
    if [item.get("id") for item in spec["organizations"]] != list(ORGANIZATION_IDS):
        raise ValueError("organizations must be hmo, hospital-a, hospital-b in that order")
    _exact_fields(spec["generator"], {"adapter", "strictPartialPerOrganization", "serialization"}, "generator")
    for item in spec["organizations"]:
        _exact_fields(item, {"id", "name", "kind"}, f"organization {item.get('id')}")
    _exact_fields(spec["identity"], {"cohortIdentifierSystem", "testIdentifierSystem", "strongIdentifierPrefix", "anchorPatients"}, "identity")
    for item in spec["identity"]["anchorPatients"]:
        _exact_fields(item, {"cohortId", "testId", "given", "family"}, "anchor patient")
    _exact_fields(spec["metadata"], {
        "datasetTagSystem", "sourceTagSystem", "problemMarkerSystem", "problemKindSystem", "problemIdSystem",
        "bucketSystem", "actReasonSystem", "confidentialitySystem", "clinicalSensitivitySystem", "testReasonCode",
        "allowedConfidentialityCodes", "forbiddenConfidentialityCodes",
    }, "metadata")
    if schema_version == "2.0.0":
        _exact_fields(spec["sensitivity"], {"catalog", "factCohorts"}, "sensitivity")
        _exact_fields(spec["archetypes"], {"catalog"}, "archetypes")
        _exact_fields(spec["sourceModels"], {"catalog"}, "sourceModels")
        _exact_fields(spec["clinicalDistributions"], {"catalog"}, "clinicalDistributions")
        _exact_fields(spec["identityPolicy"], {"catalog"}, "identityPolicy")
        if "clinicalRealism" in spec:
            _exact_fields(spec["clinicalRealism"], {"catalog"}, "clinicalRealism")
        if "remediation" in spec:
            _exact_fields(spec["remediation"], REMEDIATION_FIELDS, "remediation")
            if not spec["datasetVersion"].startswith("4."):
                raise ValueError("R37-R45 remediation requires a v4 dataset version")
    else:
        _exact_fields(spec["sensitivity"], {"catalog", "restrictedMentalHealthEvery", "veryRestrictedHivEvery", "veryRestrictedNoPatientAccessEvery"}, "sensitivity")
    _exact_fields(spec["terminology"], {"catalog"}, "terminology")
    problem_fields = {"catalogVersion", "patientsWithOne", "patientsWithSecond", "crossSourceClinicalPatients", "maxPatientProblemIds", "maxTaggedClinicalRate", "kinds"}
    if spec.get("remediation") and "allocationPolicyVersion" in spec["problems"]:
        problem_fields.add("allocationPolicyVersion")
        if spec["problems"]["allocationPolicyVersion"] not in {NATURAL_ALLOCATION_POLICY, NESTED_ALLOCATION_POLICY}:
            raise ValueError("unsupported v4 allocation policy")
    _exact_fields(spec["problems"], problem_fields, "problems")
    demographic_fields = {"nameVariationRate", "addressVariationRate", "telecomVariationRate", "telephonePrefix"}
    if schema_version == "2.0.0":
        demographic_fields |= {"namesCatalog", "genderDistribution", "pediatricAgeBands"}
        if "lifecyclePolicyCatalog" in spec["demographics"]:
            demographic_fields.add("lifecyclePolicyCatalog")
    _exact_fields(spec["demographics"], demographic_fields, "demographics")
    if "lifecyclePolicyCatalog" in spec["demographics"]:
        from .catalogs import load_catalog
        policy = load_catalog(spec["demographics"]["lifecyclePolicyCatalog"])
        _exact_fields(policy, {"version", "deceasedPerThousand", "pediatricPerThousand", "olderAdultMinPercent", "olderAdultMaxPercent", "minimumDeceasedAgeYears", "earliestDeathDate", "latestDeathDate"}, "lifecycle policy")
        if (policy["version"] != "1.0.0" or not isinstance(policy["deceasedPerThousand"], int)
                or not 0 <= policy["deceasedPerThousand"] <= 1000
                or not isinstance(policy["minimumDeceasedAgeYears"], int)
                or policy["minimumDeceasedAgeYears"] < 18):
            raise ValueError("invalid lifecycle policy target or age eligibility")
        if (not isinstance(policy["pediatricPerThousand"], int)
                or not 0 <= policy["pediatricPerThousand"] <= 1000
                or not 0 <= policy["olderAdultMinPercent"] <= policy["olderAdultMaxPercent"] <= 100):
            raise ValueError("invalid lifecycle age targets")
        from datetime import date
        if not (date.fromisoformat(policy["earliestDeathDate"]) <= date.fromisoformat(policy["latestDeathDate"])
                < date.fromisoformat(spec["timelineAnchor"][:10])):
            raise ValueError("lifecycle policy death window must precede timeline anchor")
    _exact_fields(spec["geography"], {"catalog", "requiredDistricts"}, "geography")
    _exact_fields(spec["storage"], {"minimumFreeReserveGiB", "projectedFinalMultiplier", "fixedValidatorOverheadGiB", "monitorEveryPatients", "monitorEveryBytes", "fallbackBytesPerPatient"}, "storage")
    validation_fields = {"canonicalMode", "knownTransitionAllowlist", "clinicalReviewStatus"}
    if "envelopeStrategy" in spec["validation"]:
        validation_fields.add("envelopeStrategy")
        if spec["validation"]["envelopeStrategy"] != "patient-mini-bundles-v1":
            raise ValueError("unsupported formal envelope strategy")
    _exact_fields(spec["validation"], validation_fields, "validation")
    if spec["generator"].get("adapter") != "native":
        raise ValueError("only the native adapter is accepted for canonical v1 runs")
    if not spec["generator"].get("strictPartialPerOrganization"):
        raise ValueError("canonical v1 requires strictPartialPerOrganization")
    problem = spec["problems"]
    if problem["patientsWithSecond"] > problem["patientsWithOne"] or problem["patientsWithOne"] > spec["populationSize"]:
        raise ValueError("problem allocation exceeds population")
    if problem["crossSourceClinicalPatients"] > problem["patientsWithOne"]:
        raise ValueError("cross-source cohort must be within the first-problem cohort")
    if "N" not in spec["metadata"]["forbiddenConfidentialityCodes"]:
        raise ValueError("canonical spec must forbid confidentiality N")
    if set(spec["metadata"]["allowedConfidentialityCodes"]) != {"R", "V"}:
        raise ValueError("canonical spec permits only R and V confidentiality labels")
    if any(not 0 <= spec["demographics"][key] <= 1 for key in ("nameVariationRate", "addressVariationRate", "telecomVariationRate")):
        raise ValueError("demographic variation rates must be between zero and one")
    if schema_version == "1.0.0" and any(spec["sensitivity"][key] < 1 for key in ("restrictedMentalHealthEvery", "veryRestrictedHivEvery", "veryRestrictedNoPatientAccessEvery")):
        raise ValueError("sensitivity selection intervals must be positive")
    if schema_version == "2.0.0":
        gender = spec["demographics"]["genderDistribution"]
        if set(gender) != {"female", "male"} or abs(sum(gender.values()) - 1.0) > 1e-9:
            raise ValueError("genderDistribution must contain female/male probabilities summing to one")
    if spec["generator"]["serialization"] != "json-pretty-2-final-newline":
        raise ValueError("unsupported serialization contract")
    if spec.get("remediation") and spec["problems"].get("allocationPolicyVersion") == NESTED_ALLOCATION_POLICY:
        if spec["populationSize"] % 1000:
            raise ValueError("nested allocation requires complete 1,000-person blocks")
        blocks = spec["populationSize"] // 1000
        from .archetypes import distribution
        archetypes = {row["id"]: row["count"] for row in distribution({**spec, "populationSize": 1000})}
        if sum(archetypes.values()) != 1000:
            raise ValueError("nested allocation infeasible: archetype quotas")
        if sum(spec["demographics"]["pediatricAgeBands"].values()) != archetypes["pediatric"]:
            raise ValueError("nested allocation infeasible: pediatric age bands")
        female_target = round(1000 * spec["demographics"]["genderDistribution"]["female"])
        if female_target < archetypes["womens-health"] + 1 or female_target > 999:
            raise ValueError("nested allocation infeasible: sex and anchor eligibility")
        fact_slots = {"chronic-care": 0, "pediatric": 0, "womens-health": 0}
        for fact, count in spec["sensitivity"]["factCohorts"].items():
            if count % blocks:
                raise ValueError(f"nested allocation infeasible: {fact} quota")
            owner = ("pediatric" if fact.startswith("minor-") else "womens-health" if fact in {"protected-domestic-violence-indicator", "ordinary-injury"} else "chronic-care")
            fact_slots[owner] += count // blocks
        if any(fact_slots[key] > archetypes[key] for key in fact_slots):
            raise ValueError("nested allocation infeasible: sensitivity eligibility")
        for key in ("patientsWithOne", "patientsWithSecond", "crossSourceClinicalPatients"):
            if problem[key] % blocks:
                raise ValueError(f"nested allocation infeasible: {key} quota")
        if problem["patientsWithOne"] // blocks - problem["crossSourceClinicalPatients"] // blocks < problem["patientsWithSecond"] // blocks:
            raise ValueError("nested allocation infeasible: secondary problems require non-scenario patients")
        if (problem["patientsWithOne"] + problem["crossSourceClinicalPatients"]) // blocks > 1000:
            raise ValueError("nested allocation infeasible: distinct matched controls")
        from .problems import scenario_name
        compatible = {
            "renal-medication-risk": {"chronic-care", "medication-intensive", "oncology", "rare-complex"},
            "bleeding-risk": {"medication-intensive", "perioperative-surgical", "oncology", "rare-complex"},
            "allergy-treatment-risk": {"chronic-care", "pediatric", "medication-intensive", "perioperative-surgical", "oncology", "rare-complex"},
        }
        scenarios = list(compatible)
        per_block = problem["crossSourceClinicalPatients"] // blocks
        targets = {name: sum(scenario_name(i, per_block) == name for i in range(1, per_block + 1)) for name in scenarios}
        for length in range(1, 4):
            for subset in combinations(scenarios, length):
                eligible = set().union(*(compatible[name] for name in subset))
                if 2 * sum(targets[name] for name in subset) > sum(archetypes[name] for name in eligible):
                    raise ValueError(f"nested allocation infeasible: clinical eligibility for {subset}")
    return spec


def scale_spec(source: dict[str, Any], population_size: int, dataset_version: str, *, zero_problems: bool = False) -> dict[str, Any]:
    if population_size < 1:
        raise ValueError("population size must be positive")
    result = copy.deepcopy(source)
    result["populationSize"] = population_size
    result["datasetVersion"] = dataset_version
    source_n = source["populationSize"]
    if source.get("remediation"):
        result["problems"]["allocationPolicyVersion"] = NESTED_ALLOCATION_POLICY if population_size >= 1000 and population_size % 1000 == 0 else NATURAL_ALLOCATION_POLICY
    for key in ("patientsWithOne", "crossSourceClinicalPatients"):
        result["problems"][key] = int(population_size * source["problems"][key] / source_n + 0.5)
    # A secondary problem is optional enrichment, so a small pilot must not round
    # one into existence and accidentally exceed the fixed intentional-noise cap.
    result["problems"]["patientsWithSecond"] = population_size * source["problems"]["patientsWithSecond"] // source_n
    if source["problems"]["crossSourceClinicalPatients"] and not source.get("remediation") and not zero_problems:
        if population_size < 3:
            raise ValueError("scaled problem mode requires at least three patients for scenario coverage")
        result["problems"]["crossSourceClinicalPatients"] = max(3, result["problems"]["crossSourceClinicalPatients"])
        result["problems"]["patientsWithOne"] = max(
            result["problems"]["patientsWithOne"], result["problems"]["crossSourceClinicalPatients"]
        )
    result["problems"]["patientsWithOne"] = min(population_size, result["problems"]["patientsWithOne"])
    result["problems"]["patientsWithSecond"] = min(result["problems"]["patientsWithOne"], result["problems"]["patientsWithSecond"])
    result["problems"]["crossSourceClinicalPatients"] = min(result["problems"]["patientsWithOne"], result["problems"]["crossSourceClinicalPatients"])
    if zero_problems:
        for key in ("patientsWithOne", "patientsWithSecond", "crossSourceClinicalPatients"):
            result["problems"][key] = 0
    if source.get("schemaVersion") == "2.0.0":
        source_facts = source["sensitivity"]["factCohorts"]
        exact = {key: population_size * value / source_n for key, value in source_facts.items()}
        fact_counts = {key: int(value) for key, value in exact.items()}
        target = int(population_size * sum(source_facts.values()) / source_n + 0.5)
        for key in sorted(source_facts, key=lambda item: (exact[item] - fact_counts[item], item), reverse=True)[:target - sum(fact_counts.values())]:
            fact_counts[key] += 1
        result["sensitivity"]["factCohorts"] = fact_counts
    return result
