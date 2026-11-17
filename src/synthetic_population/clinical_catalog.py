from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .catalogs import load_catalog
from .clinical_policy import age_applicability, birth_date_or_none, completed_age_years, unresolved_age_applicability
from .profiles import expected_profile, RESOURCE_PROFILES
from .fhir_projection import ROLE_RESOURCE_TYPE


MODEL_FIELDS = {
    "version", "aliases", "fhirResourceTypes", "profiles", "paths",
    "eligibility", "contraindications", "indications", "compatibleRoles",
    "valueModel", "interpretationRules", "repeatGroup", "correlationGroup",
    "captureExpectations", "provenance", "reviewedStatus", "uncertainty",
    "clinicalReviewQuestion", "contentClass",
}

ROW_FIELDS = {
    "analytes": {"system", "code", "display", "unit", "low", "high", "model"},
    "medications": {"system", "code", "display", "model"},
    "immunizations": {"system", "code", "display", "minimumAgeDays", "model"},
}

SECONDARY_BRANCH_FIELDS = {"trait", "eligibleArchetypes", "condition", "medication"}
SECONDARY_CONDITION_FIELDS = {"system", "code", "display", "status"}
SECONDARY_MEDICATION_FIELDS = {"system", "code"}
INTENSIVE_ARCHETYPES = {"chronic-care", "medication-intensive", "oncology", "rare-complex"}

AGE_APPLICABILITY_RULE_FIELDS = {
    "id", "minimumAgeYears", "maximumAgeYears", "referenceRangeHandling",
    "interpretationHandling", "ordinaryDistributionGuardApplicable",
    "requiresDerivedPair", "requiresPositiveFiniteValue",
}
AGE_APPLICABILITY_POLICY_FIELDS = {
    "policyVersion", "rules", "sourceEvidence", "reviewStatus", "pediatricDisposition",
}

CODE_PATHS = {
    "Encounter": "Encounter.reasonCode",
    "Condition": "Condition.code",
    "Observation": "Observation.code",
    "RiskAssessment": "RiskAssessment.method",
    "DiagnosticReport": "DiagnosticReport.code",
    "ServiceRequest": "ServiceRequest.code",
    "Specimen": "Specimen.type",
    "MedicationRequest": "MedicationRequest.medicationCodeableConcept",
    "MedicationDispense": "MedicationDispense.medicationCodeableConcept",
    "MedicationStatement": "MedicationStatement.medicationCodeableConcept",
    "MedicationAdministration": "MedicationAdministration.medicationCodeableConcept",
    "Procedure": "Procedure.code",
    "Immunization": "Immunization.vaccineCode",
    "DocumentReference": "DocumentReference.type",
}


def _profile_path_supported(repository_root: Path, lock: dict[str, Any], profile: str, path: str) -> bool:
    if profile.startswith("http://hl7.org/fhir/StructureDefinition/"):
        resource_type = profile.rsplit("/", 1)[-1]
        return path == resource_type or path.startswith(f"{resource_type}.")
    for package in lock["profiles"]:
        package_path = repository_root / package["path"]
        if not package_path.is_dir():
            continue
        for candidate in [package_path / f"StructureDefinition-{profile.rsplit('/', 1)[-1]}.json"]:
            try:
                import json
                definition = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if definition.get("url") != profile:
                continue
            return any(
                element.get("path") == path or (
                    element.get("path", "").endswith("[x]") and any(
                        element["path"][:-3] + row["code"][:1].upper() + row["code"][1:] == path
                        for row in element.get("type", [])
                    )
                )
                for element in definition.get("snapshot", {}).get("element", [])
            )
    return False


def _validate_age_applicability(content: dict[str, Any]) -> None:
    analyte_codes = {row["code"] for row in content["analytes"]}
    value_model = content["modelDefinitions"].get("quantitative-observation", {}).get("valueModel", {})
    policies = value_model.get("ageApplicability", {})
    if not isinstance(policies, dict):
        raise ValueError("quantitative age applicability must be an object")
    for code, policy in policies.items():
        if code not in analyte_codes:
            raise ValueError(f"age applicability references unknown analyte {code}")
        if set(policy) != AGE_APPLICABILITY_POLICY_FIELDS:
            raise ValueError(
                f"age applicability {code} fields mismatch: "
                f"unknown={sorted(set(policy)-AGE_APPLICABILITY_POLICY_FIELDS)}, "
                f"missing={sorted(AGE_APPLICABILITY_POLICY_FIELDS-set(policy))}"
            )
        if not policy["policyVersion"] or not policy["rules"] or not policy["sourceEvidence"]:
            raise ValueError(f"age applicability {code} has no version, rules, or source evidence")
        if policy["reviewStatus"] not in {"provisional-pending-designated-clinical-review", "reviewed"}:
            raise ValueError(f"age applicability {code} has an invalid review status")
        prior_max = 0.0
        open_ended = False
        for rule in sorted(policy["rules"], key=lambda item: item["minimumAgeYears"]):
            if set(rule) != AGE_APPLICABILITY_RULE_FIELDS:
                raise ValueError(
                    f"age applicability {code} rule fields mismatch: "
                    f"unknown={sorted(set(rule)-AGE_APPLICABILITY_RULE_FIELDS)}, "
                    f"missing={sorted(AGE_APPLICABILITY_RULE_FIELDS-set(rule))}"
                )
            minimum = rule["minimumAgeYears"]
            maximum = rule["maximumAgeYears"]
            if not isinstance(minimum, (int, float)) or minimum < 0 or minimum != prior_max:
                raise ValueError(f"age applicability {code} has a gap or invalid minimum at {minimum}")
            if maximum is not None and (not isinstance(maximum, (int, float)) or maximum <= minimum):
                raise ValueError(f"age applicability {code} has an invalid maximum at {maximum}")
            if open_ended:
                raise ValueError(f"age applicability {code} has a rule after an open-ended interval")
            prior_max = maximum if maximum is not None else minimum
            open_ended = maximum is None
            if rule["referenceRangeHandling"] not in {"required", "omit"}:
                raise ValueError(f"age applicability {code} has invalid reference-range handling")
            if rule["interpretationHandling"] not in {"required", "omit"}:
                raise ValueError(f"age applicability {code} has invalid interpretation handling")
            if not isinstance(rule["ordinaryDistributionGuardApplicable"], bool) \
                    or not isinstance(rule["requiresDerivedPair"], bool) \
                    or not isinstance(rule["requiresPositiveFiniteValue"], bool):
                raise ValueError(f"age applicability {code} has a non-boolean check flag")
        if not open_ended:
            raise ValueError(f"age applicability {code} does not cover older ages")


def _resolved_row(row: dict[str, Any], model: dict[str, Any], kind: str) -> dict[str, Any]:
    eligibility = dict(model["eligibility"])
    if kind == "immunizations":
        eligibility["minimumAgeDays"] = row["minimumAgeDays"]
    return {
        "kind": kind.removesuffix("s"),
        "system": row["system"], "version": model["version"],
        "code": row["code"], "canonicalDisplay": row["display"],
        "aliases": model["aliases"],
        "fhirResourceTypes": model["fhirResourceTypes"],
        "profiles": model["profiles"], "paths": model["paths"],
        "allowedUnits": [row["unit"]] if kind == "analytes" else [],
        "eligibility": eligibility, "contraindications": model["contraindications"],
        "indications": model["indications"], "compatibleRoles": model["compatibleRoles"],
        "valueModel": model["valueModel"], "interpretationRules": model["interpretationRules"],
        "repeatGroup": model["repeatGroup"], "correlationGroup": model["correlationGroup"],
        "captureExpectations": model["captureExpectations"], "provenance": model["provenance"],
        "reviewedStatus": model["reviewedStatus"], "uncertainty": model["uncertainty"],
        "clinicalReviewQuestion": model["clinicalReviewQuestion"],
        "contentClass": model["contentClass"],
        "ageApplicability": model.get("valueModel", {}).get("ageApplicability", {}).get(row["code"]),
        **({"referenceRange": {"low": row["low"], "high": row["high"]}} if kind == "analytes" else {}),
    }


def validate_clinical_content_catalog(
    content_path: str, pathway_path: str, repository_root: Path, lock: dict[str, Any],
) -> dict[str, Any]:
    content = load_catalog(content_path)
    expected_root = {
        "catalogVersion", "status", "modelDefinitions", "pathwayConcepts",
        "secondaryClinicalBranches", "branchConcepts", "analytes", "medications", "immunizations",
    }
    if "localConditionConcepts" in content:
        expected_root.add("localConditionConcepts")
        from .condition_coding import CLINICAL_CONTENT_CATALOG, condition_code_inventory
        if content.get("localConditionConcepts") != load_catalog(CLINICAL_CONTENT_CATALOG).get("localConditionConcepts"):
            raise ValueError("successor local Condition concept inventory differs from the reviewed inventory")
        condition_code_inventory()
    if "branchQuantitativeModels" in content:
        expected_root.add("branchQuantitativeModels")
        from .clinical_policy import validate_branch_quantitative_models
        validate_branch_quantitative_models(content)
    if set(content) != expected_root:
        raise ValueError(
            f"clinical content root fields mismatch: unknown={sorted(set(content)-expected_root)}, "
            f"missing={sorted(expected_root-set(content))}"
        )
    models = content["modelDefinitions"]
    _validate_age_applicability(content)
    branch_fields = {"id", "system", "code", "display", "roles", "paths", "specimen", "method", "eligibility", "provenance", "accessDate"}
    branch_ids = set()
    branch_rows = []
    for row in content["branchConcepts"]:
        if set(row) != branch_fields or not all(row.get(key) for key in ("id", "system", "code", "display", "roles", "paths", "eligibility", "provenance", "accessDate")):
            raise ValueError("incomplete S14A branch concept")
        if row["id"] in branch_ids or len(row["roles"]) != len(row["paths"]):
            raise ValueError("duplicate or unmapped S14A branch concept")
        branch_ids.add(row["id"])
        for role, path in zip(row["roles"], row["paths"]):
            resource_type = ROLE_RESOURCE_TYPE[role]
            if path != CODE_PATHS[resource_type]:
                raise ValueError(f"S14A branch concept role/path mismatch: {row['id']}/{role}/{path}")
            profile = ("http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-observation"
                       if row["id"] == "cervical-hpv-test" and role == "result"
                       else RESOURCE_PROFILES[resource_type])
            branch_rows.append({
                "kind": "branch-concept", "system": row["system"], "code": row["code"],
                "canonicalDisplay": row["display"], "fhirResourceTypes": [resource_type],
                "profiles": [profile], "paths": [path], "compatibleRoles": [role],
                "eligibility": row["eligibility"], "provenance": row["provenance"],
                "accessDate": row["accessDate"], "specimen": row["specimen"], "method": row["method"],
            })
    if branch_ids != {"cervical-hpv-test", "laparoscopic-appendectomy", "appendectomy-discharge-summary"}:
        raise ValueError("S14A branch concept inventory is incomplete")
    for model_id, model in models.items():
        if set(model) != MODEL_FIELDS:
            raise ValueError(
                f"clinical content model {model_id} fields mismatch: "
                f"unknown={sorted(set(model)-MODEL_FIELDS)}, missing={sorted(MODEL_FIELDS-set(model))}"
            )
        if not model["profiles"] or not model["paths"] or not model["compatibleRoles"]:
            raise ValueError(f"clinical content model {model_id} has an empty selector contract")
        if len(model["aliases"]) != len(set(model["aliases"])):
            raise ValueError(f"clinical content model {model_id} has duplicate/non-equivalent aliases")
    explicit_rows = list(branch_rows)
    semantic_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for kind, expected_fields in ROW_FIELDS.items():
        for row in content[kind]:
            if set(row) != expected_fields:
                raise ValueError(
                    f"clinical {kind} row {row.get('code')} fields mismatch: "
                    f"unknown={sorted(set(row)-expected_fields)}, missing={sorted(expected_fields-set(row))}"
                )
            if row["model"] not in models:
                raise ValueError(f"clinical row {row['code']} references unknown model {row['model']}")
            key = (row["system"], row["code"])
            if key in semantic_rows:
                raise ValueError(f"duplicate clinical semantic row: {key}")
            if kind == "analytes" and (
                not isinstance(row["low"], (int, float)) or not isinstance(row["high"], (int, float))
                or row["low"] >= row["high"] or not row["unit"]
            ):
                raise ValueError(f"conflicting unit/range for analyte {row['code']}")
            semantic_rows[key] = row
            explicit_rows.append(_resolved_row(row, models[row["model"]], kind))
    secondary_traits = set()
    medication_keys = {(row["system"], row["code"]) for row in content["medications"]}
    for branch in content["secondaryClinicalBranches"]:
        if set(branch) != SECONDARY_BRANCH_FIELDS:
            raise ValueError(f"secondary clinical branch fields mismatch: {branch.get('trait')}")
        trait = branch["trait"]
        if trait in secondary_traits:
            raise ValueError(f"duplicate secondary clinical branch: {trait}")
        secondary_traits.add(trait)
        if set(branch["eligibleArchetypes"]) != INTENSIVE_ARCHETYPES:
            raise ValueError(f"secondary branch {trait} has incomplete intensive archetype eligibility")
        condition = branch["condition"]
        if set(condition) != SECONDARY_CONDITION_FIELDS or condition["status"] not in {"active", "resolved"}:
            raise ValueError(f"secondary branch {trait} has an invalid condition contract")
        medication = branch["medication"]
        if medication is not None:
            if set(medication) != SECONDARY_MEDICATION_FIELDS:
                raise ValueError(f"secondary branch {trait} has an invalid medication selector")
            if (medication["system"], medication["code"]) not in medication_keys:
                raise ValueError(f"secondary branch {trait} references an unknown medication")
        explicit_rows.append({
            "kind": "secondary-condition", "system": condition["system"],
            "code": condition["code"], "canonicalDisplay": condition["display"],
            "status": condition["status"], "eligibleArchetypes": branch["eligibleArchetypes"],
            "secondaryTrait": trait, "fhirResourceTypes": ["Condition"],
            "profiles": [RESOURCE_PROFILES["Condition"]],
            "paths": ["Condition.code"], "compatibleRoles": ["condition"],
            "provenance": "reviewed-v4-secondary-clinical-branch",
        })
    if secondary_traits != {"metabolic-risk", "respiratory-history", "medication-exposure"}:
        raise ValueError("secondary clinical branch catalog must define the three supported traits")
    pathway_pointer = content["pathwayConcepts"]
    if set(pathway_pointer) != {"catalog", "model"} or pathway_pointer["catalog"] != pathway_path:
        raise ValueError("clinical pathway concept pointer is missing, unused, or points at another catalog")
    if pathway_pointer["model"] not in models:
        raise ValueError("clinical pathway concept pointer references an unknown model")
    pathway = load_catalog(pathway_path)
    pathway_model = models[pathway_pointer["model"]]
    pathway_rows = {}
    occurrences: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for archetype, variants in pathway["archetypes"].items():
        for variant in variants:
            for event in variant["events"]:
                key = (event["system"], event["code"])
                prior = pathway_rows.get(key)
                if prior and prior["canonicalDisplay"] != event["display"]:
                    raise ValueError(f"conflicting pathway display for {key}")
                resource_types = [ROLE_RESOURCE_TYPE[event["role"]]]
                paths = [CODE_PATHS[resource_type] for resource_type in resource_types]
                profiles = [expected_profile({
                    "resourceType": resource_type, "code": {"coding": [{"system": event["system"], "code": event["code"]}]},
                    "class": {"code": "IMP" if event["role"] == "inpatient-encounter" else "EMER" if event["role"] == "emergency-encounter" else "AMB"},
                }, "hmo" if event["setting"] == "community" else "hospital-a") for resource_type in resource_types]
                pathway_rows[key] = {
                    **_resolved_row({
                        "system": event["system"], "code": event["code"],
                        "display": event["display"], "model": pathway_pointer["model"],
                    }, pathway_model, "pathway-events"),
                    "fhirResourceTypes": resource_types, "profiles": profiles, "paths": paths,
                    "compatibleRoles": sorted(set(
                        pathway_rows.get(key, {}).get("compatibleRoles", []) + [event["role"]]
                    )),
                }
                occurrences[key].append({
                    "archetype": archetype, "variant": variant["id"],
                    "role": event["role"], "setting": event["setting"],
                })
    local_concepts = content.get("localConditionConcepts", {}).get("concepts", [])
    allowed_condition_keys = {
        (row["system"], row["code"])
        for row in pathway_rows.values() if "condition" in row["compatibleRoles"]
    } | {
        (branch["condition"]["system"], branch["condition"]["code"])
        for branch in content["secondaryClinicalBranches"]
    }
    for concept in local_concepts:
        mapping = concept["mapping"]
        if (mapping["system"], mapping["code"]) not in allowed_condition_keys:
            raise ValueError(f"local Condition mapping is outside the intended clinical mix: {concept['code']}")
        explicit_rows.append({
            "kind": "local-condition", "system": content["localConditionConcepts"]["codeSystem"],
            "code": concept["code"], "canonicalDisplay": concept["display"],
            "fhirResourceTypes": concept["resourceTypes"], "profiles": concept["profiles"],
            "paths": concept["paths"], "compatibleRoles": concept["compatibleRoles"],
            "eligibility": concept["eligibility"], "expectedProminence": concept["expectedProminence"],
            "mapping": mapping, "provenance": concept["provenance"],
        })
    unsupported_paths = []
    for row in explicit_rows + list(pathway_rows.values()):
        for path in row["paths"]:
            resource_type = path.split(".", 1)[0]
            profile = next((
                candidate for candidate in row["profiles"]
                if candidate.rsplit("/", 1)[-1] == resource_type
                or candidate == RESOURCE_PROFILES.get(resource_type)
                or candidate == "http://fhir.health.gov.il/StructureDefinition/il-core-immunization"
                and resource_type == "Immunization"
            ), row["profiles"][0])
            if not _profile_path_supported(repository_root, lock, profile, path):
                unsupported_paths.append({"profile": profile, "path": path, "code": row["code"]})
    if unsupported_paths:
        raise ValueError(f"clinical catalog paths are unsupported by locked profiles: {unsupported_paths[:5]}")
    return {
        "schemaVersion": "4.0.0", "status": "pass",
        "explicitRowCount": len(explicit_rows), "pathwaySemanticRowCount": len(pathway_rows),
        "resolvedRowCount": len(explicit_rows) + len(pathway_rows),
        "modelCount": len(models), "duplicateSemanticRowCount": 0,
        "localConditionConceptCount": len(local_concepts),
        "conflictingUnitRangeCount": 0, "unsupportedProfilePathCount": 0,
        "rows": explicit_rows,
        "pathwayRows": [
            {**row, "occurrences": occurrences[key]}
            for key, row in sorted(pathway_rows.items())
        ],
    }
