"""S18-LC source Condition coding selection before population allocation.

The caller declares a class for one source Condition record. This module does
not allocate the 5/30/65 mix or make local-only records conformant.
"""

from __future__ import annotations

import copy
from functools import lru_cache
from typing import Any

from .catalogs import load_catalog


TERMINOLOGY_CATALOG = "tools/synthetic_population/data/terminology.v2.4.0.json"
CLINICAL_CONTENT_CATALOG = "tools/synthetic_population/data/clinical-content.v4.5.0.json"
CLASS_SYSTEM = "urn:hdp-test-bundle:synthetic:condition-coding-class"
LOCAL_ONLY_KIND = "local-only-condition-coding"
SNOMED = "http://snomed.info/sct"
CLASSES = {"standard-only", "standard-plus-local", "local-only"}


@lru_cache(maxsize=1)
def condition_code_inventory() -> dict[str, dict[str, Any]]:
    """Join the two existing versioned catalogs and reject drift or aliases."""
    terminology = load_catalog(TERMINOLOGY_CATALOG)
    clinical = load_catalog(CLINICAL_CONTENT_CATALOG)
    code_system = terminology["localConditionCodeSystem"]
    concept_set = clinical["localConditionConcepts"]
    if (code_system["url"], code_system["version"]) != (concept_set["codeSystem"], concept_set["version"]):
        raise ValueError("local Condition CodeSystem identity differs between catalogs")
    rows = [row for row in terminology["localSyntheticCodings"] if row["system"] == code_system["url"]]
    concepts = concept_set["concepts"]
    if code_system["resourceType"] != "CodeSystem" or code_system["content"] != "complete" or not code_system["caseSensitive"]:
        raise ValueError("local Condition CodeSystem is not a complete case-sensitive FHIR CodeSystem")
    if len(rows) != len(code_system["concept"]) or len(concepts) != len(rows):
        raise ValueError("local Condition concept count differs between catalogs")
    by_code = {row["code"]: row for row in rows}
    if len(by_code) != len(rows) or len({row["code"] for row in concepts}) != len(concepts):
        raise ValueError("duplicate local Condition code")
    fhir_concepts = {row["code"]: row for row in code_system["concept"]}
    if len(fhir_concepts) != len(code_system["concept"]):
        raise ValueError("duplicate local Condition FHIR concept")
    mapped_standard: set[tuple[str, str]] = set()
    for concept in concepts:
        code = concept["code"]
        row = by_code.get(code)
        if row is None or any(row[field] != concept[field] for field in ("display", "definition", "aliases")):
            raise ValueError(f"local Condition semantic catalog mismatch: {code}")
        if row["version"] != code_system["version"] or row["resourceTypes"] != ["Condition"] or row["pathSuffixes"] != ["code.coding[*]"]:
            raise ValueError(f"invalid local Condition terminology role: {code}")
        if fhir_concepts.get(code) != {key: concept[key] for key in ("code", "display", "definition")}:
            raise ValueError(f"local Condition FHIR CodeSystem concept mismatch: {code}")
        if concept["paths"] != ["Condition.code"] or concept["compatibleRoles"] != ["condition"]:
            raise ValueError(f"invalid local Condition clinical role: {code}")
        if concept["aliases"] or row["aliases"]:
            raise ValueError(f"local Condition aliases would inflate clinical breadth: {code}")
        mapping = concept["mapping"]
        if mapping["system"] != SNOMED or mapping["relationship"] != "reviewed-equivalent-by-local-definition":
            raise ValueError(f"local Condition cannot be selected as a coding for this source fact: {code}")
        if mapping["display"] != concept["display"]:
            raise ValueError(f"local Condition display changes mapped clinical meaning: {code}")
        if (mapping["system"], mapping["code"]) in mapped_standard:
            raise ValueError(f"duplicate mapped clinical concept: {code}")
        mapped_standard.add((mapping["system"], mapping["code"]))
        if code == mapping["code"] or code.isdecimal():
            raise ValueError(f"local Condition identifier copies a standard identifier: {code}")
        if not concept["expectedProminence"]["basis"] or not concept["eligibility"]["archetypes"] or not concept["provenance"]["method"]:
            raise ValueError(f"incomplete local Condition eligibility, prominence, or provenance: {code}")
    return {row["code"]: row for row in concepts}


def semantic_condition_codings(resource: dict[str, Any]) -> set[tuple[str, str]]:
    """Return concept-equivalent codes for native clinical audits only.

    Raw FHIR coding, profile validation, and terminology accounting continue to
    see the local-only Condition without SNOMED. The equivalent concept is
    admitted only for an exact version/display match to the reviewed mapping.
    """
    from .observable import codings

    observed = set(codings(resource))
    if resource.get("resourceType") != "Condition":
        return observed
    local = [row for row in resource.get("code", {}).get("coding", [])
             if row.get("system") == "urn:hdp-test-bundle:synthetic:condition"]
    if len(local) != 1:
        return observed
    declaration = condition_code_inventory().get(local[0].get("code"))
    if (declaration is not None and local[0].get("version") == "1.0.0"
            and local[0].get("display") == declaration["display"]):
        observed.add((declaration["mapping"]["system"], declaration["mapping"]["code"]))
    return observed


def select_condition_coding(
    resource: dict[str, Any], selection: dict[str, str], metadata: dict[str, str],
) -> dict[str, Any]:
    """Return an explicitly classified Condition while preserving its source fact.

    Selection is per emitted source record. The local-only class deliberately
    records the missing-SNOMED quality problem using the existing problem tags.
    """
    if resource.get("resourceType") != "Condition":
        raise ValueError("Condition coding selection requires a Condition")
    kind = selection.get("class")
    if kind not in CLASSES:
        raise ValueError(f"unknown Condition coding class: {kind}")
    codings = resource.get("code", {}).get("coding", [])
    if len(codings) != 1 or codings[0].get("system") != SNOMED or not codings[0].get("code"):
        raise ValueError("Condition coding selection requires one source SNOMED coding")
    standard = codings[0]
    if kind == "standard-only":
        if set(selection) != {"class"}:
            raise ValueError("standard-only cannot select a local code or problem ID")
        selected = [standard]
    else:
        required = {"class", "localCode"} | ({"problemId"} if kind == "local-only" else set())
        if set(selection) != required:
            raise ValueError(f"{kind} selection fields must be {sorted(required)}")
        concept = condition_code_inventory().get(selection["localCode"])
        if concept is None:
            raise ValueError("undeclared local Condition code")
        mapping = concept["mapping"]
        if (standard["system"], standard["code"], standard.get("display")) != (mapping["system"], mapping["code"], mapping["display"]):
            raise ValueError("local Condition code does not match the source clinical fact")
        profile = resource.get("meta", {}).get("profile", [])
        if not any(item in concept["profiles"] for item in profile):
            raise ValueError("local Condition code used outside its approved profile")
        tags = resource.get("meta", {}).get("tag", [])
        sources = [tag.get("code") for tag in tags if tag.get("system") == metadata["sourceTagSystem"]]
        archetypes = [tag.get("code") for tag in tags if tag.get("system") == "urn:hdp-test-bundle:synthetic-archetype"]
        eligibility = concept["eligibility"]
        if len(sources) != 1 or sources[0] not in eligibility["sources"] or len(archetypes) != 1 or archetypes[0] not in eligibility["archetypes"]:
            raise ValueError("local Condition code outside declared source/archetype eligibility")
        local = {
            "system": load_catalog(TERMINOLOGY_CATALOG)["localConditionCodeSystem"]["url"],
            "version": load_catalog(TERMINOLOGY_CATALOG)["localConditionCodeSystem"]["version"],
            "code": concept["code"], "display": concept["display"],
        }
        selected = [standard, local] if kind == "standard-plus-local" else [local]
        if kind == "local-only" and not selection["problemId"].strip():
            raise ValueError("local-only Condition requires a stable problem ID")
    result = copy.deepcopy(resource)
    result["code"]["coding"] = selected
    tags = result.setdefault("meta", {}).setdefault("tag", [])
    if any(tag.get("system") == CLASS_SYSTEM for tag in tags):
        raise ValueError("Condition coding class has already been assigned")
    tags.append({"system": CLASS_SYSTEM, "code": kind})
    if kind == "local-only":
        marker_system = metadata["problemMarkerSystem"]
        if any(tag.get("system") == marker_system for tag in tags):
            raise ValueError("local-only Condition already has an intentional-problem marker")
        tags.extend([
            {"system": marker_system, "code": "intentional-problem"},
            {"system": metadata["problemKindSystem"], "code": LOCAL_ONLY_KIND},
            {"system": metadata["problemIdSystem"], "code": selection["problemId"]},
        ])
    return result
