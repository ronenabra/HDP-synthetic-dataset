from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .utils import read_json


def iter_codings(value: Any, path: str = "") -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        if isinstance(value.get("system"), str) and isinstance(value.get("code"), str):
            yield path, value
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            yield from iter_codings(child, child_path)
    elif isinstance(value, list):
        for child in value:
            yield from iter_codings(child, f"{path}[*]")


def _local_code_systems(repository_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    systems: dict[str, dict[str, dict[str, Any]]] = {}
    for directory in (repository_root / "reference/ilcore", repository_root / "reference/hdp-il"):
        for path in directory.glob("CodeSystem-*.json"):
            resource = read_json(path)
            if resource.get("url"):
                systems.setdefault(resource["url"], {}).update({row["code"]: row for row in resource.get("concept", [])})
    return systems


def audit_terminology(bundles: dict[str, dict[str, Any]], spec: dict[str, Any], repository_root: Path) -> dict[str, Any]:
    catalog = read_json(repository_root / spec["terminology"]["catalog"])
    if "localSyntheticCodings" not in catalog:
        occurrence_count = sum(
            1
            for bundle in bundles.values()
            for entry in bundle["entry"]
            for _path, _coding in iter_codings(entry["resource"])
        )
        return {
            "schemaVersion": "1.0.0", "catalogVersion": catalog["catalogVersion"],
            "codingOccurrenceCount": occurrence_count,
            "accountedCodingOccurrenceCount": occurrence_count,
            "status": "pass", "errors": [],
            "counts": [{"system": "legacy-v1-contract", "provenanceClass": "legacy-catalog", "problemStatus": "mixed", "count": occurrence_count}],
        }
    local_rows = {(row["system"], row["code"]): row for row in catalog["localSyntheticCodings"]}
    external_rows: dict[tuple[str, str], dict[str, Any]] = {}
    scoped_rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def approve_role(system: str, code: str, display: str, resource_type: str, path: str, provenance: str) -> None:
        # Keep new permissions as exact resource/path pairs, not a Cartesian
        # product with a code's existing permissions in the legacy inventory.
        scoped_rows[(system, code, resource_type, path)] = {
            "canonicalDisplay": display, "aliases": [],
            "resourceTypes": [resource_type], "paths": [path],
            "provenanceClass": provenance, "problemStatus": "ordinary",
        }
    for declared in catalog.get("approvedExternalCodings", []):
        key = (declared["system"], declared["code"])
        existing = external_rows.get(key)
        if existing is None:
            external_rows[key] = declared
            continue
        external_rows[key] = {
            **existing,
            "canonicalDisplay": existing.get("canonicalDisplay") or declared.get("canonicalDisplay"),
            "aliases": sorted(set(existing.get("aliases", [])) | set(declared.get("aliases", []))),
            "resourceTypes": sorted(set(existing["resourceTypes"]) | set(declared["resourceTypes"])),
            "paths": sorted(set(existing["paths"]) | set(declared["paths"])),
            "problemStatus": (
                existing["problemStatus"]
                if existing["problemStatus"] == declared["problemStatus"]
                else "mixed"
            ),
        }
    if spec.get("clinicalRealism"):
        realism = read_json(repository_root / spec["clinicalRealism"]["catalog"])
        for added in realism.get("ordinaryCodings", []):
            key = (added["system"], added["code"])
            existing = external_rows.get(key)
            if existing:
                external_rows[key] = {
                    **existing,
                    "resourceTypes": sorted(set(existing["resourceTypes"]) | set(added["resourceTypes"])),
                    "paths": sorted(set(existing["paths"]) | set(added["paths"])),
                    "aliases": sorted(set(existing.get("aliases", [])) | set(added.get("aliases", []))),
                    "problemStatus": existing["problemStatus"] if existing["problemStatus"] == added["problemStatus"] else "mixed",
                }
            else:
                external_rows[key] = added
    if spec.get("remediation"):
        # v4's reviewed pathway grammar is also an executable terminology
        # inventory. Only the exact observable locations emitted by the typed
        # projection are admitted; unused catalog fields cannot silently pass.
        pathway = read_json(repository_root / spec["remediation"]["pathwayTemplateCatalog"])
        role_paths = {
            "encounter": ("Encounter", "reasonCode[*].coding[*]"),
            "emergency-encounter": ("Encounter", "reasonCode[*].coding[*]"),
            "inpatient-encounter": ("Encounter", "reasonCode[*].coding[*]"),
            "follow-up": ("Encounter", "reasonCode[*].coding[*]"),
            "condition": ("Condition", "code.coding[*]"),
            "assessment": ("Observation", "code.coding[*]"),
            "order": ("ServiceRequest", "code.coding[*]"),
            "referral": ("ServiceRequest", "code.coding[*]"),
            "collection": ("Specimen", "type.coding[*]"),
            "result": ("Observation", "code.coding[*]"),
            "growth-measurement": ("Observation", "code.coding[*]"),
            "prescription": ("MedicationRequest", "medicationCodeableConcept.coding[*]"),
            "dispense": ("MedicationDispense", "medicationCodeableConcept.coding[*]"),
            "medication-review": ("MedicationStatement", "medicationCodeableConcept.coding[*]"),
            "medication-administration": ("MedicationAdministration", "medicationCodeableConcept.coding[*]"),
            "procedure": ("Procedure", "code.coding[*]"),
            "oncology-treatment": ("Procedure", "code.coding[*]"),
            "immunization": ("Immunization", "vaccineCode.coding[*]"),
            "discharge": ("DocumentReference", "type.coding[*]"),
        }
        for variants in pathway["archetypes"].values():
            for variant in variants:
                for event in variant["events"]:
                    resource_type, coding_path = role_paths[event["role"]]
                    key = (event["system"], event["code"])
                    addition = {
                        "system": key[0], "code": key[1],
                        "canonicalDisplay": event["display"], "aliases": [],
                        "resourceTypes": [resource_type], "paths": [coding_path],
                        "provenanceClass": "reviewed-v4-pathway-grammar",
                        "problemStatus": "ordinary",
                    }
                    existing = external_rows.get(key)
                    if existing:
                        addition["resourceTypes"] = sorted(set(existing["resourceTypes"]) | {resource_type})
                        addition["paths"] = sorted(set(existing["paths"]) | {coding_path})
                        addition["aliases"] = sorted(set(existing.get("aliases", [])) | {existing.get("canonicalDisplay"), event["display"]} - {None})
                        addition["problemStatus"] = "mixed" if existing["problemStatus"] != "ordinary" else "ordinary"
                    external_rows[key] = addition
        # Required IL Core encounter participant slice, from the locked official profile.
        external_rows[("http://terminology.hl7.org/CodeSystem/v3-ParticipationType", "PPRF")] = {
            "canonicalDisplay": "primary performer", "aliases": [],
            "resourceTypes": ["Encounter"], "paths": ["participant[*].type[*].coding[*]"],
            "provenanceClass": "locked-il-core-encounter-primary-performer", "problemStatus": "ordinary",
        }
        external_rows[("http://terminology.hl7.org/CodeSystem/v3-ActCode", "FFC")] = {
            "canonicalDisplay": "First Fill - Complete", "aliases": [],
            "resourceTypes": ["MedicationDispense"], "paths": ["type.coding[*]"],
            "provenanceClass": "locked-r4-act-code", "problemStatus": "ordinary",
        }
        approve_role("http://terminology.hl7.org/CodeSystem/v3-ActCode", "RFC",
                     "Refill - Complete", "MedicationDispense", "type.coding[*]", "locked-r4-act-code")
        approve_role("http://terminology.hl7.org/CodeSystem/data-absent-reason", "not-performed",
                     "Not Performed", "Observation", "dataAbsentReason.coding[*]", "locked-r4-data-absent-reason")
        dose_unit = external_rows[("http://unitsofmeasure.org", "mg")]
        dose_unit["resourceTypes"] = sorted(set(dose_unit["resourceTypes"]) | {"MedicationDispense"})
        dose_unit["paths"] = sorted(set(dose_unit["paths"]) | {"dosageInstruction[*].doseAndRate[*].doseQuantity"})
        # The emitted ServiceRequest carries the preceding clinical indication.
        for variants in pathway["archetypes"].values():
            for variant in variants:
                for position, event in enumerate(variant["events"]):
                    if event["role"] not in {"order", "referral"}:
                        continue
                    indication = variant["events"][max(0, position - 1)]
                    key = (indication["system"], indication["code"])
                    external_rows[key]["resourceTypes"] = sorted(set(external_rows[key]["resourceTypes"]) | {"ServiceRequest"})
                    external_rows[key]["paths"] = sorted(set(external_rows[key]["paths"]) | {"reasonCode[*].coding[*]"})
        # The renal potassium monitoring order is an approved secondary
        # clinical branch. Its CKD indication is linked by reasonReference and
        # is also required as reasonCode by the official ServiceRequest profile.
        renal_reason = external_rows[("http://snomed.info/sct", "709044004")]
        renal_reason["resourceTypes"] = sorted(set(renal_reason["resourceTypes"]) | {"ServiceRequest"})
        renal_reason["paths"] = sorted(set(renal_reason["paths"]) | {"reasonCode[*].coding[*]"})
        referral = external_rows[("http://snomed.info/sct", "306206005")]
        referral["paths"] = sorted(set(referral["paths"]) | {"category[*].coding[*]"})
        content = read_json(repository_root / spec["remediation"]["clinicalContentCatalog"])
        from .clinical_semantics import CONTRACT_PATH
        observable = read_json(repository_root / CONTRACT_PATH)
        from .clinical_state import CONTRACT as state_contract
        from .medication_safety import CONTRACT as medication_contract
        replacement = medication_contract["renalReplacement"]["to"]
        replacement_display = medication_contract["additionalRegimens"][replacement]["display"]
        for resource_type in ("MedicationRequest", "MedicationDispense", "MedicationStatement", "MedicationAdministration"):
            approve_role("http://www.nlm.nih.gov/research/umls/rxnorm", replacement,
                         replacement_display, resource_type, "medicationCodeableConcept.coding[*]",
                         "reviewed-s21-renal-medication-replacement")
        approve_role("http://loinc.org", state_contract["hiv"]["virologyCode"],
                     "HIV 1 RNA [Presence] in Specimen by NAA with probe detection",
                     "Observation", "code.coding[*]", "reviewed-s20-infant-virology")
        # R4 oxygensat requires 2708-6 alongside the more specific pulse-ox code.
        approve_role("http://loinc.org", "2708-6", "Oxygen saturation in Arterial blood",
                     "Observation", "code.coding[*]", "locked-r4-oxygen-saturation-code")
        approve_role("http://terminology.hl7.org/CodeSystem/data-absent-reason", "unknown", "Unknown",
                     "Observation", "dataAbsentReason.coding[*]", "locked-r4-data-absent-reason")
        perioperative = observable["perioperativeRegimens"]["2180"]
        approve_role("http://www.nlm.nih.gov/research/umls/rxnorm", "2180", perioperative["display"],
                     "MedicationAdministration", "medicationCodeableConcept.coding[*]",
                     "reviewed-s22-perioperative-prophylaxis")
        approve_role("http://snomed.info/sct", perioperative["routeCode"], "Intravenous route",
                     "MedicationAdministration", "dosage.route.coding[*]",
                     "reviewed-s22-perioperative-prophylaxis")
        approve_role("http://snomed.info/sct", "50697003", "General anesthesia",
                     "Procedure", "code.coding[*]", "reviewed-s22-perioperative-anesthesia")
        analytes = {row["code"]: row for row in content["analytes"]}
        for codes in observable["branchTests"].values():
            for code in codes:
                display = (analytes[code]["display"] if code in analytes else
                           {"58410-2": "CBC panel - Blood by Automated count",
                            "59408-5": "Oxygen saturation by pulse oximetry"}[code])
                for resource_type in (["Observation"] if code == "59408-5" else ["Observation", "ServiceRequest"]):
                    approve_role("http://loinc.org", code, display, resource_type,
                                 "code.coding[*]", "reviewed-s22-observable-branch-test")
        # add_branch_event binds the branch order's indication to the last
        # primary event, not the event preceding an order in the base template.
        # fhir_projection emits that exact fact as ServiceRequest.reasonCode.
        for variants in pathway["archetypes"].values():
            for variant in variants:
                if not any(any(code != "59408-5" for code in observable["branchTests"].get(branch, []))
                           for branch in variant.get("secondaryBranches", [])):
                    continue
                indication = variant["events"][-1]
                approve_role(indication["system"], indication["code"], indication["display"],
                             "ServiceRequest", "reasonCode[*].coding[*]", "reviewed-s22-branch-indication")
        for analyte in content["analytes"]:
            key = (analyte["system"], analyte["code"])
            existing = external_rows.get(key, {})
            external_rows[key] = {
                "system": key[0], "code": key[1],
                "canonicalDisplay": analyte["display"],
                "aliases": sorted(set(existing.get("aliases", [])) | {existing.get("canonicalDisplay")} - {None}),
                "resourceTypes": sorted(set(existing.get("resourceTypes", [])) | {"Observation"}),
                "paths": sorted(set(existing.get("paths", [])) | {"code.coding[*]"}),
                "provenanceClass": "reviewed-v4-clinical-content",
                "problemStatus": "mixed" if existing and existing.get("problemStatus") != "ordinary" else "ordinary",
            }
        for medication in content["medications"]:
            key = (medication["system"], medication["code"])
            existing = external_rows.get(key, {})
            external_rows[key] = {
                "system": key[0], "code": key[1],
                "canonicalDisplay": medication["display"],
                "aliases": sorted(set(existing.get("aliases", [])) | {existing.get("canonicalDisplay")} - {None}),
                "resourceTypes": sorted(set(existing.get("resourceTypes", [])) | {"MedicationRequest"}),
                "paths": sorted(set(existing.get("paths", [])) | {"medicationCodeableConcept.coding[*]"}),
                "provenanceClass": "reviewed-v4-clinical-content",
                "problemStatus": "mixed" if existing and existing.get("problemStatus") != "ordinary" else "ordinary",
            }
        for branch in content.get("secondaryClinicalBranches", []):
            condition = branch["condition"]
            key = (condition["system"], condition["code"])
            existing = external_rows.get(key, {})
            external_rows[key] = {
                "system": key[0], "code": key[1],
                "canonicalDisplay": condition["display"],
                "aliases": sorted(set(existing.get("aliases", [])) | {existing.get("canonicalDisplay")} - {None}),
                "resourceTypes": sorted(set(existing.get("resourceTypes", [])) | {"Condition"}),
                "paths": sorted(set(existing.get("paths", [])) | {"code.coding[*]"}),
                "provenanceClass": "reviewed-v4-secondary-clinical-branch",
                "problemStatus": "mixed" if existing and existing.get("problemStatus") != "ordinary" else "ordinary",
            }
        for vaccine in content.get("immunizations", []):
            key = (vaccine["system"], vaccine["code"])
            existing = external_rows.get(key, {})
            external_rows[key] = {
                "system": key[0], "code": key[1], "canonicalDisplay": vaccine["display"],
                "aliases": sorted(set(existing.get("aliases", [])) | {existing.get("canonicalDisplay")} - {None}),
                "resourceTypes": sorted(set(existing.get("resourceTypes", [])) | {"Immunization"}),
                "paths": sorted(set(existing.get("paths", [])) | {"vaccineCode.coding[*]"}),
                "provenanceClass": "reviewed-israeli-schedule-snomed-product",
                "problemStatus": "ordinary",
            }
        for unit in {row["unit"] for row in content["analytes"]}:
            key = ("http://unitsofmeasure.org", unit)
            existing = external_rows.get(key, {})
            external_rows[key] = {
                "system": key[0], "code": key[1], "canonicalDisplay": None,
                "aliases": existing.get("aliases", []),
                "resourceTypes": sorted(set(existing.get("resourceTypes", [])) | {"Observation"}),
                "paths": sorted(set(existing.get("paths", [])) | {
                    "valueQuantity", "referenceRange[*].low", "referenceRange[*].high",
                }),
                "provenanceClass": "reviewed-ucum-unit",
                "problemStatus": "mixed" if existing and existing.get("problemStatus") != "ordinary" else "ordinary",
            }
    local_code_systems = _local_code_systems(repository_root)
    counts: Counter[tuple[str, str, str]] = Counter()
    errors: list[str] = []
    occurrence_count = 0
    status_dispositions = []
    for organization, bundle in bundles.items():
        for entry in bundle["entry"]:
            resource = entry["resource"]
            problem = any(tag.get("system") == spec["metadata"]["problemMarkerSystem"] for tag in resource.get("meta", {}).get("tag", []))
            if resource.get("resourceType") == "Condition" and catalog.get("localConditionCodeSystem"):
                from .condition_coding import CLASS_SYSTEM, LOCAL_ONLY_KIND, SNOMED
                tags = resource.get("meta", {}).get("tag", [])
                classes = [tag.get("code") for tag in tags if tag.get("system") == CLASS_SYSTEM]
                local_system = catalog["localConditionCodeSystem"]["url"]
                code_rows = resource.get("code", {}).get("coding", [])
                local_count = sum(row.get("system") == local_system for row in code_rows)
                standard_count = sum(row.get("system") == SNOMED for row in code_rows)
                expected = {
                    "standard-only": (0, 1),
                    "standard-plus-local": (1, 1),
                    "local-only": (1, 0),
                }
                if (local_count or classes) and (
                    len(classes) != 1 or classes[0] not in expected
                    or (local_count, standard_count) != expected[classes[0]]
                ):
                    errors.append(f"{organization}:{entry['fullUrl']}: invalid Condition coding class or code composition")
                if classes == ["local-only"]:
                    marker = [tag.get("code") for tag in tags if tag.get("system") == spec["metadata"]["problemMarkerSystem"]]
                    kinds = [tag.get("code") for tag in tags if tag.get("system") == spec["metadata"]["problemKindSystem"]]
                    ids = [tag.get("code") for tag in tags if tag.get("system") == spec["metadata"]["problemIdSystem"]]
                    if marker != ["intentional-problem"] or LOCAL_ONLY_KIND not in kinds or len(ids) != 1:
                        errors.append(f"{organization}:{entry['fullUrl']}: local-only Condition lacks intentional-dirty attribution")
                elif local_count and problem:
                    errors.append(f"{organization}:{entry['fullUrl']}: standard-plus-local Condition marked as a local-only problem")
            for path, coding in iter_codings(resource):
                occurrence_count += 1
                system, code = coding["system"], coding["code"]
                classification = None
                if not problem and any(system.startswith(prefix) for prefix in catalog["forbiddenOrdinarySystemPrefixes"]):
                    errors.append(f"{organization}:{entry['fullUrl']}:{path}: untagged pedagogical example-domain coding {system}|{code}")
                    classification = "forbidden-example-domain"
                elif (system, code) in local_rows:
                    row = local_rows[(system, code)]
                    classification = row["provenanceClass"]
                    if resource["resourceType"] not in row["resourceTypes"] or not any(path.endswith(suffix) for suffix in row["pathSuffixes"]):
                        errors.append(f"{organization}:{entry['fullUrl']}:{path}: local coding used outside approved semantic role")
                    display = coding.get("display")
                    if row["display"] is not None and display not in {row["display"], *row.get("aliases", [])}:
                        errors.append(f"{organization}:{entry['fullUrl']}:{path}: wrong approved-local display for {system}|{code}")
                elif any(system.startswith(prefix) for prefix in catalog["projectControlPrefixes"]):
                    classification = "project-control"
                elif system in local_code_systems:
                    classification = "pinned-local-codesystem"
                    concept = local_code_systems[system].get(code)
                    if not concept:
                        errors.append(f"{organization}:{entry['fullUrl']}:{path}: unknown code in pinned local CodeSystem {system}|{code}")
                    elif concept.get("display") and coding.get("display") and coding["display"] != concept["display"]:
                        errors.append(f"{organization}:{entry['fullUrl']}:{path}: wrong normative display for {system}|{code}")
                elif (system, code) in external_rows or (system, code, resource["resourceType"], path) in scoped_rows:
                    row = scoped_rows.get((system, code, resource["resourceType"], path), external_rows.get((system, code)))
                    classification = row["provenanceClass"]
                    expected_problem_status = row["problemStatus"]
                    actual_problem_status = "intentional-problem" if problem else "ordinary"
                    if expected_problem_status not in {actual_problem_status, "mixed"}:
                        if False:
                            status_dispositions.append({"source": organization, "fullUrl": entry["fullUrl"],
                                "resourceType": resource["resourceType"], "path": path, "system": system, "code": code,
                                "expectedProblemStatus": expected_problem_status, "actualProblemStatus": actual_problem_status,
                                "disposition": "ordinary-approved-code-inside-exact-intentional-conflicting-diagnosis", "dirtyAttributionPreserved": True})
                        else:
                            errors.append(
                                f"{organization}:{entry['fullUrl']}:{path}: {system}|{code} used with unapproved problem status"
                            )
                    if resource["resourceType"] not in row["resourceTypes"] or path not in row["paths"]:
                        errors.append(
                            f"{organization}:{entry['fullUrl']}:{path}: {system}|{code} "
                            "external coding used outside approved semantic role"
                        )
                    display = coding.get("display")
                    expected_displays = {row.get("canonicalDisplay"), *row.get("aliases", [])} - {None}
                    if display is not None and expected_displays and display not in expected_displays:
                        errors.append(f"{organization}:{entry['fullUrl']}:{path}: wrong normative display for {system}|{code}")
                else:
                    policy = next((row for row in catalog["authoritativeSystems"] if system.startswith(row["prefix"])), None)
                    if policy:
                        classification = "unapproved-external-code"
                        errors.append(f"{organization}:{entry['fullUrl']}:{path}: unknown code in approved external system {system}|{code}")
                    else:
                        errors.append(f"{organization}:{entry['fullUrl']}:{path}: undeclared coding system {system}|{code}")
                        classification = "undeclared"
                counts[(system, classification, "intentional-problem" if problem else "ordinary")] += 1
    return {
        "schemaVersion": "2.0.0", "catalogVersion": catalog["catalogVersion"],
        "codingOccurrenceCount": occurrence_count, "accountedCodingOccurrenceCount": sum(counts.values()),
        "status": "pass" if not errors else "fail", "errors": errors,
        "counts": [
            {"system": system, "provenanceClass": provenance, "problemStatus": problem_status, "count": count}
            for (system, provenance, problem_status), count in sorted(counts.items())
        ],
    }
