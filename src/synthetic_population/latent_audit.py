from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .catalogs import load_catalog
from .capture import DISPOSITIONS, capture_coverage_fixtures, load_capabilities, observable_origin, capture_field_present, metadata_content_leaks
from .condition_coding import semantic_condition_codings
from .observable import codings, patient_graphs, references
from .temporal import occurrence_from_resource, parse_fhir_time


ADMINISTRATIVE_TYPES = {
    "Patient", "RelatedPerson", "Organization", "Practitioner", "PractitionerRole",
}


ROLE_STATUSES = {
    "encounter": {"finished"}, "emergency-encounter": {"finished"},
    "inpatient-encounter": {"finished"}, "follow-up": {"finished"},
    "condition": {"active", "resolved"}, "assessment": {"final"},
    "order": {"completed"}, "referral": {"active", "completed"},
    "collection": {"available"}, "result": {"final"},
    "growth-measurement": {"final"}, "prescription": {"active", "stopped", "completed"},
    "dispense": {"completed"}, "medication-review": {"active", "stopped"},
    "medication-administration": {"completed"}, "procedure": {"completed"},
    "oncology-treatment": {"completed"}, "immunization": {"completed"},
    "discharge": {"current"},
}


def _observable_statuses(resource: dict[str, Any]) -> set[str]:
    values = {str(resource.get("status", ""))}
    values.update(code for system, code in codings(resource.get("clinicalStatus")))
    return values - {""}


def _link_field_supported(resource_type: str, target_type: str) -> bool:
    return (
        (resource_type == "Encounter" and target_type in {"Condition", "Procedure", "Observation"})
        or (resource_type == "Observation" and target_type in {"ServiceRequest", "Specimen"})
        or (resource_type == "Specimen" and target_type == "ServiceRequest")
        or (resource_type == "MedicationDispense" and target_type == "MedicationRequest")
        or (resource_type == "MedicationStatement" and target_type in {"MedicationRequest", "MedicationDispense"})
        or (resource_type == "MedicationRequest" and target_type == "Condition")
        or (resource_type == "Procedure" and target_type in {"CarePlan", "ServiceRequest", "Condition", "Observation", "DiagnosticReport", "DocumentReference"})
    )


def _index_event_source(mappings: list[dict]) -> dict:
    index = defaultdict(list)
    for row in mappings:
        index[(row["eventId"], row["source"])].append(row)
    return index


def build_latent_mapping_audit(
    bundles: dict[str, dict[str, Any]], crosswalk: dict[str, Any],
    ledger: list[dict[str, Any]], spec: dict[str, Any],
) -> dict[str, Any]:
    latent = {row["eventId"]: row for row in ledger if row.get("rowType") == "latent-event"}
    captures: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mappings = [row for row in ledger if row.get("rowType") == "fhir-mapping"]
    supplemental = [row for row in ledger if row.get("rowType") == "supplemental-source-capture"]
    for row in ledger:
        if row.get("rowType") == "source-capture":
            captures[row["latent_event_id"]].append(row)
    mapping_by_key = {(row["source"], row["resourceFullUrl"]): row for row in mappings}
    supplemental_by_key = {(row["source"], row["resourceFullUrl"]): row for row in supplemental}
    mapping_by_event_source = _index_event_source(mappings)
    lookup_count = 0
    mapping_counts = Counter((row["source"], row["resourceFullUrl"]) for row in mappings)
    resources = {
        (source, entry["fullUrl"]): entry["resource"]
        for source, bundle in bundles.items() for entry in bundle["entry"]
    }
    capabilities = load_capabilities(spec["remediation"]["capturePolicyCatalog"])
    fixture_evidence = capture_coverage_fixtures(
        spec["remediation"]["capturePolicyCatalog"], spec["seed"],
    )
    patient_urls = {
        (source, entry["fullUrl"])
        for source, bundle in bundles.items() for entry in bundle["entry"]
        if entry["resource"].get("resourceType") == "Patient"
    }
    graphs, _patients = patient_graphs(bundles, spec)
    declarations = {row["cohortId"]: row for row in crosswalk["patients"]}
    ordinary_urls = {
        (resource["_source"], resource["_fullUrl"])
        for cohort, rows in graphs.items()
        if declarations.get(cohort, {}).get("ordinaryPathwayEligible", True)
        for resource in rows
        if resource.get("resourceType") not in ADMINISTRATIVE_TYPES
    }
    errors = []
    agreement_counts = Counter()
    agreement_denominators = Counter()
    compatibility_errors: dict[str, list[str]] = {
        "encounterReason": [], "orderResult": [], "medicationLifecycle": [],
        "procedureFollowUp": [], "observableLinks": [], "captureContract": [],
    }
    for event_id, row in latent.items():
        event_captures = captures.get(event_id, [])
        if len(event_captures) != 3 or {item["source"] for item in event_captures} != set(bundles):
            errors.append(f"{event_id}: expected exactly one disposition per source")
        for capture in event_captures:
            expected = 0 if capture["disposition"] == "absent" else 1
            lookup_count += 1
            actual = len(mapping_by_event_source.get((event_id, capture["source"]), ()))
            if actual != expected:
                errors.append(f"{event_id}/{capture['source']}: mapping count {actual} != {expected}")
    for key, count in mapping_counts.items():
        if count != 1:
            errors.append(f"{key[0]}/{key[1]}: duplicate mapping")
    capture_catalog = load_catalog(load_catalog(spec["clinicalRealism"]["catalog"])["sourceCaptureCatalog"]) if spec.get("clinicalRealism") else {}
    capture_branches = {
        (source, branch["id"]): branch
        for source, row in capture_catalog.get("sources", {}).items()
        for branch in row["branches"]
    }
    supplemental_groups: dict[tuple[str, str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    if len(supplemental_by_key) != len(supplemental):
        errors.append("duplicate supplemental source-capture mapping")
    for key, row in supplemental_by_key.items():
        resource = resources.get(key)
        if key in mapping_by_key or key not in ordinary_urls or resource is None:
            errors.append(f"{key[0]}/{key[1]}: invalid supplemental source-capture mapping")
            continue
        if resource.get("resourceType") != row["resourceType"] or (key[0], row["captureVariant"]) not in capture_branches:
            errors.append(f"{key[0]}/{key[1]}: supplemental capture type or branch differs")
        if not any(tag.get("system") == "urn:hdp-test-bundle:synthetic-capture-variant" and tag.get("code") == row["captureVariant"] for tag in resource.get("meta", {}).get("tag", [])):
            errors.append(f"{key[0]}/{key[1]}: supplemental capture tag differs")
        if resource.get("meta", {}).get("source") != f"urn:hdp-test-bundle:source:{key[0]}":
            errors.append(f"{key[0]}/{key[1]}: supplemental capture lacks source provenance")
        observed_occurrence = occurrence_from_resource(resource)
        declared_occurrence = row.get("occurrence")
        if bool(observed_occurrence) != bool(declared_occurrence) or (
            observed_occurrence is not None and declared_occurrence is not None and (
                observed_occurrence.interval.start != parse_fhir_time(declared_occurrence["start"]).start
                or observed_occurrence.interval.end != parse_fhir_time(declared_occurrence["end"]).start
            )
        ):
            errors.append(f"{key[0]}/{key[1]}: supplemental capture occurrence differs")
        supplemental_groups[(row["cohortId"], key[0], row["captureVariant"])].append((key[1], resource))
    for (_cohort, source, variant), members in supplemental_groups.items():
        branch = capture_branches.get((source, variant))
        if not branch:
            continue
        expected_types = set(branch["resourceTypes"])
        actual_types = {resource["resourceType"] for _url, resource in members}
        if actual_types != expected_types or sum(resource["resourceType"] == "Encounter" for _url, resource in members) != 1:
            errors.append(f"{source}/{variant}: supplemental capture graph has incorrect resource types")
        cancelled = variant in {"cancelled-order", "cancelled-diagnostic"}
        requests = [(url, resource) for url, resource in members if resource["resourceType"] == "ServiceRequest"]
        for url, request in requests:
            if cancelled and request.get("status") != "revoked":
                errors.append(f"{source}/{url}: cancelled capture order is not revoked")
            if cancelled and any(url in set(references(resource.get("basedOn", []))) for (_other_source, _other_url), resource in resources.items() if _other_source == source and resource.get("resourceType") == "Observation"):
                errors.append(f"{source}/{url}: cancelled capture order has a performed result")
        if variant in {"summary-only", "transfer-summary", "discharge-follow-up"} and "Procedure" in actual_types:
            errors.append(f"{source}/{variant}: summary capture invents a performed procedure")
        if variant in {"external-result", "external-specimen-result", "received-result"}:
            for _url, resource in members:
                if resource["resourceType"] == "Observation" and any(ref in {url for url, _row in members} for ref in references({"basedOn": resource.get("basedOn"), "specimen": resource.get("specimen")})):
                    errors.append(f"{source}/{variant}: external result claims a local order or specimen")
    unmapped = sorted(ordinary_urls - set(mapping_by_key) - set(supplemental_by_key))
    if unmapped:
        errors.extend(f"{source}/{url}: ordinary clinical resource is unmapped" for source, url in unmapped[:50])
    for key, mapping in mapping_by_key.items():
        event = latent.get(mapping["eventId"])
        resource = resources.get(key)
        if event is None or resource is None:
            errors.append(f"{key[0]}/{key[1]}: mapping target is missing")
            continue
        if not declarations.get(mapping["cohortId"], {}).get("ordinaryPathwayEligible", True):
            continue
        fact = event["fact"]
        owner = observable_origin(resource, key[0])
        if owner != event["routedSource"] or mapping.get("originSource") != event["routedSource"]:
            compatibility_errors["captureContract"].append(f"{key[0]}/{key[1]}: performing/authoring origin {owner} differs from {event['routedSource']}")
        metadata_only = mapping["disposition"] == "metadata-only"
        if metadata_only and metadata_content_leaks(resource):
            compatibility_errors["captureContract"].append(f"{key[0]}/{key[1]}: metadata-only capture leaks clinical content")
        coded_match = (fact["system"], fact["code"]) in semantic_condition_codings(resource) if fact["system"] and fact["code"] else False
        text_match = (not fact["system"] and not fact["code"] and
                      any(resource.get(field, {}).get("text") == fact["display"] for field in ("code", "type")))
        if not metadata_only and not (coded_match or text_match):
            errors.append(f"{key[0]}/{key[1]}: observable code differs from latent fact")
        elif not metadata_only:
            agreement_counts["codeFamily"] += 1
        if not metadata_only and fact.get("coded_value") is not None:
            expected_value = fact["coded_value"]
            if (expected_value["system"], expected_value["code"]) not in set(codings(resource.get("valueCodeableConcept"))):
                errors.append(f"{key[0]}/{key[1]}: coded result differs from latent fact")
            if any(key.startswith("value") and key != "valueCodeableConcept" for key in resource):
                errors.append(f"{key[0]}/{key[1]}: multiple incompatible result value fields")
        if not metadata_only and resource.get("resourceType") == "Observation" and fact.get("result_text") is not None and resource.get("valueString") != fact["result_text"]:
            errors.append(f"{key[0]}/{key[1]}: narrative result differs from latent fact")
        if resource.get("resourceType") != mapping["resourceType"]:
            errors.append(f"{key[0]}/{key[1]}: resource type differs from mapping")
        else:
            agreement_counts["resourceType"] += 1
        occurrence = occurrence_from_resource(resource)
        expected_start = parse_fhir_time(fact["medication_detail"]["exposureStart"] if resource.get("resourceType") == "MedicationStatement" and fact.get("medication_detail") else event["occurrence"]["start"])
        if occurrence is None or occurrence.interval.start != expected_start.start:
            errors.append(f"{key[0]}/{key[1]}: clinical occurrence differs from latent event")
        else:
            agreement_counts["occurrenceInterval"] += 1
        if not metadata_only and resource.get("resourceType") == "Observation" and fact.get("value") is not None:
            node = next((row for row in resource.get("component", []) if (fact["system"], fact["code"]) in set(codings(row.get("code")))), resource)
            if node.get("valueQuantity", {}).get("value") != fact["value"]:
                errors.append(f"{key[0]}/{key[1]}: final quantity differs from latent fact")
        if not metadata_only and resource.get("resourceType", "").startswith("Medication"):
            if (fact["system"], fact["code"]) not in set(codings(resource.get("medicationCodeableConcept"))):
                errors.append(f"{key[0]}/{key[1]}: medication ingredient differs from latent fact")
            if resource.get("status") != fact["status"]:
                errors.append(f"{key[0]}/{key[1]}: medication exposure status differs from latent fact")
            detail = fact.get("medication_detail")
            if detail and resource["resourceType"] != "MedicationAdministration":
                dosage = resource.get("dosageInstruction", resource.get("dosage", [{}]))[0]
                if dosage.get("doseAndRate", [{}])[0].get("doseQuantity", {}).get("value") != detail["doseMg"]:
                    errors.append(f"{key[0]}/{key[1]}: medication dose differs from latent fact")
                expected_period = {"start": detail["exposureStart"], **({"end": detail["exposureEnd"]} if detail.get("exposureEnd") else {})}
                if dosage.get("timing", {}).get("repeat", {}).get("boundsPeriod") != expected_period:
                    errors.append(f"{key[0]}/{key[1]}: medication exposure interval differs from latent fact")
        allowed_statuses = {"current"} if metadata_only else {"active"} if mapping["disposition"] == "referral-only" else ROLE_STATUSES[event["eventRole"]]
        if fact.get("status") == "unsupported-age-method" and event["eventRole"] == "result":
            allowed_statuses = {"cancelled"}
            if resource.get("dataAbsentReason", {}).get("coding", [{}])[0].get("code") != "not-performed" or any(key.startswith("value") for key in resource):
                errors.append(f"{key[0]}/{key[1]}: unsupported method must have no result")
        if not (_observable_statuses(resource) & allowed_statuses):
            errors.append(f"{key[0]}/{key[1]}: lifecycle/status differs from latent event role")
        else:
            agreement_counts["status"] += 1
        patient_reference = resource.get("subject", {}).get("reference") or resource.get("patient", {}).get("reference")
        if not patient_reference or (key[0], patient_reference) not in patient_urls:
            errors.append(f"{key[0]}/{key[1]}: subject does not resolve to the source-local patient")
        else:
            agreement_counts["subject"] += 1
        capability = capabilities[key[0]]
        missing_fields = [field for field in capability.mandatory_fields if not capture_field_present(resource, field)]
        unexpected_optional = [field for field in ("note", "text", "supportingInfo")
                               if field in resource and field not in capability.optional_fields and field not in capability.mandatory_fields]
        if unexpected_optional:
            compatibility_errors["captureContract"].append(f"{key[0]}/{key[1]}: withheld optional fields emitted {unexpected_optional}")
        if missing_fields:
            compatibility_errors["captureContract"].append(
                f"{key[0]}/{key[1]}: missing mandatory fields {missing_fields}"
            )
        if mapping["disposition"] not in capability.enabled_dispositions:
            compatibility_errors["captureContract"].append(
                f"{key[0]}/{key[1]}: emitted disabled disposition {mapping['disposition']}"
            )
        if mapping["provenanceRequired"] and resource.get("meta", {}).get("source") != f"urn:hdp-test-bundle:source:{key[0]}":
            errors.append(f"{key[0]}/{key[1]}: shared capture lacks explicit source provenance")
        if resource.get("resourceType") == "Encounter" and (fact["system"], fact["code"]) not in set(codings(resource.get("reasonCode"))):
            compatibility_errors["encounterReason"].append(f"{key[0]}/{key[1]}: reason differs from encounter fact")
        for link in ([] if metadata_only else event.get("links", [])):
            lookup_count += 1
            targets = mapping_by_event_source.get((link["target_event_id"], key[0]), ())
            target_mapping = targets[0] if targets else None
            if not target_mapping:
                continue
            target_resource = resources.get((key[0], target_mapping["resourceFullUrl"]))
            if target_resource is None:
                # The mapping pass reports this missing target once; broken links
                # must not abort the remaining independent audit findings.
                continue
            if _link_field_supported(resource["resourceType"], target_resource["resourceType"]):
                agreement_denominators["keyClinicalLinks"] += 1
                if target_mapping["resourceFullUrl"] not in set(references(resource)):
                    compatibility_errors["observableLinks"].append(
                        f"{key[0]}/{key[1]}: {link['relationship']} target is not referenced"
                    )
                else:
                    agreement_counts["keyClinicalLinks"] += 1
                if event.get("indicationFactId") == target_mapping.get("factId"):
                    agreement_denominators["indication"] += 1
                    if target_mapping["resourceFullUrl"] in set(references(resource)):
                        agreement_counts["indication"] += 1
            if event["eventRole"] in {"result", "collection"} and target_resource["resourceType"] == "ServiceRequest":
                if target_mapping["resourceFullUrl"] not in set(references(resource)):
                    compatibility_errors["orderResult"].append(f"{key[0]}/{key[1]}: order/result link missing")
            if event["eventRole"] == "follow-up" and target_resource["resourceType"] == "Procedure":
                if target_mapping["resourceFullUrl"] not in set(references(resource)):
                    compatibility_errors["procedureFollowUp"].append(f"{key[0]}/{key[1]}: procedure/follow-up link missing")
    # Medication lifecycle facts with the same reviewed medication concept must
    # not restart after a stopped assertion unless an explicit supersedes link
    # makes the transition observable.
    by_pathway_medication: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in latent.values():
        fact = event["fact"]
        if fact["system"] == "http://www.nlm.nih.gov/research/umls/rxnorm":
            by_pathway_medication[(event["pathwayId"], fact["system"], fact["code"])].append(event)
    for key, events in by_pathway_medication.items():
        ordered = sorted(events, key=lambda row: row["occurrence"]["start"])
        stopped = False
        for event in ordered:
            status = event["fact"]["status"]
            if stopped and status == "active" and not any(link["relationship"] == "supersedes" for link in event.get("links", [])):
                compatibility_errors["medicationLifecycle"].append(
                    f"{key[0]}/{key[2]}: active event follows stop without supersedes"
                )
            stopped = status == "stopped"
    counts = Counter(row["disposition"] for row in mappings)
    capture_counts = Counter((row["source"], row["disposition"]) for rows in captures.values() for row in rows)
    branch_rows = []
    canonical = spec["populationSize"] >= 1000
    for source, capability in sorted(capabilities.items()):
        for disposition in sorted(set(capability.enabled_dispositions) | set(capability.expected_zero_dispositions)):
            count = capture_counts[(source, disposition)]
            expected_zero = disposition in capability.expected_zero_dispositions
            branch_rows.append({
                "source": source, "disposition": disposition, "count": count,
                "expectedCanonicalCountZero": expected_zero,
                "canonicalMinimum": 0 if expected_zero else 5,
            })
            if expected_zero and count:
                compatibility_errors["captureContract"].append(f"{source}/{disposition}: expected zero but observed {count}")
            if canonical and not expected_zero and count < 5:
                compatibility_errors["captureContract"].append(f"{source}/{disposition}: canonical count {count} below 5")
    compatibility_failure_count = sum(len(rows) for rows in compatibility_errors.values())
    errors.extend(
        message for category in compatibility_errors.values() for message in category
    )
    errors.extend(fixture_evidence["failures"])
    return {
        "schemaVersion": "4.1.0", "work": {"indexedMappings": len(mappings), "eventSourceLookups": lookup_count}, "latentEventCount": len(latent),
        "sourceDispositionCount": sum(len(rows) for rows in captures.values()),
        "fhirMappingCount": len(mappings), "ordinaryClinicalResourceCount": len(ordinary_urls),
        "supplementalCaptureResourceCount": len(supplemental),
        "supplementalCaptureGraphCount": len(supplemental_groups),
        "mappedOrdinaryClinicalResourceCount": len(ordinary_urls & (set(mapping_by_key) | set(supplemental_by_key))),
        "mappingCoverageRate": round(len(ordinary_urls & (set(mapping_by_key) | set(supplemental_by_key))) / max(1, len(ordinary_urls)), 6),
        "mappingCountsByDisposition": dict(sorted(counts.items())),
        "withheldClinicalContentMappingCount": counts["metadata-only"],
        "sourceCaptureDispositionCounts": {
            source: {
                disposition: capture_counts[(source, disposition)]
                for disposition in sorted(DISPOSITIONS)
            } for source in sorted(capabilities)
        },
        "captureBranchCoverage": branch_rows,
        "explicitCaptureCoverageFixtures": fixture_evidence,
        "factAgreement": {
            key: {"passed": agreement_counts[key], "denominator": len([
                row for row in mappings
                if declarations.get(row["cohortId"], {}).get("ordinaryPathwayEligible", True)
                and (key != "codeFamily" or row["disposition"] != "metadata-only")
            ])}
            for key in ("codeFamily", "status", "subject", "occurrenceInterval", "resourceType")
        } | {
            key: {"passed": agreement_counts[key], "denominator": agreement_denominators[key]}
            for key in ("indication", "keyClinicalLinks")
        },
        "compatibility": {
            key: {"failureCount": len(value), "failures": value[:100]}
            for key, value in compatibility_errors.items()
        },
        "compatibilityFailureCount": compatibility_failure_count,
        "failureCount": len(errors), "failures": errors,
        "status": "pass" if not errors else "fail",
    }
