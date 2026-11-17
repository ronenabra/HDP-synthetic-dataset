from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from .care_pathway import ClinicalEvent, PatientPathway
from .catalogs import load_catalog
from .temporal import ClinicalInterval


DISPOSITIONS = {
    "performed", "authored", "received", "summarized", "referral-only",
    "metadata-only", "absent",
}
EVENT_ROLES = {
    "encounter", "emergency-encounter", "inpatient-encounter", "follow-up",
    "condition", "assessment", "order", "referral", "collection", "result",
    "growth-measurement", "prescription", "dispense", "medication-review",
    "medication-administration", "procedure", "oncology-treatment",
    "immunization", "discharge",
}


@dataclass(frozen=True)
class SourceCapability:
    source: str
    care_settings: tuple[str, ...]
    performed_roles: tuple[str, ...]
    authored_roles: tuple[str, ...]
    received_roles: tuple[str, ...]
    summary_roles: tuple[str, ...]
    referral_only_roles: tuple[str, ...]
    metadata_only_roles: tuple[str, ...]
    mandatory_fields: tuple[str, ...]
    optional_fields: tuple[str, ...]
    reference_behavior: str
    provenance_behavior: str
    enabled_dispositions: tuple[str, ...]
    expected_zero_dispositions: tuple[str, ...]


@dataclass(frozen=True)
class CaptureDisposition:
    latent_event_id: str
    source: str
    disposition: str
    documentation_lag_minutes: int
    provenance_required: bool
    reason: str
    origin_source: str | None = None


def _slot(seed: int, event_id: str, source: str, modulus: int) -> int:
    digest = hashlib.sha256(f"{seed}|capture|{event_id}|{source}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def load_capabilities(path: str) -> dict[str, SourceCapability]:
    catalog = load_catalog(path)
    if set(catalog) != {"catalogVersion", "status", "sources", "coverageFixtures"}:
        raise ValueError("capture policy catalog has unknown or missing root fields")
    if set(catalog["sources"]) != {"hmo", "hospital-a", "hospital-b"}:
        raise ValueError("capture policy must declare exactly the three active sources")
    result = {}
    for source, row in catalog["sources"].items():
        expected = {
            "careSettings", "performedRoles", "receivedRoles", "summaryRoles",
            "authoredRoles", "referralOnlyRoles", "metadataOnlyRoles",
            "mandatoryFields", "optionalFields", "referenceBehavior",
            "provenanceBehavior", "enabledDispositions",
            "expectedCanonicalZeroDispositions", "documentationLagMinutes",
            "omissionPercent",
        }
        if set(row) != expected:
            raise ValueError(
                f"capture policy fields mismatch for {source}: "
                f"unknown={sorted(set(row) - expected)}, missing={sorted(expected - set(row))}"
            )
        role_fields = (
            "performedRoles", "authoredRoles", "receivedRoles", "summaryRoles",
            "referralOnlyRoles", "metadataOnlyRoles",
        )
        if any(len(row[field]) != len(set(row[field])) for field in role_fields):
            raise ValueError(f"duplicate capture role in policy for {source}")
        if any(not set(row[field]) <= EVENT_ROLES for field in role_fields):
            raise ValueError(f"unsupported capture role in policy for {source}")
        if not row["careSettings"] or len(row["careSettings"]) != len(set(row["careSettings"])):
            raise ValueError(f"invalid care settings in policy for {source}")
        if len(row["mandatoryFields"]) != len(set(row["mandatoryFields"])) or len(row["optionalFields"]) != len(set(row["optionalFields"])):
            raise ValueError(f"duplicate capture fields in policy for {source}")
        if set(row["mandatoryFields"]) & set(row["optionalFields"]):
            raise ValueError(f"mandatory and optional capture fields overlap for {source}")
        if not set(row["authoredRoles"]).issubset(row["performedRoles"]):
            raise ValueError(f"authoredRoles must be a subset of performedRoles for {source}")
        priority_role_sets = (
            set(row["referralOnlyRoles"]), set(row["metadataOnlyRoles"]),
            set(row["receivedRoles"]), set(row["summaryRoles"]),
        )
        if any(left & right for position, left in enumerate(priority_role_sets) for right in priority_role_sets[position + 1:]):
            raise ValueError(f"capture role appears in multiple non-owner disposition rules for {source}")
        enabled = set(row["enabledDispositions"])
        expected_zero = set(row["expectedCanonicalZeroDispositions"])
        if not enabled <= DISPOSITIONS or not expected_zero <= DISPOSITIONS or enabled & expected_zero:
            raise ValueError(f"invalid disposition declaration for {source}")
        if enabled | expected_zero != DISPOSITIONS:
            raise ValueError(f"every capture disposition must be enabled or declared zero for {source}")
        if row["referenceBehavior"] != "source-local-references-only":
            raise ValueError(f"unsupported reference behavior for {source}")
        if row["provenanceBehavior"] != "received-and-summarized-require-explicit-source":
            raise ValueError(f"unsupported provenance behavior for {source}")
        if not 0 <= row["omissionPercent"] <= 100 or row["documentationLagMinutes"] < 0:
            raise ValueError(f"invalid omission/lag policy for {source}")
        result[source] = SourceCapability(
            source, tuple(row["careSettings"]), tuple(row["performedRoles"]),
            tuple(row["authoredRoles"]), tuple(row["receivedRoles"]),
            tuple(row["summaryRoles"]), tuple(row["referralOnlyRoles"]),
            tuple(row["metadataOnlyRoles"]), tuple(row["mandatoryFields"]),
            tuple(row["optionalFields"]), row["referenceBehavior"],
            row["provenanceBehavior"], tuple(row["enabledDispositions"]),
            tuple(row["expectedCanonicalZeroDispositions"]),
        )
    fixture_keys = []
    for fixture in catalog["coverageFixtures"]:
        if set(fixture) != {"source", "disposition", "eventRole", "careSetting"}:
            raise ValueError("capture coverage fixture has unknown or missing fields")
        if fixture["source"] not in result or fixture["eventRole"] not in EVENT_ROLES:
            raise ValueError("capture coverage fixture names an unsupported source or event role")
        capability = result[fixture["source"]]
        if fixture["disposition"] not in capability.enabled_dispositions:
            raise ValueError("capture coverage fixture targets a disabled disposition")
        fixture_keys.append((fixture["source"], fixture["disposition"]))
    expected_fixture_keys = {
        (source, disposition)
        for source, capability in result.items()
        for disposition in capability.enabled_dispositions
    }
    if len(fixture_keys) != len(set(fixture_keys)) or set(fixture_keys) != expected_fixture_keys:
        raise ValueError("capture coverage fixtures must cover every enabled source/disposition exactly once")
    return result


def _capture_disposition(
    event: ClinicalEvent, owner: str, source: str, capability: SourceCapability,
    policy: dict[str, Any], seed: int,
) -> tuple[str, str]:
    if source == owner:
        disposition = "authored" if event.role in capability.authored_roles else "performed"
        reason = "routed authoring source" if disposition == "authored" else "routed performing source"
    elif event.role in capability.referral_only_roles:
        disposition, reason = "referral-only", "source retains a referral-only record"
    elif event.role in capability.metadata_only_roles:
        disposition, reason = "metadata-only", "source retains metadata without clinical content"
    elif event.role in capability.received_roles:
        disposition, reason = "received", "source receives this event role"
    elif event.role in capability.summary_roles:
        disposition, reason = "summarized", "source retains summary of this event role"
    else:
        disposition, reason = "absent", "source lacks a legal capture role"
    if source != owner and _slot(seed, event.id, source, 100) < policy["omissionPercent"]:
        disposition, reason = "absent", "deterministic optional omission branch"
    if source != owner and disposition not in event.permitted_dispositions:
        disposition, reason = "absent", "clinical event has no authorized external handoff"
    return disposition, reason


def route_event(event: ClinicalEvent, capabilities: dict[str, SourceCapability], routing_key: str | None = None) -> str:
    candidates = [
        source for source, capability in sorted(capabilities.items())
        if event.care_setting in capability.care_settings
        and event.role in capability.performed_roles
    ]
    if not candidates:
        raise ValueError(f"no source can perform latent event {event.id}/{event.role}/{event.care_setting}")
    # The event ID is derived from truth and the route uses capability only; it
    # cannot change the diagnosis/fact chosen by the latent template.
    return candidates[int(hashlib.sha256((routing_key or event.id).encode()).hexdigest(), 16) % len(candidates)]


def project_capture(
    pathway: PatientPathway, policy_path: str, seed: int, *, locality_code: str = "", sensitivity_catalog: str | None = None,
) -> tuple[dict[str, list[CaptureDisposition]], dict[str, str]]:
    catalog = load_catalog(policy_path)
    capabilities = load_capabilities(policy_path)
    # Related events share an episode route when their capabilities permit it.
    # Setting changes are explicit handoffs; clinical facts are never rewritten.
    routes = {event.id: route_event(event, capabilities, f"{seed}|{pathway.cohort_id}|{event.episode_id}|{event.care_setting}|{locality_code}")
              for event in pathway.events}
    from .sensitivity import owning_source
    for event in pathway.events:
        if event.sensitivity_fact:
            # Policy-defined ownership is itself an explicit required event.
            # It does not authorize sensitive capture in the other sources.
            if sensitivity_catalog is None:
                raise ValueError("sensitivity capture requires its explicit catalog")
            spec_stub = {"sensitivity": {"catalog": sensitivity_catalog}}
            routes[event.id] = owning_source(spec_stub, event.sensitivity_fact)
    care_sources = set(routes.values())
    result: dict[str, list[CaptureDisposition]] = {source: [] for source in capabilities}
    for event in pathway.events:
        owner = routes[event.id]
        for source, capability in sorted(capabilities.items()):
            policy = catalog["sources"][source]
            disposition, reason = _capture_disposition(
                event, owner, source, capability, policy, seed,
            )
            if (pathway.variant == "emergency-appendectomy"
                    and (pathway.secondary_branch == "short-stay"
                         or pathway.secondary_branch == "open" and disposition == "absent")
                    and event.role == "procedure" and source == "hmo" and owner != source):
                # The HMO-authored community review must have its source-local
                # received surgical event, even when optional import omission
                # would otherwise hide that event.
                disposition, reason = "received", "surgical event received for community review"
            if event.sensitivity_fact and source != owner:
                disposition, reason = "absent", "sensitive fact is restricted to its policy owner"
            if source != "hmo" and source not in care_sources and source != owner:
                disposition, reason = "absent", "no performed-care or referral relationship with this hospital"
            if disposition not in capability.enabled_dispositions:
                raise ValueError(f"policy produced disabled disposition {source}/{event.id}/{disposition}")
            if disposition not in event.permitted_dispositions or disposition not in DISPOSITIONS:
                raise ValueError(f"illegal capture disposition {source}/{event.id}/{disposition}")
            result[source].append(CaptureDisposition(
                event.id, source, disposition, policy["documentationLagMinutes"],
                event.provenance_required or disposition in {
                    "received", "summarized", "referral-only", "metadata-only",
                }, reason, owner,
            ))
    for event in pathway.events:
        rows = [row for values in result.values() for row in values if row.latent_event_id == event.id]
        if len(rows) != 3:
            raise ValueError(f"latent event {event.id} does not have exactly three source dispositions")
        if not any(row.disposition != "absent" for row in rows):
            raise ValueError(f"observable latent event {event.id} has no capture")
    return result, routes


def capture_coverage_fixtures(policy_path: str, seed: int) -> dict[str, Any]:
    """Exercise every enabled disposition through the production rule engine."""
    catalog = load_catalog(policy_path)
    capabilities = load_capabilities(policy_path)
    instant = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    failures = []
    fixture_by_key = {
        (row["source"], row["disposition"]): row for row in catalog["coverageFixtures"]
    }
    for source, capability in sorted(capabilities.items()):
        for target in sorted(DISPOSITIONS):
            if target in capability.expected_zero_dispositions:
                rows.append({
                    "source": source, "disposition": target, "fixtureId": None,
                    "expectedCanonicalCountZero": True, "status": "not-applicable-declared-zero",
                })
                continue
            fixture = fixture_by_key[(source, target)]
            matched = None
            for ordinal in range(1, 501):
                role, setting = fixture["eventRole"], fixture["careSetting"]
                event = ClinicalEvent(
                    f"capture-fixture-{source}-{target}-{ordinal}-{role}-{setting}",
                    "capture-fixture-episode", "capture-fixture-fact", role,
                    "capture-policy-fixture", setting, role, "completed",
                    ClinicalInterval(instant, instant, "instant"),
                )
                try:
                    owner = route_event(event, capabilities)
                except ValueError:
                    continue
                actual, reason = _capture_disposition(
                    event, owner, source, capability, catalog["sources"][source], seed,
                )
                if actual == target:
                    matched = {
                        "source": source, "disposition": target,
                        "fixtureId": event.id, "eventRole": role,
                        "careSetting": setting, "routedSource": owner,
                        "reason": reason, "expectedCanonicalCountZero": False,
                        "status": "pass",
                    }
                    break
                if matched:
                    break
            if matched is None:
                failures.append(f"{source}/{target}: no explicit coverage fixture exercises enabled branch")
                rows.append({
                    "source": source, "disposition": target, "fixtureId": None,
                    "expectedCanonicalCountZero": False, "status": "fail",
                })
            else:
                rows.append(matched)
    return {
        "schemaVersion": "1.0.0", "fixtureCount": sum(row["fixtureId"] is not None for row in rows),
        "rows": rows, "failureCount": len(failures), "failures": failures,
        "status": "pass" if not failures else "fail",
    }


def capture_ledger_rows(
    pathway: PatientPathway, captures: dict[str, list[CaptureDisposition]], routes: dict[str, str],
) -> list[dict[str, Any]]:
    facts = {row.id: row for row in pathway.facts}
    rows = []
    for event in pathway.events:
        fact = facts[event.fact_id]
        rows.append({
            "rowType": "latent-event", "pathwayId": pathway.id,
            "cohortId": pathway.cohort_id, "episodeId": event.episode_id,
            "eventId": event.id, "factId": event.fact_id,
            "fact": asdict(fact), "eventRole": event.role,
            "clinicalDomain": event.clinical_domain, "careSetting": event.care_setting,
            "subjectId": event.subject_id, "participantRoles": list(event.participant_roles),
            "indicationFactId": event.indication_fact_id,
            "permittedDispositions": list(event.permitted_dispositions),
            "provenanceRequired": event.provenance_required,
            "sensitivityFact": event.sensitivity_fact,
            "secondaryTrait": pathway.secondary_trait,
            "routedSource": routes[event.id],
            "handoffs": [{"fromSource": routes[link.target_event_id], "toSource": routes[event.id],
                          "fromEventId": link.target_event_id, "toEventId": event.id}
                         for link in event.links if routes.get(link.target_event_id) != routes[event.id]],
            "occurrence": {
                "start": event.occurrence.start.isoformat().replace("+00:00", "Z"),
                "end": event.occurrence.end.isoformat().replace("+00:00", "Z"),
                "precision": event.occurrence.precision,
            },
            "links": [asdict(link) for link in event.links],
        })
        rows.extend({"rowType": "source-capture", **asdict(capture)} for capture in (
            next(row for row in captures[source] if row.latent_event_id == event.id)
            for source in sorted(captures)
        ))
    return rows


def observable_origin(resource: dict[str, Any], capturing_source: str) -> str:
    """Read performing/authoring origin from clinical actor fields."""
    candidates = [resource.get(key, {}) for key in ('serviceProvider', 'recorder', 'requester', 'informationSource', 'authenticator')]
    candidates.extend(resource.get('author', []))
    candidates.extend(row.get('actor', row) for row in resource.get('performer', []))
    candidates.extend(row.get('individual', {}) for row in resource.get('participant', []))
    for actor in candidates:
        system = actor.get('identifier', {}).get('system', '')
        for source in ('hmo', 'hospital-a', 'hospital-b'):
            if system.startswith(f'urn:hdp-test-bundle:synthetic:{source}:'):
                return source
    return capturing_source


def capture_field_present(resource: dict[str, Any], field: str) -> bool:
    if field == 'subject-or-patient':
        return any(resource.get(key, {}).get('reference') or resource.get(key, {}).get('identifier')
                   for key in ('subject', 'patient'))
    value: Any = resource
    for part in field.split('.'):
        if not isinstance(value, dict):
            return False
        value = value.get(part)
    return value not in (None, '', [], {})


def metadata_only_record(resource: dict[str, Any]) -> bool:
    return resource.get('resourceType') == 'DocumentReference' and any(
        row.get('attachment', {}).get('url', '').startswith('urn:hdp-test-bundle:withheld-record:')
        for row in resource.get('content', [])
    )


def metadata_content_leaks(resource: dict[str, Any]) -> bool:
    return (resource.get('resourceType') != 'DocumentReference'
            or any(field in resource for field in ('valueQuantity', 'valueString', 'component', 'code', 'medicationCodeableConcept'))
            or any(row.get('attachment', {}).get('data') for row in resource.get('content', [])))
