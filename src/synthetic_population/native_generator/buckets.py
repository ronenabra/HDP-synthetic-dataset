from __future__ import annotations

from .catalogs import load_catalog

CATALOG_PATH = "tools/synthetic_population/data/bucket-semantics.v2.0.1.json"
_CATALOG = load_catalog(CATALOG_PATH)
ROLE_TO_BUCKET = _CATALOG["roles"]
COVERAGE_RESOURCE_ROLES = _CATALOG["coverageResourceRoles"]
PROFILE_OVERRIDES = _CATALOG["profileOverrides"]
BUCKET_REQUIREMENT_IDS = _CATALOG["bucketRequirementIds"]

if set(ROLE_TO_BUCKET.values()) != set(BUCKET_REQUIREMENT_IDS):
    raise ValueError("bucket semantic catalog must link every active bucket to one stable requirement ID")


def bucket_for_role(role: str) -> str:
    try:
        return ROLE_TO_BUCKET[role]
    except KeyError as exc:
        raise ValueError(f"unknown or ambiguous semantic bucket role: {role}") from exc


def coverage_role(resource_type: str, profile: str) -> str:
    if profile in PROFILE_OVERRIDES:
        return PROFILE_OVERRIDES[profile]
    try:
        return COVERAGE_RESOURCE_ROLES[resource_type]
    except KeyError as exc:
        raise ValueError(f"coverage resource has no reviewed semantic bucket role: {resource_type}|{profile}") from exc


def observable_role(resource: dict, source: str = "hmo") -> str:
    from .profiles import expected_profile
    profile = expected_profile(resource, source)
    if resource['resourceType'] == 'Observation':
        codes = {row.get('code') for row in resource.get('code', {}).get('coding', [])}
        if '24606-6' in codes:
            return 'diagnostic-imaging'
        if profile and profile.endswith('/il-hdp-observation'):
            return 'assessment'
        return coverage_role('Observation', profile or '')
    return coverage_role(resource['resourceType'], profile or '')
