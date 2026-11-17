from __future__ import annotations

from datetime import date
from typing import Any


def _default_age_applicability(code: str) -> dict[str, Any]:
    return {
        "id": "universal",
        "minimumAgeYears": 0,
        "maximumAgeYears": None,
        "referenceRangeHandling": "required",
        "interpretationHandling": "required",
        "ordinaryDistributionGuardApplicable": True,
        # Preserve the pre-policy BMI invariant for older content catalogs;
        # explicit age policy rows may narrow it to under-20 observations.
        "requiresDerivedPair": code == "39156-5",
        "requiresPositiveFiniteValue": True,
        "policyVersion": "default-1.0.0",
        "sourceEvidence": [],
        "reviewStatus": "inherited-from-model",
        "ageResolutionRequired": False,
    }


def unresolved_age_applicability(code: str | None = None) -> dict[str, Any]:
    """Fail-closed policy used when birth date or occurrence time is absent."""
    return {
        "id": "unknown-age",
        "minimumAgeYears": 0,
        "maximumAgeYears": None,
        "referenceRangeHandling": "required",
        "interpretationHandling": "required",
        "ordinaryDistributionGuardApplicable": True,
        # An unresolved BMI age must still retain the historical pair check;
        # failing to resolve age cannot silently turn off derived-value safety.
        "requiresDerivedPair": code == "39156-5",
        "requiresPositiveFiniteValue": True,
        "policyVersion": "unknown-age-fail-closed-1.0.0",
        "sourceEvidence": [],
        "reviewStatus": "fail-closed",
        "ageResolutionRequired": True,
    }


def birth_date_or_none(patient: dict[str, Any]) -> date | None:
    """Parse a Patient birthDate without turning missing data into an audit crash."""
    value = patient.get("birthDate")
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def completed_age_years(birth_date: date, at_date: date) -> int:
    """Return completed calendar years so the 20th birthday is unambiguous."""
    return at_date.year - birth_date.year - ((at_date.month, at_date.day) < (birth_date.month, birth_date.day))


def age_applicability(content: dict[str, Any], code: str, age_years: float) -> dict[str, Any]:
    """Resolve the one explicit age rule for a quantitative code.

    A missing override means the model-wide quantitative contract applies. An
    override must select exactly one half-open age interval, so a boundary
    change cannot silently make two policies apply or leave a gap.
    """
    value_model = content.get("modelDefinitions", {}).get("quantitative-observation", {}).get("valueModel", {})
    branch = next((row for row in content.get("branchQuantitativeModels", []) if row["code"] == code), None)
    if branch:
        return {**_default_age_applicability(code), "id": "branch-generation-only",
                "referenceRangeHandling": branch["referenceRangeHandling"],
                "interpretationHandling": branch["interpretationHandling"],
                "ordinaryDistributionGuardApplicable": branch["ordinaryDistributionGuardApplicable"],
                "requiresPositiveFiniteValue": False,
                "policyVersion": content["catalogVersion"], "reviewStatus": branch["reviewStatus"]}
    policy = value_model.get("ageApplicability", {}).get(code)
    if policy is None:
        result = _default_age_applicability(code)
        if code in {"29463-7", "8302-2"}:
            # Growth anchors are synthetic distributions, not clinical ranges.
            result.update(id="event-age-anthropometry", referenceRangeHandling="omit",
                          interpretationHandling="omit", ordinaryDistributionGuardApplicable=False,
                          policyVersion="clinical-state-1.0.0", ageResolutionRequired=False,
                          reviewStatus="engineering-contract-pending-independent-clinical-review")
        return result
    matches = [
        rule for rule in policy["rules"]
        if age_years >= rule["minimumAgeYears"]
        and (rule["maximumAgeYears"] is None or age_years < rule["maximumAgeYears"])
    ]
    if len(matches) != 1:
        raise ValueError(f"age applicability for {code} has {len(matches)} matching rules at age {age_years}")
    return {
        **matches[0],
        "policyVersion": policy["policyVersion"],
        "sourceEvidence": policy["sourceEvidence"],
        "reviewStatus": policy["reviewStatus"],
        "ageResolutionRequired": False,
    }


def validate_branch_quantitative_models(content: dict[str, Any]) -> None:
    """Validate the isolated synthetic branch inventory, without clinical bands."""
    from math import isfinite
    fields = {"system", "code", "display", "unit", "generationMinimum", "generationMaximum",
              "validMinimum", "validMaximum", "allowedStatuses", "requiresEventTime",
              "referenceRangeHandling", "interpretationHandling", "ordinaryDistributionGuardApplicable",
              "provenance", "reviewStatus"}
    seen = {row["code"] for row in content["analytes"]}
    for row in content.get("branchQuantitativeModels", []):
        bounds = [row.get(k) for k in ("validMinimum", "generationMinimum", "generationMaximum", "validMaximum")]
        if (set(row) != fields or row["code"] in seen
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and isfinite(x) for x in bounds)
                or bounds != sorted(bounds) or bounds[0] >= bounds[-1]
                or row["system"] != "http://loinc.org" or not row["code"] or not row["unit"]
                or row["allowedStatuses"] != ["final"] or row["requiresEventTime"] is not True
                or row["referenceRangeHandling"] != "omit" or row["interpretationHandling"] != "omit"
                or row["ordinaryDistributionGuardApplicable"] is not False
                or not row["provenance"] or not row["reviewStatus"]):
            raise ValueError("invalid isolated branch quantitative model")
        seen.add(row["code"])


def branch_quantitative_findings(resource: dict[str, Any], content: dict[str, Any]) -> list[str]:
    """Enforce branch safety even without age/reference-band applicability."""
    from math import isfinite
    from .temporal import occurrence_from_resource
    if resource.get("resourceType") != "Observation":
        return []
    findings = []
    observed = {(row.get("system"), row.get("code")) for row in resource.get("code", {}).get("coding", [])}
    for model in content.get("branchQuantitativeModels", []):
        selector = (model["system"], model["code"])
        if any(selector in {(row.get("system"), row.get("code")) for row in component.get("code", {}).get("coding", [])}
               for component in resource.get("component", [])):
            findings.append(f"{model['code']}: unsupported branch quantitative component")
        if selector not in observed:
            continue
        quantity = resource.get("valueQuantity", {})
        value = quantity.get("value")
        if (not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value)
                or not model["validMinimum"] <= value <= model["validMaximum"]):
            findings.append(f"{model['code']}: invalid branch quantitative value")
        if quantity.get("system") != "http://unitsofmeasure.org" or quantity.get("code") != model["unit"] or quantity.get("unit") != model["unit"]:
            findings.append(f"{model['code']}: incompatible branch quantitative unit")
        if resource.get("status") not in model["allowedStatuses"]:
            findings.append(f"{model['code']}: unreviewed branch quantitative status")
        if occurrence_from_resource(resource) is None:
            findings.append(f"{model['code']}: missing branch quantitative event time")
        if resource.get("referenceRange") or resource.get("interpretation"):
            findings.append(f"{model['code']}: unsupported branch clinical reference metadata")
    return findings


def population_branch_quantitative_findings(bundles: dict, content: dict) -> list[dict]:
    return [{"source": source, "resource": entry["fullUrl"], "reason": reason}
            for source, bundle in bundles.items() for entry in bundle.get("entry", [])
            for reason in branch_quantitative_findings(entry["resource"], content)]
