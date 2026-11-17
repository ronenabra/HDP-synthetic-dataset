from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable


@dataclass(frozen=True, order=True)
class ClinicalInterval:
    """Closed clinical-time interval with explicit source precision."""

    start: datetime
    end: datetime
    precision: str

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("clinical interval ends before it starts")
        if self.precision not in {"instant", "day", "period"}:
            raise ValueError(f"unsupported clinical interval precision: {self.precision}")


@dataclass(frozen=True)
class ClinicalOccurrence:
    resource_type: str
    event_role: str
    interval: ClinicalInterval
    resource_id: str | None = None
    latent_event_id: str | None = None


def parse_fhir_time(value: str) -> ClinicalInterval:
    """Parse a FHIR date/dateTime as a conservative closed UTC interval."""
    if not isinstance(value, str) or len(value) < 10:
        raise ValueError("FHIR time must contain at least a calendar date")
    if len(value) == 10:
        day = date.fromisoformat(value)
        start = datetime.combine(day, time.min, timezone.utc)
        end = (
            datetime.max.replace(tzinfo=timezone.utc)
            if day == date.max
            else start + timedelta(days=1) - timedelta(microseconds=1)
        )
        return ClinicalInterval(start, end, "day")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    instant = datetime.fromisoformat(normalized)
    if instant.tzinfo is None:
        raise ValueError("FHIR dateTime used for clinical ordering must carry a timezone")
    instant = instant.astimezone(timezone.utc)
    return ClinicalInterval(instant, instant, "instant")


def period_interval(value: dict[str, Any]) -> ClinicalInterval:
    if not isinstance(value, dict) or not value.get("start"):
        raise ValueError("clinical period requires start")
    start = parse_fhir_time(value["start"])
    end = parse_fhir_time(value.get("end", value["start"]))
    return ClinicalInterval(start.start, end.end, "period")


def strictly_before(left: ClinicalInterval, right: ClinicalInterval) -> bool:
    return left.end < right.start


def strictly_after(left: ClinicalInterval, right: ClinicalInterval) -> bool:
    return strictly_before(right, left)


def starts_before(left: ClinicalInterval, right: ClinicalInterval) -> bool:
    return left.end < right.start if left.precision == "day" or right.precision == "day" else left.start < right.start


def ends_after(left: ClinicalInterval, right: ClinicalInterval) -> bool:
    return left.start > right.end if left.precision == "day" or right.precision == "day" else left.end > right.end


def contains(outer: ClinicalInterval, inner: ClinicalInterval) -> bool:
    return outer.start <= inner.start and outer.end >= inner.end


def overlaps(left: ClinicalInterval, right: ClinicalInterval) -> bool:
    return left.start <= right.end and right.start <= left.end


def active_during(exposure: ClinicalInterval, event: ClinicalInterval) -> bool:
    return overlaps(exposure, event)


def within_window(
    earlier: ClinicalInterval,
    later: ClinicalInterval,
    minimum: timedelta = timedelta(0),
    maximum: timedelta | None = None,
) -> bool:
    """Require every possible interpretation to satisfy the inclusive window."""
    minimum_gap = later.start - earlier.end
    maximum_gap = later.end - earlier.start
    return minimum_gap >= minimum and (maximum is None or maximum_gap <= maximum)


def documentation_after_occurrence(occurrence: ClinicalInterval, documented: ClinicalInterval) -> bool:
    return occurrence.end < documented.start


def monotonic_trend(values: list[float], direction: str, tolerance: float = 0.0) -> bool:
    if len(values) < 2:
        return False
    if direction == "increasing":
        return all(right + tolerance >= left for left, right in zip(values, values[1:])) and values[-1] > values[0] + tolerance
    if direction == "decreasing":
        return all(right - tolerance <= left for left, right in zip(values, values[1:])) and values[-1] < values[0] - tolerance
    if direction == "stable":
        return max(values) - min(values) <= tolerance
    raise ValueError(f"unsupported trend direction: {direction}")


def _event_role(resource: dict[str, Any]) -> str:
    resource_type = resource.get("resourceType", "")
    if resource_type == "Encounter":
        return "attended-care"
    if resource_type == "Observation":
        return "measurement"
    if resource_type == "Procedure":
        return "performed-procedure"
    if resource_type == "ServiceRequest":
        return "order"
    if resource_type == "MedicationRequest":
        return "prescribing"
    if resource_type == "MedicationDispense":
        return "dispensing"
    if resource_type == "MedicationAdministration":
        return "administration"
    if resource_type == "MedicationStatement":
        return "exposure"
    if resource_type == "Immunization":
        return "administration"
    if resource_type == "Specimen":
        return "collection"
    if resource_type == "DiagnosticReport":
        return "diagnostic-event"
    if resource_type == "CarePlan":
        return "care-plan"
    return ""


def occurrence_from_resource(
    resource: dict[str, Any], *, resource_id: str | None = None,
    latent_event_id: str | None = None,
) -> ClinicalOccurrence | None:
    """Return the single primary clinical occurrence declared by the R40 registry."""
    resource_type = resource.get("resourceType")
    value: Any = None
    if resource_type == "Encounter":
        value = resource.get("period")
    elif resource_type == "Observation":
        value = resource.get("effectiveDateTime") or resource.get("effectivePeriod")
    elif resource_type == "Condition":
        value = resource.get("onsetDateTime") or resource.get("onsetPeriod") or resource.get("recordedDate")
    elif resource_type == "AllergyIntolerance":
        value = resource.get("onsetDateTime") or resource.get("onsetPeriod") or resource.get("recordedDate")
    elif resource_type == "Procedure":
        value = resource.get("performedDateTime") or resource.get("performedPeriod")
    elif resource_type == "ServiceRequest":
        value = resource.get("authoredOn")
    elif resource_type == "MedicationRequest":
        value = resource.get("authoredOn")
    elif resource_type == "MedicationDispense":
        value = resource.get("whenHandedOver")
    elif resource_type == "MedicationAdministration":
        value = resource.get("effectiveDateTime") or resource.get("effectivePeriod")
    elif resource_type == "MedicationStatement":
        # A MedicationStatement is clinical time only when it represents exposure.
        if resource.get("status") not in {"active", "completed", "stopped"}:
            return None
        value = resource.get("effectiveDateTime") or resource.get("effectivePeriod")
    elif resource_type == "Immunization":
        value = resource.get("occurrenceDateTime") or resource.get("occurrenceString")
    elif resource_type == "Specimen":
        value = resource.get("collection", {}).get("collectedDateTime") or resource.get("collection", {}).get("collectedPeriod")
    elif resource_type == "DiagnosticReport":
        value = resource.get("effectiveDateTime") or resource.get("effectivePeriod")
    elif resource_type == "CarePlan":
        value = resource.get("period")
    elif resource_type == "DocumentReference":
        value = resource.get("date") or resource.get("context", {}).get("period")
    if value is None or (resource_type == "Immunization" and "occurrenceString" in resource):
        return None
    interval = period_interval(value) if isinstance(value, dict) else parse_fhir_time(value)
    return ClinicalOccurrence(resource_type, _event_role(resource), interval, resource_id, latent_event_id)


def clinical_occurrences(resources: Iterable[dict[str, Any]]) -> list[ClinicalOccurrence]:
    result = []
    seen_event_keys = set()
    for resource in resources:
        if resource.get("resourceType") in {"Condition", "AllergyIntolerance"}:
            continue
        event_key = next((
            row.get("value") for row in resource.get("identifier", [])
            if row.get("system") == "urn:hdp-test-bundle:clinical-event-key"
        ), None)
        if event_key and event_key in seen_event_keys:
            continue
        occurrence = occurrence_from_resource(
            resource, resource_id=resource.get("_fullUrl") or resource.get("id"),
            latent_event_id=event_key,
        )
        if occurrence is not None:
            result.append(occurrence)
            if event_key:
                seen_event_keys.add(event_key)
    return result


def history_decision(occurrences: Iterable[ClinicalOccurrence], minimum_days: int) -> dict[str, Any]:
    rows = list(occurrences)
    distinct = sorted({row.interval.start.date() for row in rows})
    earliest = min((row.interval.start for row in rows), default=None)
    latest = max((row.interval.end for row in rows), default=None)
    span = (latest.date() - earliest.date()).days if earliest and latest else 0
    passed = len(rows) >= 3 and len(distinct) >= 3 and span >= minimum_days
    return {
        "eventCount": len(rows),
        "distinctDateCount": len(distinct),
        "earliest": earliest.isoformat().replace("+00:00", "Z") if earliest else None,
        "latest": latest.isoformat().replace("+00:00", "Z") if latest else None,
        "spanDays": span,
        "minimumDays": minimum_days,
        "passed": passed,
    }
