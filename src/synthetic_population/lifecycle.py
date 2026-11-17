"""Synthetic demographic lifecycle assignment and native Bundle audit."""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
import hashlib
from math import ceil, floor

from .catalogs import load_catalog
from .observable import patient_graphs
from .problems import allocation
from .temporal import occurrence_from_resource


def policy_for(spec: dict) -> dict | None:
    path = spec.get("demographics", {}).get("lifecyclePolicyCatalog")
    return load_catalog(path) if path else None


def age_years(born: date, as_of: date) -> int:
    return as_of.year - born.year - ((as_of.month, as_of.day) < (born.month, born.day))


def clinical_occurrence(resource: dict):
    """Clinical event time; recording and document filing are administrative."""
    kind = resource.get("resourceType")
    if kind in {"Condition", "AllergyIntolerance"} and not (resource.get("onsetDateTime") or resource.get("onsetPeriod")):
        return None
    if kind == "DocumentReference" and not resource.get("context", {}).get("period"):
        return None
    if kind == "DocumentReference":
        return occurrence_from_resource({**resource, "date": None})
    return occurrence_from_resource(resource)


def assign_deaths(spec: dict, per_org: dict, crosswalk: list[dict], clinical_by_cohort: dict) -> None:
    policy = policy_for(spec)
    if policy is None:
        return
    cutoff = date.fromisoformat(policy["latestDeathDate"])
    earliest = date.fromisoformat(policy["earliestDeathDate"])
    patient_by_url = {
        (source, url): resource
        for source, rows in per_org.items() for url, resource in rows
        if resource.get("resourceType") == "Patient"
    }
    block_size = 1000 if spec["populationSize"] >= 1000 else spec["populationSize"]
    for block_start in range(1, spec["populationSize"] + 1, block_size):
        block_end = min(spec["populationSize"], block_start + block_size - 1)
        target = round((block_end - block_start + 1) * policy["deceasedPerThousand"] / 1000)
        candidates = []
        for index in range(block_start, block_end + 1):
            row = crosswalk[index - 1]
            if row["evaluationClass"] != "ordinary":
                continue
            # Reserve the final two positions of every stable block. The
            # coverage fixture uses the final two people of a population, and
            # the per-block reservation preserves nested death assignment.
            if index > block_end - 2:
                continue
            patient = patient_by_url[("hmo", row["sources"]["hmo"]["patientFullUrl"])]
            born = date.fromisoformat(patient["birthDate"])
            events = [clinical_occurrence(resource) for rows in clinical_by_cohort[index].values() for resource in rows]
            if any(event and event.interval.end.date() > cutoff for event in events):
                continue
            rank = hashlib.sha256(f"{spec['seed']}|deceased|{index:06d}".encode()).hexdigest()
            latest_event = max((event.interval.end.date() for event in events if event), default=born)
            first_possible = max(earliest, latest_event)
            span = (cutoff - first_possible).days
            death = first_possible + timedelta(days=int(rank[16:24], 16) % (span + 1))
            # A Condition's completed course can end after its onset event.
            # Preserve already valid assignments, and constrain only candidates
            # whose proposed death would precede an explicitly saved abatement.
            state_ends = [date.fromisoformat(value[:10])
                          for rows in clinical_by_cohort[index].values() for resource in rows
                          if resource.get("resourceType") == "Condition"
                          for value in [resource.get("abatementDateTime") or resource.get("abatementPeriod", {}).get("end")]
                          if value]
            latest_state_end = max(state_ends, default=latest_event)
            if death < latest_state_end:
                if latest_state_end > cutoff:
                    continue
                first_possible = max(first_possible, latest_state_end)
                span = (cutoff - first_possible).days
                death = first_possible + timedelta(days=int(rank[16:24], 16) % (span + 1))
            if age_years(born, death) < policy["minimumDeceasedAgeYears"]:
                continue
            candidates.append((rank, index, death))
        if len(candidates) < target:
            raise ValueError(f"lifecycle target infeasible in block {block_start}: {len(candidates)} eligible for {target}")
        for _, index, death in sorted(candidates)[:target]:
            row = crosswalk[index - 1]
            for source, view in row["sources"].items():
                patient = patient_by_url[(source, view["patientFullUrl"])]
                patient["deceasedDateTime"] = death.isoformat()
                patient["meta"]["lastUpdated"] = spec["timelineAnchor"]


def build_lifecycle_audit(bundles: dict, spec: dict) -> dict:
    policy = policy_for(spec)
    if policy is None:
        return {"status": "not-applicable"}
    graphs, _ = patient_graphs(bundles, spec)
    anchor = date.fromisoformat(spec["timelineAnchor"][:10])
    earliest_death = date.fromisoformat(policy["earliestDeathDate"])
    latest_death = date.fromisoformat(policy["latestDeathDate"])
    system = spec["identity"]["cohortIdentifierSystem"]
    patient_views = {}
    for source, bundle in bundles.items():
        for entry in bundle["entry"]:
            resource = entry["resource"]
            if resource.get("resourceType") != "Patient":
                continue
            cohort = next((item.get("value") for item in resource.get("identifier", []) if item.get("system") == system), None)
            if cohort:
                patient_views.setdefault(cohort, {})[source] = resource
    failures = []
    age_counts = Counter()
    deceased = []
    living = []
    declared_noise = 0
    noise_examples = []
    event_count = 0
    for cohort, views in sorted(patient_views.items()):
        if set(views) != set(bundles):
            failures.append(f"{cohort}: missing source Patient view")
            continue
        facts = {(p.get("birthDate"), p.get("deceasedDateTime"), p.get("deceasedBoolean")) for p in views.values()}
        if len(facts) != 1:
            failures.append(f"{cohort}: source lifecycle truth disagrees")
        patient = views["hmo"]
        born = date.fromisoformat(patient["birthDate"])
        age = age_years(born, anchor)
        band = "under-18" if age < 18 else "18-39" if age < 40 else "40-64" if age < 65 else "65-74" if age < 75 else "75-plus"
        age_counts[band] += 1
        death = patient.get("deceasedDateTime")
        if any(p.get("deceasedBoolean") is not None for p in views.values()):
            failures.append(f"{cohort}: lifecycle policy requires a dated death projection")
        if death:
            deceased.append(cohort)
            if allocation(spec, int(cohort.rsplit("-", 1)[-1])) or any(
                tag.get("system") == "urn:hdp-test-bundle:synthetic-scenario-role"
                and tag.get("code") == "near-miss-control"
                for resource in graphs.get(cohort, [])
                for tag in resource.get("meta", {}).get("tag", [])
            ):
                failures.append(f"{cohort}: deceased cohort is not ordinary")
            if (not earliest_death <= date.fromisoformat(death[:10]) <= latest_death
                    or age_years(born, date.fromisoformat(death[:10])) < policy["minimumDeceasedAgeYears"]):
                failures.append(f"{cohort}: death date or age ineligible")
            if any(p.get("meta", {}).get("lastUpdated", "")[:10] < death[:10] for p in views.values()):
                failures.append(f"{cohort}: source Patient update predates recorded death")
        else:
            living.append(cohort)
        for resource in graphs.get(cohort, []):
            event = clinical_occurrence(resource)
            if event is None:
                continue
            event_count += 1
            if event.interval.start.date() < born or (death and event.interval.end.date() > date.fromisoformat(death[:10])):
                kinds = {tag.get("code") for tag in resource.get("meta", {}).get("tag", [])
                         if tag.get("system") == spec["metadata"]["problemKindSystem"]}
                problem_ids = {tag.get("code") for tag in resource.get("meta", {}).get("tag", [])
                               if tag.get("system") == spec["metadata"]["problemIdSystem"]}
                declared_pre_birth = any(
                    row["kind"] == "timestamp-order" and row["variant"] == "pre-birth-event"
                    and row["problemId"] in problem_ids
                    for row in allocation(spec, int(cohort.rsplit("-", 1)[-1]))
                )
                if (event.interval.start.date() < born and resource.get("resourceType") == "Observation"
                        and "timestamp-order" in kinds and declared_pre_birth):
                    declared_noise += 1
                    if len(noise_examples) < 10:
                        noise_examples.append({"cohortId": cohort, "source": resource.get("_source"),
                                               "fullUrl": resource.get("_fullUrl")})
                else:
                    failures.append(f"{cohort}: {resource.get('resourceType')} occurrence outside life interval")
    expected = round(len(patient_views) * policy["deceasedPerThousand"] / 1000)
    if len(deceased) != expected:
        failures.append(f"deceased target: observed {len(deceased)}, expected {expected}")
    pediatric_target = round(len(patient_views) * policy["pediatricPerThousand"] / 1000)
    older_count = age_counts["65-74"] + age_counts["75-plus"]
    older_bounds = [ceil(len(patient_views) * policy["olderAdultMinPercent"] / 100),
                    floor(len(patient_views) * policy["olderAdultMaxPercent"] / 100)]
    if age_counts["under-18"] != pediatric_target:
        failures.append(f"pediatric target: observed {age_counts['under-18']}, expected {pediatric_target}")
    if not older_bounds[0] <= older_count <= older_bounds[1]:
        failures.append(f"older-adult target: observed {older_count}, expected range {older_bounds}")
    return {
        "status": "pass" if not failures else "fail", "policyVersion": policy["version"],
        "denominator": len(patient_views),
        "sourcePatientRecordCount": sum(
            entry["resource"].get("resourceType") == "Patient"
            for bundle in bundles.values() for entry in bundle["entry"]
        ),
        "targetDeceased": expected, "observedDeceased": len(deceased),
        "targetPediatric": pediatric_target, "observedPediatric": age_counts["under-18"],
        "targetOlderAdultRange": older_bounds, "observedOlderAdult": older_count,
        "ageBands": dict(sorted(age_counts.items())), "clinicalOccurrencesChecked": event_count,
        "declaredPreBirthNoise": declared_noise,
        "declaredNoiseExamples": noise_examples,
        "examples": {"living": living[:3], "deceased": deceased[:3]},
        "failures": failures[:100],
    }
