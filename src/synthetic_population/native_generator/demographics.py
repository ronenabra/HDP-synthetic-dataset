from __future__ import annotations

from datetime import date, timedelta
import hashlib

from .archetypes import distribution, primary_archetype, archetype_ordinal
from .nested_allocation import nested, block_position
from .catalogs import load_catalog
from .ids import cohort_id, deterministic_uuid, strong_identifier
from .metadata import meta
from .utils import sha256_json


_GENDER_CACHE: dict[tuple, dict[int, str]] = {}


def gender_for_patient(spec: dict, index: int) -> str:
    if nested(spec):
        block, _ = block_position(index)
        key = ("nested", spec["seed"], block, spec["archetypes"]["catalog"],
               tuple(sorted(spec["demographics"]["genderDistribution"].items())))
        if key not in _GENDER_CACHE:
            start = block * 1000
            target_female = round(1000 * spec["demographics"]["genderDistribution"]["female"])
            mandatory_female = {2} if block == 0 else set()
            mandatory_male = {1} if block == 0 else set()
            mandatory_female.update(patient_index for patient_index in range(start + 1, start + 1001)
                                    if primary_archetype(spec, patient_index) == "womens-health")
            if mandatory_female & mandatory_male or len(mandatory_female) > target_female:
                raise ValueError("nested allocation infeasible: sex and archetype eligibility")
            candidates = sorted((patient_index for patient_index in range(start + 1, start + 1001)
                                 if patient_index not in mandatory_female | mandatory_male),
                                key=lambda patient_index: hashlib.sha256(
                                    (f"{spec['seed']}|sex|cohort-{patient_index:06d}" if block == 0 else
                                     f"{spec['seed']}|sex|block-{block}|cohort-{patient_index:06d}").encode()).hexdigest())
            female = mandatory_female | set(candidates[:target_female - len(mandatory_female)])
            _GENDER_CACHE[key] = {patient_index: "female" if patient_index in female else "male"
                                  for patient_index in range(start + 1, start + 1001)}
        return _GENDER_CACHE[key][index]
    key = (spec["populationSize"], spec["seed"], tuple((row["id"], row["count"]) for row in distribution(spec)), tuple(sorted(spec["demographics"]["genderDistribution"].items())))
    if key not in _GENDER_CACHE:
        target_female = round(spec["populationSize"] * spec["demographics"]["genderDistribution"]["female"])
        mandatory_female = {2}
        mandatory_male = {1}
        for patient_index in range(1, spec["populationSize"] + 1):
            if primary_archetype(spec, patient_index) == "womens-health":
                mandatory_female.add(patient_index)
        candidates = sorted(
            (patient_index for patient_index in range(1, spec["populationSize"] + 1) if patient_index not in mandatory_female | mandatory_male),
            key=lambda patient_index: hashlib.sha256(f"{spec['seed']}|sex|cohort-{patient_index:06d}".encode()).hexdigest(),
        )
        female = mandatory_female | set(candidates[:max(0, target_female - len(mandatory_female))])
        _GENDER_CACHE[key] = {patient_index: "female" if patient_index in female else "male" for patient_index in range(1, spec["populationSize"] + 1)}
    return _GENDER_CACHE[key][index]


def _pediatric_birth_date(spec: dict, pediatric_position: int) -> date:
    anchor = date.fromisoformat(spec["timelineAnchor"][:10])
    base = spec["demographics"]["pediatricAgeBands"]
    pediatric_count = next(row["count"] for row in distribution(spec) if row["id"] == "pediatric")
    total = sum(base.values())
    exact = {key: pediatric_count * value / total for key, value in base.items()}
    bands = {key: int(value) for key, value in exact.items()}
    for key in sorted(base, key=lambda item: (exact[item] - bands[item], item), reverse=True)[:pediatric_count - sum(bands.values())]:
        bands[key] += 1
    if spec["populationSize"] != 1000:
        if pediatric_position <= bands["infant"]:
            return anchor - timedelta(days=30 + (pediatric_position * 17) % 300)
        if pediatric_position <= bands["infant"] + bands["child"]:
            offset = pediatric_position - bands["infant"] - 1
            age = 1 + offset % 12
            return date(anchor.year - age, anchor.month, anchor.day) - timedelta(days=(offset * 11) % 300)
        offset = pediatric_position - bands["infant"] - bands["child"] - 1
        age = 13 + offset % 5
        return date(anchor.year - age, anchor.month, anchor.day) - timedelta(days=(offset * 13) % 300)
    if pediatric_position <= bands["infant"]:
        return anchor - timedelta(days=30 + (pediatric_position * 17) % 300)
    if pediatric_position <= bands["infant"] + bands["child"]:
        offset = pediatric_position - bands["infant"]
        return anchor - timedelta(days=365 * (2 + offset % 10) + offset * 11 % 365)
    offset = pediatric_position - bands["infant"] - bands["child"]
    return anchor - timedelta(days=365 * (13 + offset % 5) + offset * 13 % 365)


def demographic_truth(spec: dict, index: int, locality: dict) -> dict:
    cohort = cohort_id(index)
    is_v2 = spec["schemaVersion"] == "2.0.0"
    archetype = primary_archetype(spec, index) if is_v2 else "strict-patient-clinical"
    catalog = load_catalog(spec["demographics"]["namesCatalog"]) if is_v2 else {
        "givenMale": ["Noam", "Ariel", "Daniel", "Eitan", "Yosef", "Omer", "Lior", "Amir"],
        "givenFemale": ["Yael", "Maya", "Tamar", "Noa", "Leah", "Shira", "Michal", "Dana"],
        "families": ["Levi", "Cohen", "Mizrahi", "Peretz", "Biton", "Avraham", "Shalev", "Azoulay"],
    }
    if is_v2:
        gender = gender_for_patient(spec, index)
    else:
        gender = "male" if index % 2 else "female"
    anchor = next((item for item in spec["identity"]["anchorPatients"] if item["cohortId"] == cohort), None)
    names = catalog["givenMale"] if gender == "male" else catalog["givenFemale"]
    given = anchor["given"] if anchor else names[(index * 17 + index // 32) % len(names)]
    family = anchor["family"] if anchor else catalog["families"][(index // len(names) + index * 11) % len(catalog["families"])]
    if not is_v2 and index % 40 == 0:
        born = date(2010, 1, 1) + timedelta(days=(index * 137) % 1800)
    elif archetype == "pediatric":
        born = _pediatric_birth_date(
            {**spec, "populationSize": 1000} if nested(spec) else spec,
            archetype_ordinal(spec, index, archetype) + 1,
        )
    else:
        born = date(1940, 1, 1) + timedelta(days=(index * 7919) % 24400)
    return {
        "cohortId": cohort, "given": given, "family": family, "gender": gender,
        "birthDate": born.isoformat(), "locality": locality,
        "streetToken": f"Synthetic {1 + (index * 29) % 997} Street",
        "building": str(1 + index % 63),
        "phone": f"{spec['demographics']['telephonePrefix']}{index % 1000:03d}",
        "anchorTestId": anchor["testId"] if anchor else None,
        "primaryArchetype": archetype,
    }


def patient_resource(spec: dict, organization: str, truth: dict, index: int, problem_ids: list[str] | None = None,
                     problem_kinds: list[str] | None = None, problem_variants: dict[str, str] | None = None) -> tuple[dict, str, str]:
    version = spec["datasetVersion"]
    cohort = truth["cohortId"]
    identifier_value = deterministic_uuid(version, organization, cohort, "Patient")
    is_v2 = spec["schemaVersion"] == "2.0.0"
    snapshot_start = {"hmo": "2020-01-01", "hospital-a": "2023-01-01", "hospital-b": "2024-01-01"}[organization]
    name = {"family": truth["family"], "given": [truth["given"]], "use": "official"}
    if is_v2:
        name["period"] = {"start": snapshot_start}
    if index % 100 < int(spec["demographics"]["nameVariationRate"] * 100):
        if organization == "hospital-a" and len(name["family"]) > 5:
            name["family"] = name["family"][:-1] + "-" + name["family"][-1]
        elif organization == "hospital-b":
            name = {**name, "family": truth["family"].upper(), "use": "usual"}
    city_extension = {"extension": [{"url": "http://fhir.health.gov.il/StructureDefinition/ext-city-code", "valueCodeableConcept": {"coding": [{"system": "http://fhir.health.gov.il/cs/city-symbol", "code": truth["locality"]["code"], "display": truth["locality"]["nameHe"]}]}}]}
    address = {
        "use": "home", "type": "physical", "line": [truth["streetToken"], truth["building"]],
        "city": truth["locality"]["nameEn"], "district": truth["locality"]["district"], "country": "IL",
        "_city": city_extension,
    }
    if is_v2:
        address["period"] = {"start": snapshot_start}
    if organization == "hospital-a" and index % 100 < int(spec["demographics"]["addressVariationRate"] * 100):
        address["line"] = [truth["building"], truth["streetToken"]]
    telecom = [{"system": "phone", "value": truth["phone"], "use": "mobile"}]
    if is_v2:
        telecom[0]["period"] = {"start": snapshot_start}
    if organization == "hospital-b" and index % 100 < int(spec["demographics"]["telecomVariationRate"] * 100):
        if is_v2:
            telecom[0]["value"] = truth["phone"] + f"x{index % 10}"
        else:
            telecom = []
    identifiers = [strong_identifier(spec, organization, "Patient", identifier_value), {"system": spec["identity"]["cohortIdentifierSystem"], "value": cohort}]
    if truth["anchorTestId"]:
        identifiers.append({"system": spec["identity"]["testIdentifierSystem"], "value": truth["anchorTestId"]})
    resource = {
        "resourceType": "Patient",
        "meta": meta(spec, organization, "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-patient", "patient-demographics",
                     problem_ids=problem_ids or [], problem_kinds=problem_kinds or [], archetype=truth["primaryArchetype"]),
        "identifier": identifiers, "active": True, "name": [name],
        "gender": truth["gender"], "birthDate": truth["birthDate"], "address": [address],
        "extension": [{"url": "http://fhir.health.gov.il/StructureDefinition/ext-il-hmo", "valueCodeableConcept": {"coding": [{"system": "http://fhir.health.gov.il/cs/paying-entity-moh", "code": "101", "display": "קופת חולים כללית"}]}}],
        "text": {"status": "generated", "div": f"<div xmlns=\"http://www.w3.org/1999/xhtml\">Synthetic {truth['primaryArchetype']} patient view.</div>"},
    }
    if telecom:
        resource["telecom"] = telecom
    resource["meta"]["lastUpdated"] = {
        "hmo": "2025-12-01T08:00:00Z", "hospital-a": "2025-12-02T09:30:00Z",
        "hospital-b": "2025-12-03T11:00:00Z",
    }[organization]
    if "stale-demographics" in (problem_kinds or []):
        variant = (problem_variants or {}).get("stale-demographics")
        resource["meta"]["lastUpdated"] = {"stale-address-snapshot": "2021-12-31T08:00:00Z", "stale-telecom-snapshot": "2020-06-30T08:00:00Z", "stale-name-snapshot": "2019-12-31T08:00:00Z"}.get(variant, "2021-12-31T08:00:00Z")
        if spec.get("remediation"):
            # One stale-field variant must not accidentally assert future element
            # periods or unrelated source-format changes as additional conflicts.
            start = resource["meta"]["lastUpdated"][:10]
            resource["name"] = [{"family": truth["family"], "given": [truth["given"]],
                                 "use": "official", "period": {"start": start}}]
            resource["address"][0]["line"] = [truth["streetToken"], truth["building"]]
            resource["address"][0]["period"] = {"start": start}
            resource["telecom"] = [{"system": "phone", "value": truth["phone"],
                                    "use": "mobile", "period": {"start": start}}]
        if variant == "stale-telecom-snapshot":
            resource["telecom"][0]["value"] = resource["telecom"][0]["value"] + "x9"
        elif variant == "stale-name-snapshot":
            resource["name"][0]["use"] = "old"
        else:
            resource["address"] = [{
                **address,
                "line": [f"Synthetic Former {index} Street", truth["building"]],
                "period": {"start": "2017-01-01"},
            }]
    anchor_digest = sha256_json({key: truth[key] for key in ("cohortId", "birthDate", "gender")})
    view_digest = sha256_json({"name": resource["name"], "address": resource["address"], "telecom": resource.get("telecom", [])})
    return resource, anchor_digest, view_digest
