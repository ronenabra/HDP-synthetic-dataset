from __future__ import annotations

import hashlib
from datetime import date
from typing import Any

from .archetypes import distribution, archetype_ordinal
from .nested_allocation import nested
from .catalogs import load_catalog


def _slot(spec: dict[str, Any], index: int) -> int:
    digest = hashlib.sha256(f"{spec['seed']}|care-pathway|cohort-{index:06d}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 100


def _choice(spec: dict[str, Any], index: int, namespace: str, size: int) -> int:
    digest = hashlib.sha256(f"{spec['seed']}|{namespace}|cohort-{index:06d}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % size


REASONS = (
    ("185349003", "Encounter for check up"), ("49727002", "Cough"),
    ("195662009", "Acute upper respiratory infection"), ("34014006", "Viral disease"),
    ("38341003", "Hypertensive disorder"), ("161891005", "Backache"),
    ("386661006", "Fever"), ("21522001", "Abdominal pain"),
    ("233604007", "Pneumonia"), ("74400008", "Appendicitis"),
    ("125666000", "Burn"), ("89164003", "Breast lump"),
)


def care_pathway(spec: dict[str, Any], index: int, archetype: str, *, force_all_sources: bool = False,
                 birth_date: str | None = None) -> dict[str, Any]:
    """Deterministic utilization state machine shared by generation and QA."""
    slot = _slot(spec, index)
    allocation = load_catalog(spec["sourceModels"]["catalog"])["utilizationAllocation"]
    hmo_only_end = allocation["hmoOnlyPercent"]
    hospital_a_end = hmo_only_end + allocation["hmoPlusHospitalAPercent"]
    hospital_b_end = hospital_a_end + allocation["hmoPlusHospitalBPercent"]
    active = {"hmo"}
    pattern = "hmo-only"
    if hmo_only_end <= slot < hospital_a_end:
        active.add("hospital-a")
        pattern = "hmo-plus-hospital-a"
    elif hospital_a_end <= slot < hospital_b_end:
        active.add("hospital-b")
        pattern = "hmo-plus-hospital-b"
    elif slot >= hospital_b_end:
        active.update({"hospital-a", "hospital-b"})
        pattern = "multi-hospital"
    if archetype == "perioperative-surgical":
        active.add("hospital-a")
        pattern = "surgical-referral"
    elif archetype == "oncology":
        active.update({"hospital-a", "hospital-b"})
        pattern = "distributed-oncology"
    elif archetype == "rare-complex":
        active.add("hospital-a" if index % 2 else "hospital-b")
        if index % 4 == 0:
            active.update({"hospital-a", "hospital-b"})
        pattern = "multi-specialty"
    if force_all_sources:
        active.update({"hospital-a", "hospital-b"})
        pattern = "scenario-required-multi-source"
    hospital_modes = {
        "hospital-a": ("IMP", "inpatient", 24 + (index * 17) % 97) if index % 5 < 2 else (("EMER", "emergency", 4 + (index * 7) % 9) if index % 5 == 2 else ("AMB", "specialty-ambulatory", 1 + (index % 3))),
        "hospital-b": ("IMP", "short-inpatient", 18 + (index * 13) % 55) if index % 6 == 0 else (("EMER", "regional-emergency", 2 + (index * 5) % 7) if index % 3 == 0 else ("AMB", "regional-specialty", 1 + (index % 2))),
    }
    if "clinicalRealism" not in spec:
        return {
            "id": pattern, "activeSources": sorted(active), "slot": slot,
            "encounters": {
                "hmo": {"class": "AMB", "role": "community-follow-up", "durationHours": round(.5 + (index % 4) * .25, 2)},
                **{
                    organization: {"class": values[0], "role": values[1], "durationHours": values[2]}
                    for organization, values in hospital_modes.items()
                },
            },
        }
    encounters = {
        "hmo": {"class": "AMB", "role": "community-follow-up", "durationHours": round(.5 + (index % 4) * .25, 2)},
        **{
            organization: {"class": values[0], "role": values[1], "durationHours": values[2]}
            for organization, values in hospital_modes.items()
        },
    }
    realism_catalog = load_catalog(spec["clinicalRealism"]["catalog"])
    capture_catalog = load_catalog(realism_catalog["sourceCaptureCatalog"])
    pathway_variants = realism_catalog["pathwayVariants"][archetype]
    local_ordinal = archetype_ordinal(spec, index, archetype)
    seed_offset = _choice(spec, 0, f"pathway-variant-offset|{archetype}", len(pathway_variants))
    pathway_variant_index = (local_ordinal + seed_offset) % len(pathway_variants)
    pathway_variant = pathway_variants[pathway_variant_index]
    if archetype == "womens-health" and pathway_variant == "age-eligible-screening" and birth_date:
        anchor = date.fromisoformat(spec["timelineAnchor"][:10])
        born = date.fromisoformat(birth_date)
        age = anchor.year - born.year - ((anchor.month, anchor.day) < (born.month, born.day))
        try:
            thirtieth = born.replace(year=born.year + 30)
        except ValueError:
            thirtieth = born.replace(year=born.year + 30, day=28)
        ineligible = age < 30 or (anchor - thirtieth).days < 50
        if age > 74 and not ineligible:
            if spec.get("remediation"):
                # An older patient can still have an eligible screening order
                # in the earlier frozen timeline. Reuse the event builder's
                # exact order-time selection for this boundary route.
                from .care_pathway import build_patient_pathway
                try:
                    build_patient_pathway(
                        spec, index, archetype, pathway_variant, birth_date,
                        spec["remediation"]["pathwayTemplateCatalog"],
                        spec["remediation"]["clinicalContentCatalog"],
                    )
                except ValueError as error:
                    if str(error) != "routine screening template selected for an ineligible age":
                        raise
                    ineligible = True
            else:
                ineligible = True
        if ineligible:
            pathway_variant = "preventive-risk-assessment"
            pathway_variant_index = pathway_variants.index(pathway_variant)
    if archetype == "pediatric":
        # Age-named variants are constrained by observable birth date; acute
        # variants remain available across clinically compatible pediatric bands.
        from .demographics import _pediatric_birth_date
        anchor = spec["timelineAnchor"][:10]
        born = _pediatric_birth_date({**spec, "populationSize": 1000} if nested(spec) else spec, local_ordinal + 1)
        anchor_year, anchor_month, anchor_day = map(int, anchor.split("-"))
        age = anchor_year - born.year - ((anchor_month, anchor_day) < (born.month, born.day))
        allowed = (
            ["infant-well-child", "respiratory-acute", "gastrointestinal-acute"] if age < 1
            else ["child-well-child", "respiratory-acute", "gastrointestinal-acute", "minor-injury"] if age < 13
            else ["adolescent-preventive", "respiratory-acute", "gastrointestinal-acute", "minor-injury"]
        )
        pathway_variant = allowed[(local_ordinal + seed_offset) % len(allowed)]
        pathway_variant_index = pathway_variants.index(pathway_variant)
    for organization, encounter in encounters.items():
        variants = capture_catalog["sources"][organization]["branches"]
        selected_variants = variants
        if archetype in {"chronic-care", "medication-intensive", "womens-health", "oncology", "rare-complex"}:
            selected_variants = [row for row in variants if "Observation" in row["resourceTypes"]]
        selected = selected_variants[
            (
                pathway_variant_index
                + _choice(spec, index, f"capture|{organization}|{archetype}", len(selected_variants))
            ) % len(selected_variants)
        ]
        reason = REASONS[
            (pathway_variant_index * 2 + _choice(
                spec, index, f"reason|{organization}|{archetype}", len(REASONS)
            )) % len(REASONS)
        ]
        encounter.update({
            "captureVariant": selected["id"],
            "captureResourceTypes": list(selected["resourceTypes"]),
            "captureMode": selected["captureMode"],
            "workflowFamily": selected["workflowFamily"],
            "reason": {"system": "http://snomed.info/sct", "code": reason[0], "display": reason[1]},
            "documentationLagMinutes": 15 + 15 * _choice(spec, index, f"documentation-lag|{organization}", 16),
        })
    secondary_trait = (
        ("none", "metabolic-risk", "respiratory-history", "medication-exposure")[
            _choice(spec, index, f"secondary-trait|{archetype}", 4)
        ]
        if archetype in {"chronic-care", "medication-intensive", "oncology", "rare-complex"}
        else "none"
    )
    return {
        "id": pattern, "activeSources": sorted(active), "slot": slot,
        "pathwayVariant": pathway_variant,
        "secondaryTrait": secondary_trait,
        "encounters": encounters,
    }


def utilizes_source(spec: dict[str, Any], index: int, archetype: str, organization: str, *, force_all_sources: bool = False) -> bool:
    return organization in care_pathway(spec, index, archetype, force_all_sources=force_all_sources)["activeSources"]
