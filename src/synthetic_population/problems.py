from __future__ import annotations

import hashlib

from .archetypes import primary_archetype
from .nested_allocation import nested, block_spec, BLOCK_SIZE


DATA_QUALITY_KINDS = [
    "conflicting-diagnosis",
    "discordant-lab",
    "timestamp-order",
    "semantically-wrong-code",
    "unit-mismatch",
    "stale-demographics",
    "probable-duplicate",
]

PROBLEM_VARIANTS = {
    "conflicting-diagnosis": ("code-family-conflict", "status-lifecycle-conflict", "onset-provenance-conflict"),
    "discordant-lab": ("same-specimen-disagreement", "near-time-method-disagreement", "unit-normalized-disagreement"),
    "timestamp-order": ("issued-before-effective", "future-result", "pre-birth-event"),
    "semantically-wrong-code": ("specimen-as-diagnosis", "body-site-as-diagnosis", "procedure-as-diagnosis"),
    "unit-mismatch": ("display-code-mismatch", "scale-mismatch", "wrong-unit-family"),
    "stale-demographics": ("stale-address-snapshot", "stale-telecom-snapshot", "stale-name-snapshot"),
    "probable-duplicate": ("transliteration-spelling", "changed-phone", "minor-dob-discrepancy"),
}

SCENARIO_VARIANTS = {
    "renal-medication-risk": ("chronic-renal-trend", "acute-renal-precipitant", "medication-dose-escalation"),
    "bleeding-risk": ("overt-bleeding", "falling-hemoglobin", "interacting-exposure"),
    "allergy-treatment-risk": ("prescribing-after-allergy", "administration-after-allergy", "culture-informed-treatment"),
}


def _rank(seed: int, namespace: str, index: int) -> str:
    return hashlib.sha256(f"{seed}|{namespace}|cohort-{index:06d}".encode()).hexdigest()


def _spec_key(spec: dict) -> tuple:
    cfg = spec["problems"]
    return (
        spec["populationSize"], spec["seed"], cfg["patientsWithOne"],
        cfg["patientsWithSecond"], cfg["crossSourceClinicalPatients"],
        bool(spec.get("clinicalRealism")), bool(spec.get("remediation")),
        cfg.get("allocationPolicyVersion"),
        spec.get("archetypes", {}).get("catalog"),
        spec.get("remediation", {}).get("pathwayTemplateCatalog"),
        tuple((row["id"], row["count"]) for row in spec["archetypes"]["primaryDistribution"])
        if "primaryDistribution" in spec.get("archetypes", {}) else (),
    )


_ALLOCATION_CACHE: dict[tuple, dict[int, list[dict]]] = {}


def allocation_table(spec: dict) -> dict[int, list[dict]]:
    """Build a seeded, non-ordinal, archetype-stratified exact allocation."""
    if "archetypes" not in spec:
        return _legacy_allocation_table(spec)
    cache_key = _spec_key(spec)
    if cache_key in _ALLOCATION_CACHE:
        return _ALLOCATION_CACHE[cache_key]
    if nested(spec):
        table = {}
        for block in range(spec["populationSize"] // BLOCK_SIZE):
            plan = block_spec(spec, block)
            cfg = plan["problems"]
            population = list(range(block * BLOCK_SIZE + 1, (block + 1) * BLOCK_SIZE + 1))
            local = _allocate(spec, cfg, population, plan["seed"])
            problem_offset = block * (cfg["patientsWithOne"] + cfg["patientsWithSecond"])
            for index, rows in local.items():
                table[index] = [dict(row, problemId=f"problem-{int(row['problemId'][8:]) + problem_offset:06d}") for row in rows]
    else:
        table = _allocate(spec, spec["problems"], list(range(1, spec["populationSize"] + 1)), spec["seed"])
    _ALLOCATION_CACHE[cache_key] = table
    return table


def _allocate(spec: dict, cfg: dict, population: list[int], rank_seed: int) -> dict[int, list[dict]]:
    """Allocate one complete quota group against the actual clinical archetypes."""
    archetype_by_index = {index: primary_archetype(spec, index) for index in population}
    scenario_compatible = {
        "renal-medication-risk": {"chronic-care", "medication-intensive", "oncology", "rare-complex"},
        "bleeding-risk": {"medication-intensive", "perioperative-surgical", "oncology", "rare-complex"},
        "allergy-treatment-risk": {"chronic-care", "pediatric", "medication-intensive", "perioperative-surgical", "oncology", "rare-complex"},
    }
    scenario_targets = [
        scenario_name(position + 1, cfg["crossSourceClinicalPatients"])
        for position in range(cfg["crossSourceClinicalPatients"])
    ]
    selected_cross: list[int] = []
    scenario_by_index: dict[int, str] = {}
    used_cross: set[int] = set()
    for scenario in ("renal-medication-risk", "bleeding-risk", "allergy-treatment-risk"):
        target = scenario_targets.count(scenario)
        compatible = sorted(set(archetype_by_index.values()) & scenario_compatible[scenario])
        by_archetype = {
            archetype: sorted(
                (index for index in population if archetype_by_index[index] == archetype and index not in used_cross),
                key=lambda index: _rank(rank_seed, f"cross-source|{scenario}", index),
            )
            for archetype in compatible
        }
        chosen: list[int] = []
        while len(chosen) < target:
            progressed = False
            for archetype in compatible:
                while by_archetype[archetype] and by_archetype[archetype][0] in used_cross:
                    by_archetype[archetype].pop(0)
                if by_archetype[archetype]:
                    index = by_archetype[archetype].pop(0)
                    chosen.append(index)
                    used_cross.add(index)
                    progressed = True
                    if len(chosen) == target:
                        break
            if not progressed:
                raise ValueError(f"insufficient compatible patients for {scenario}")
        for index in chosen:
            selected_cross.append(index)
            scenario_by_index[index] = scenario
    remaining = sorted(
        (index for index in population if index not in set(selected_cross)),
        key=lambda index: _rank(rank_seed, "data-quality", index),
    )
    first_selected = [*selected_cross, *remaining[:max(0, cfg["patientsWithOne"] - len(selected_cross))]]
    table: dict[int, list[dict]] = {index: [] for index in population}
    for ordinal, index in enumerate(first_selected, 1):
        is_cross = index in selected_cross
        scenario = scenario_by_index.get(index)
        kind = "cross-source-clinical-risk" if is_cross else DATA_QUALITY_KINDS[(ordinal + int(_rank(rank_seed, "kind", index)[:4], 16)) % len(DATA_QUALITY_KINDS)]
        row = {
            "problemId": f"problem-{ordinal:06d}",
            "class": "cross-source-clinical" if is_cross else "data-quality",
            "kind": kind, "severity": "high" if is_cross else ("moderate" if ordinal % 3 else "low"),
            "scenario": scenario,
            "prerequisites": {"compatibleArchetypes": sorted(scenario_compatible[scenario]) if is_cross else sorted(set(archetype_by_index.values())), "requiresAllSources": is_cross},
        }
        row["variant"] = (
            ("chronic-trend", "acute-precipitant", "exposure-change")[((ordinal - 1) // max(1, cfg["crossSourceClinicalPatients"] // 3)) % 3]
            if is_cross else PROBLEM_VARIANTS[kind][int(_rank(rank_seed, f"problem-variant|{kind}", index)[:8], 16) % 3]
        )
        table[index].append(row)
    # Cross-source safety cases already carry a multi-resource high-severity
    # pattern. Keep secondary data-quality transformations on non-cross cases,
    # and pair only different resource families so one builder cannot overwrite
    # another problem's decisive observable fact.
    problem_family = {
        "conflicting-diagnosis": "condition", "semantically-wrong-code": "condition",
        "discordant-lab": "observation", "timestamp-order": "observation", "unit-mismatch": "observation",
        "stale-demographics": "patient", "probable-duplicate": "patient",
    }
    second_candidates = sorted(
        (index for index in first_selected if index not in set(selected_cross)),
        key=lambda index: _rank(rank_seed, "second-problem", index),
    )[:cfg["patientsWithSecond"]]
    for offset, index in enumerate(second_candidates, 1):
        used = {row["kind"] for row in table[index]}
        used_families = {problem_family[row] for row in used}
        kind = next(
            DATA_QUALITY_KINDS[(offset + step) % len(DATA_QUALITY_KINDS)]
            for step in range(len(DATA_QUALITY_KINDS))
            if DATA_QUALITY_KINDS[(offset + step) % len(DATA_QUALITY_KINDS)] not in used
            and problem_family[DATA_QUALITY_KINDS[(offset + step) % len(DATA_QUALITY_KINDS)]] not in used_families
        )
        table[index].append({
            "problemId": f"problem-{cfg['patientsWithOne'] + offset:06d}",
            "class": "data-quality", "kind": kind, "severity": "moderate",
            "scenario": None,
            "prerequisites": {"compatibleArchetypes": sorted(set(archetype_by_index.values())), "requiresAllSources": False},
            "variant": PROBLEM_VARIANTS[kind][int(_rank(rank_seed, f"second-problem-variant|{kind}", index)[:8], 16) % 3],
        })
    for kind, variants in PROBLEM_VARIANTS.items():
        rows = sorted(
            ((index, row) for index, patient_rows in table.items() for row in patient_rows if row["kind"] == kind),
            key=lambda item: item[1]["problemId"],
        )
        for position, (index, row) in enumerate(rows):
            variant = variants[position % len(variants)]
            if kind == "timestamp-order" and variant == "pre-birth-event" and archetype_by_index[index] == "pediatric":
                variant = "future-result"
            if spec.get("remediation") and kind == "conflicting-diagnosis" and variant == "code-family-conflict":
                from .pathways import care_pathway
                from .catalogs import default_detector_catalog, load_catalog
                selected = care_pathway(spec, index, archetype_by_index[index])["pathwayVariant"]
                template = next(item for item in load_catalog(spec["remediation"]["pathwayTemplateCatalog"])["archetypes"][archetype_by_index[index]] if item["id"] == selected)
                if not any(event["role"] == "condition" and event["code"] in {"46635009", "44054006"} for event in template["events"]):
                    variant = "status-lifecycle-conflict"
            row["variant"] = variant
    if spec.get("clinicalRealism"):
        for scenario, variants in SCENARIO_VARIANTS.items():
            rows = sorted(
                (row for patient_rows in table.values() for row in patient_rows if row.get("scenario") == scenario),
                key=lambda row: row["problemId"],
            )
            for position, row in enumerate(rows):
                row["variant"] = variants[position % len(variants)]
    return table


def _legacy_allocation_table(spec: dict) -> dict[int, list[dict]]:
    """Preserve the published v1 ordinal allocation contract byte-for-byte."""
    cfg = spec["problems"]
    table: dict[int, list[dict]] = {}
    for index in range(1, spec["populationSize"] + 1):
        rows: list[dict] = []
        if index <= cfg["patientsWithOne"]:
            is_cross = index <= cfg["crossSourceClinicalPatients"]
            rows.append({
                "problemId": f"problem-{index:06d}",
                "class": "cross-source-clinical" if is_cross else "data-quality",
                "kind": "cross-source-clinical-risk" if is_cross else DATA_QUALITY_KINDS[(index - 1) % len(DATA_QUALITY_KINDS)],
                "severity": "high" if is_cross else ("moderate" if index % 3 else "low"),
                "scenario": scenario_name(index, cfg["crossSourceClinicalPatients"]) if is_cross else None,
            })
        if index <= cfg["patientsWithSecond"]:
            rows.append({
                "problemId": f"problem-{cfg['patientsWithOne'] + index:06d}",
                "class": "data-quality",
                "kind": DATA_QUALITY_KINDS[(index + 2) % len(DATA_QUALITY_KINDS)],
                "severity": "moderate",
            })
        table[index] = rows
    return table


def allocation(spec: dict, index: int) -> list[dict]:
    return allocation_table(spec).get(index, [])


def scenario_name(index: int, cross_count: int) -> str:
    if cross_count <= 0:
        return "none"
    third = max(1, cross_count // 3)
    if index <= third:
        return "renal-medication-risk"
    if index <= 2 * third:
        return "bleeding-risk"
    return "allergy-treatment-risk"


def additional_fixture_conclusions(variant: str, *, control: bool = False) -> list[str]:
    """Declared overlapping conclusions for the bounded fixture contract."""
    from .catalogs import default_detector_catalog, load_catalog
    return list(load_catalog(default_detector_catalog())["fixtureAdditionalConclusions"]["control" if control else "positive"].get(variant, []))


def conditional_fixture_conclusions(variant: str, resources: list[dict]) -> list[str]:
    """Optional *positive* clinical overlaps, proven from actual constructed events."""
    from .catalogs import default_detector_catalog, load_catalog
    from .predicates import evaluate_variant
    contract = load_catalog("data/fixture-overlaps.v1.0.0.json")
    row = contract["permittedConditionalConclusions"].get(variant)
    if not row:
        return []
    return [conclusion for conclusion in row["conclusions"]
            if evaluate_variant(row["scenario"], conclusion, resources, require_three_source_context=True)]


def validate_fixture_conclusions(variant: str, declared: list[str], conditional: list[str], *, control: bool = False) -> None:
    from .catalogs import default_detector_catalog, load_catalog
    allowed = load_catalog("data/fixture-overlaps.v1.0.0.json")["permittedConditionalConclusions"].get(variant, {}).get("conclusions", [])
    if conditional != [item for item in allowed if item in conditional]:
        raise ValueError("conditional fixture conclusions are unknown, duplicated or out of order")
    base = ([] if control else [variant]) + additional_fixture_conclusions(variant, control=control)
    if declared != base + [item for item in conditional if item not in base]:
        raise ValueError("safety conclusions differ from the versioned fixture contracts")
