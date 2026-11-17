"""Input-only Condition capture expectations for the exact population spec.

Uses the accepted S18 pre-output projection rules without reading FHIR output.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .archetypes import primary_archetype
from .bundle import matched_controls
from .capture import project_capture
from .care_pathway import build_patient_pathway
from .catalogs import load_catalog
from .demographics import demographic_truth
from .geography import choose_locality, load_catalog as load_geography
from .nested_allocation import block_position, nested
from .pathways import care_pathway
from .problems import allocation
from .utils import sha256_json

def condition_expectation(spec: dict, root: Path) -> dict:
    geography = load_geography(root / spec["geography"]["catalog"], root)
    controls = set(matched_controls(spec))
    ordinary_variant = Counter()
    pathway_events = Counter()
    static_events = Counter()
    templates = load_catalog(spec["remediation"]["pathwayTemplateCatalog"])["archetypes"]
    template_by_variant = {
        (archetype, row["id"]): row for archetype, rows in templates.items() for row in rows
    }
    typed_source = Counter()
    typed_source_concept = Counter()
    typed_by_stratum = Counter()
    for index in range(1, spec["populationSize"] + 1):
        locality = choose_locality(geography, block_position(index)[1], 1000) if nested(spec) else choose_locality(geography, index, spec["populationSize"])
        truth = demographic_truth(spec, index, locality)
        archetype = primary_archetype(spec, index)
        problems = allocation(spec, index)
        evaluation = "problem" if problems else "matched-control" if index in controls else "ordinary"
        pathway_state = care_pathway(spec, index, archetype, force_all_sources=evaluation != "ordinary", birth_date=truth["birthDate"])
        variant = pathway_state["pathwayVariant"]
        ordinary_variant[(archetype, variant)] += evaluation == "ordinary"
        if evaluation == "ordinary":
            for template_event in template_by_variant[(archetype, variant)]["events"]:
                if template_event["role"] == "condition":
                    static_events[(archetype, template_event["code"])] += 1
        pathway = build_patient_pathway(spec, index, archetype, variant, truth["birthDate"], spec["remediation"]["pathwayTemplateCatalog"], spec["remediation"]["clinicalContentCatalog"], pathway_state["secondaryTrait"])
        captures, _routes = project_capture(pathway, spec["remediation"]["capturePolicyCatalog"], spec["seed"], locality_code=str(locality["code"]), sensitivity_catalog=spec["sensitivity"]["catalog"])
        facts = {fact.id: fact for fact in pathway.facts}
        events = {event.id: event for event in pathway.events}
        for event in pathway.events:
            if event.role == "condition" and evaluation == "ordinary":
                pathway_events[(archetype, facts[event.fact_id].code)] += 1
        for source in sorted(captures):
            source_counts = Counter()
            for capture in captures[source]:
                event = events[capture.latent_event_id]
                if event.role != "condition" or capture.disposition in {"absent", "metadata-only"}:
                    continue
                code = facts[event.fact_id].code
                typed_source[(source, evaluation)] += 1
                typed_source_concept[(source, evaluation, code)] += 1
                typed_by_stratum[(source, archetype, variant, evaluation)] += 1
                source_counts[code] += 1
    concept_totals = Counter()
    for (_, code), count in static_events.items():
        concept_totals[code] += count
    return {
        "specSha256": sha256_json(spec),
        "typedSourceConditionByConcept": [{"source": k[0], "evaluation": k[1], "snomedCode": k[2], "count": v} for k, v in sorted(typed_source_concept.items())],
        "typedSourceConditionByStratum": [{"source": k[0], "archetype": k[1], "variant": k[2], "evaluation": k[3], "count": v} for k, v in sorted(typed_by_stratum.items())],
        "staticTemplateTopConceptShare": max(concept_totals.values(), default=0) / max(1, sum(static_events.values())),
    }
