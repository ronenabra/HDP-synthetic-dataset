"""Versioned, fixed-size allocation blocks for nested v4 populations."""
from __future__ import annotations

import copy
import hashlib

POLICY_VERSION = "4.2.0-nested-cohorts"
BLOCK_SIZE = 1000


def nested(spec: dict) -> bool:
    return spec.get("problems", {}).get("allocationPolicyVersion") == POLICY_VERSION


def block_position(index: int) -> tuple[int, int]:
    return divmod(index - 1, BLOCK_SIZE)[0], (index - 1) % BLOCK_SIZE + 1


def block_spec(spec: dict, block: int) -> dict:
    """Return a feasible 1,000-person plan, independent of the requested total."""
    result = copy.deepcopy(spec)
    result["populationSize"] = BLOCK_SIZE
    result["problems"]["allocationPolicyVersion"] = "4.1.0-natural-pilot"
    if block:
        digest = hashlib.sha256(f"{spec['seed']}|allocation-block|{block}".encode()).digest()
        result["seed"] = int.from_bytes(digest[:8], "big")
    for key in ("patientsWithOne", "patientsWithSecond", "crossSourceClinicalPatients"):
        count = spec["problems"][key]
        if count % (spec["populationSize"] // BLOCK_SIZE):
            raise ValueError(f"nested allocation infeasible: {key} quota is not block divisible")
        result["problems"][key] = count // (spec["populationSize"] // BLOCK_SIZE)
    for key, count in spec["sensitivity"]["factCohorts"].items():
        if count % (spec["populationSize"] // BLOCK_SIZE):
            raise ValueError(f"nested allocation infeasible: {key} quota is not block divisible")
        result["sensitivity"]["factCohorts"][key] = count // (spec["populationSize"] // BLOCK_SIZE)
    return result
