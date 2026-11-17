"""Versioned source-capture tags; these codes do not describe clinical category."""

from __future__ import annotations

from .catalogs import load_catalog


SYSTEM = "urn:hdp-test-bundle:synthetic:capture-class"
CODE_SYSTEM_PATH = "data/CodeSystem-capture-class.v1.0.0.json"


def capture_class_coding(spec: dict, source: str, disposition: str) -> dict[str, str]:
    catalog = load_catalog(spec["remediation"]["capturePolicyCatalog"])
    sources = catalog["sources"]
    if source not in sources or disposition not in sources[source]["enabledDispositions"]:
        raise ValueError(f"invalid capture-class source/disposition pair: {source}/{disposition}")
    code_system = load_catalog(CODE_SYSTEM_PATH)
    if code_system.get("url") != SYSTEM or code_system.get("version") != "1.0.0":
        raise ValueError("capture-class CodeSystem identity changed")
    expected = {
        f"{name}-{kind}"
        for name, row in sources.items()
        for kind in row["enabledDispositions"]
    }
    actual = [row.get("code") for row in code_system.get("concept", [])]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("capture-class CodeSystem does not cover the exact allowed pairs")
    return {"system": SYSTEM, "code": f"{source}-{disposition}"}
