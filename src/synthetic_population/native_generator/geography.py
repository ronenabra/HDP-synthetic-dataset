from __future__ import annotations

import bisect
import math
from pathlib import Path

from .utils import read_json, sha256_file
from .catalogs import input_path


def load_catalog(path: Path, repository_root: Path) -> dict:
    catalog = read_json(path)
    code_source = input_path(catalog["provenance"]["codeSource"])
    if sha256_file(code_source) != catalog["provenance"]["codeSourceSha256"]:
        raise ValueError("pinned city code source digest changed")
    code_system = read_json(code_source)
    valid = {item["code"] for item in code_system["concept"]}
    if any(item["code"] not in valid for item in catalog["localities"]):
        raise ValueError("locality catalog contains an unknown IL Core city symbol")
    return catalog


def choose_locality(catalog: dict, index: int, population_size: int) -> dict:
    rows = catalog["localities"]
    district_rates = catalog.get("districtDistribution")
    if not district_rates:
        district_rates = {row["district"]: 0 for row in rows}
        for row in rows:
            district_rates[row["district"]] += float(row["weight"])
        total = sum(district_rates.values())
        district_rates = {district: weight / total for district, weight in district_rates.items()}
    exact = {district: population_size * rate for district, rate in district_rates.items()}
    counts = {district: math.floor(value) for district, value in exact.items()}
    for district in sorted(district_rates, key=lambda item: (exact[item] - counts[item], item), reverse=True)[:population_size - sum(counts.values())]:
        counts[district] += 1
    permutation = (index * 2654435761) % population_size
    cursor = 0
    selected_district = None
    for district, count in counts.items():
        cursor += count
        if permutation < cursor:
            selected_district = district
            break
    district_rows = [row for row in rows if row["district"] == selected_district]
    weights = [float(row["weight"]) for row in district_rows]
    cumulative = []
    total = 0.0
    for weight in weights:
        total += weight
        cumulative.append(total)
    point = (((index * 2246822519 + 3266489917) % 1000003) / 1000003) * total
    return district_rows[min(bisect.bisect_left(cumulative, point), len(district_rows) - 1)]
