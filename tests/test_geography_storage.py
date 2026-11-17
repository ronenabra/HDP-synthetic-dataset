import pytest

from synthetic_population.geography import choose_locality, load_catalog
from synthetic_population.utils import GIB, StorageError, assert_storage_safe, projected_storage, validate_output_target


def test_locality_catalog_matches_pinned_il_core(repository_root, tiny_spec):
    catalog = load_catalog(repository_root / tiny_spec["geography"]["catalog"], repository_root)
    assert len({row["district"] for row in catalog["localities"]}) == 7
    assert choose_locality(catalog, 1, 1000) == choose_locality(catalog, 1, 1000)


def test_projection_formula(tiny_spec):
    report = projected_storage(tiny_spec, 1000)
    assert report["projectedFinalBytes"] == tiny_spec["populationSize"] * 1000
    assert report["projectedPeakBytes"] == 3 * tiny_spec["populationSize"] * 1000 + 2 * GIB
    assert report["reserveBytes"] == 10 * GIB


def test_output_target_scope(tmp_path):
    good = tmp_path / "generated" / "synthetic-population.test"
    assert validate_output_target(good) == good.resolve()
    with pytest.raises(ValueError):
        validate_output_target(tmp_path / "generated" / "other")
    with pytest.raises(ValueError):
        validate_output_target(tmp_path / "synthetic-population.test")


def test_low_space_simulation_refuses(monkeypatch, tmp_path, tiny_spec):
    monkeypatch.setattr("synthetic_population.utils.available_bytes", lambda _: 1)
    with pytest.raises(StorageError):
        assert_storage_safe(tiny_spec, tmp_path / "generated" / "synthetic-population.test")


def test_mid_generation_storage_pause_is_not_published(monkeypatch, tmp_path, tiny_spec, repository_root):
    from synthetic_population.manifest import generate

    calls = 0

    def simulated_storage_check(spec, out, bytes_per_patient=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"availableBytes": 100, "requiredBytes": 10, "safe": True, "bytesPerPatient": 1,
                    "projectedFinalBytes": 10, "projectedPeakBytes": 20, "reserveBytes": 10}
        raise StorageError({"availableBytes": 1, "requiredBytes": 10, "safe": False})

    monkeypatch.setattr("synthetic_population.manifest.assert_storage_safe", simulated_storage_check)
    target = tmp_path / "generated" / "synthetic-population.paused"
    with pytest.raises(StorageError):
        generate(tiny_spec, target, repository_root)
    assert not target.exists()
    partials = list(target.parent.glob(".synthetic-population.paused.partial-*"))
    assert len(partials) == 1
    assert (partials[0] / "storage-pause.json").is_file()
