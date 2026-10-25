"""Developer checks only; these fixtures are never used as project results."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from criteo_ab_uplift.data import FEATURES, BINARY_COLUMNS, prepare_data
from criteo_ab_uplift.adjustment import lin_adjustment


def _fixture(path: Path, n: int = 600):
    rng = np.random.default_rng(41)
    x = rng.normal(size=(n, 12))
    # Values not representable exactly by float32 distinguish the two caches.
    x[:, 3] += 123.123456789
    t = rng.binomial(1, 0.7, n)
    visit = rng.binomial(1, 1 / (1 + np.exp(1.4 - 0.4 * x[:, 0] - 0.3 * t)))
    conversion = visit * rng.binomial(1, 0.15, n)
    exposure = t * rng.binomial(1, 0.5, n)
    frame = pd.DataFrame(x, columns=FEATURES)
    frame[BINARY_COLUMNS] = np.column_stack([t, visit, conversion, exposure])
    frame.to_csv(path, index=False)
    return frame


def test_full_file_audit_cache_and_precision(tmp_path):
    path = tmp_path / "input.csv.gz"
    frame = _fixture(path)
    cache = tmp_path / "cache"
    audit = prepare_data(path, cache, expected_rows=len(frame), threads=1)
    x32 = np.load(cache / "X.npy", mmap_mode="r")
    x64 = np.load(cache / "X64.npy", mmap_mode="r")
    assert x32.shape == x64.shape == (len(frame), 12)
    assert x32.dtype == np.float32
    assert x64.dtype == np.float64
    np.testing.assert_array_equal(x64, frame[FEATURES].to_numpy())
    np.testing.assert_array_equal(np.load(cache / "labels.npy"), frame[BINARY_COLUMNS].to_numpy())
    np.testing.assert_allclose(audit["feature_means"], frame[FEATURES].mean(), atol=2e-13)
    np.testing.assert_allclose(audit["feature_stds"], frame[FEATURES].std(), atol=2e-13)
    assert sum(item["n"] for item in audit["joint_counts"]) == len(frame)
    assert len(audit["joint_counts"]) == 8
    assert audit["control_exposure_count"] == 0
    for item in audit["groups"]:
        selected = frame[frame.treatment == item["treatment"]]
        assert item["n"] == len(selected)
        assert item["visit_exposure_sum"] == int((selected.visit * selected.exposure).sum())
    before = (cache / "X.npy").stat().st_mtime_ns
    reused = prepare_data(path, cache, expected_rows=len(frame))
    assert reused["data_sha256"] == audit["data_sha256"]
    assert (cache / "X.npy").stat().st_mtime_ns == before


def test_wrong_production_row_count_stops_before_materialization(tmp_path):
    path = tmp_path / "input.csv"
    _fixture(path, 50)
    with pytest.raises(ValueError, match="13,979,592"):
        prepare_data(path, tmp_path / "cache", threads=1)
    assert not (tmp_path / "cache").exists()


def test_invalid_indicator_rejected_not_rounded(tmp_path):
    path = tmp_path / "input.csv"
    frame = _fixture(path, 50)
    frame["visit"] = frame["visit"].astype(float)
    frame.loc[0, "visit"] = 0.5
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Invalid values"):
        prepare_data(path, tmp_path / "cache", expected_rows=50, threads=1)


def test_cache_rebuilt_when_source_hash_changes(tmp_path):
    path = tmp_path / "input.csv"
    frame = _fixture(path, 50)
    cache = tmp_path / "cache"
    original = prepare_data(path, cache, expected_rows=50, threads=1)
    frame.loc[0, "f0"] += 10.0
    frame.to_csv(path, index=False)
    updated = prepare_data(path, cache, expected_rows=50, threads=1)
    assert original["data_sha256"] != updated["data_sha256"]
    assert np.load(cache / "X64.npy")[0, 0] == frame.loc[0, "f0"]


def test_lin_chunked_matches_dense_fully_interacted_hc2(tmp_path):
    path = tmp_path / "input.csv"
    frame = _fixture(path)
    audit = prepare_data(path, tmp_path / "cache", expected_rows=len(frame), threads=1)
    result = lin_adjustment(tmp_path / "cache", audit, chunk_rows=37)
    x = frame[FEATURES].to_numpy()
    centered = (x - np.array(audit["feature_means"])) / np.array(audit["feature_stds"])
    t = frame.treatment.to_numpy()
    design = np.column_stack([np.ones(len(frame)), t, centered, t[:, None] * centered])
    inverse = np.linalg.inv(design.T @ design)
    leverage = np.einsum("ij,ij->i", design @ inverse, design)
    for name in ("visit", "conversion"):
        y = frame[name].to_numpy()
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - design @ coef
        meat = design.T @ ((residual**2 / (1 - leverage))[:, None] * design)
        variance = inverse @ meat @ inverse
        got = result["outcomes"][name]
        assert got["tau_adjusted"] == pytest.approx(coef[1], abs=2e-12)
        assert got["se_adjusted"] == pytest.approx(np.sqrt(variance[1, 1]), abs=2e-12)
        se_unadjusted = np.sqrt(sum(frame.loc[frame.treatment == arm, name].var(ddof=1)
                                   / (frame.treatment == arm).sum() for arm in (0, 1)))
        assert got["se_unadjusted_hc2"] == pytest.approx(se_unadjusted, abs=1e-14)
        assert got["se_ratio"] == pytest.approx(got["se_adjusted"] / se_unadjusted)
    other_chunking = lin_adjustment(tmp_path / "cache", audit, chunk_rows=101)
    assert other_chunking["outcomes"]["visit"]["se_adjusted"] == pytest.approx(
        result["outcomes"]["visit"]["se_adjusted"], abs=1e-12)


def test_lin_constant_covariate_uses_estimable_contrast_without_ridge(tmp_path):
    path = tmp_path / "input.csv"
    frame = _fixture(path)
    frame["f11"] = 9.0
    frame.to_csv(path, index=False)
    audit = prepare_data(path, tmp_path / "cache", expected_rows=len(frame), threads=1)
    with pytest.warns(RuntimeWarning, match="rank"):
        result = lin_adjustment(tmp_path / "cache", audit, chunk_rows=100)
    assert [item["rank"] for item in result["diagnostics"]] == [12, 12]
    assert result["outcomes"]["visit"]["se_adjusted"] > 0
