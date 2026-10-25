"""Numerical equivalence checks for the optimized metric implementation.

These are developer checks, not a user workflow or a substitute for real data.
"""
from pathlib import Path

import numpy as np
import pytest
from sklift.metrics import qini_auc_score, qini_curve

from criteo_ab_uplift.uplift import (
    BUDGETS, FIXED_PARAMETERS, METRIC_NAMES, POLICIES, RankedPolicy,
    _bootstrap, _metric_vector, _policy_predictions, _split,
)


def _fixture():
    rng = np.random.default_rng(83)
    n = 400
    treatment = np.tile(np.asarray([0, 1], dtype=np.int8), n // 2)
    outcome = rng.binomial(1, 0.2 + 0.08 * treatment).astype(np.int8)
    # Deliberate ties matter both at top-k boundaries and in Qini integration.
    scores = rng.integers(-3, 5, n).astype(np.float64)
    tie_keys = rng.permutation(n)
    return outcome, treatment, scores, tie_keys


def _literal(y, t, scores, keys, counts):
    expanded = np.repeat(np.arange(len(y)), counts)
    order = np.lexsort((keys[expanded], -scores[expanded]))
    selected_order = expanded[order]
    yy, tt = y[expanded], t[expanded]
    random = 1000 * (yy[tt == 1].mean() - yy[tt == 0].mean())
    metrics = [qini_auc_score(yy, scores[expanded], tt, negative_effect=True), random]
    u = []
    for k in BUDGETS:
        top = selected_order[:int(np.floor(k * len(expanded)))]
        u.append(1000 * (y[top][t[top] == 1].mean() - y[top][t[top] == 0].mean()))
    return np.asarray(metrics + u + [v - random for v in u])


def test_qini_matches_sklift_with_ties_and_complete_curve():
    y, t, scores, keys = _fixture()
    policy = RankedPolicy.prepare(y, t, scores, keys)
    result, (x, q) = policy.evaluate(return_curve=True)
    sx, sq = qini_curve(y, scores, t)
    np.testing.assert_array_equal(x, sx)
    np.testing.assert_allclose(q, sq, rtol=1e-14, atol=1e-12)
    assert result["qini_coefficient"] == pytest.approx(
        qini_auc_score(y, scores, t, negative_effect=True), abs=1e-12)


def test_weighted_bootstrap_matches_literal_row_resampling():
    y, t, scores, keys = _fixture()
    policy = RankedPolicy.prepare(y, t, scores, keys)
    rng = np.random.default_rng(777)
    for _ in range(30):
        counts = np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y))
        result, _ = policy.evaluate(counts)
        np.testing.assert_allclose(_metric_vector(result), _literal(y, t, scores, keys, counts),
                                   rtol=1e-12, atol=1e-10)


def test_exact_top_k_splits_duplicate_boundary_row():
    y = np.asarray([1, 0, 0, 1], dtype=np.int8)
    t = np.asarray([1, 0, 1, 0], dtype=np.int8)
    scores = np.asarray([4.0, 3.0, 2.0, 1.0])
    counts = np.asarray([2, 8, 6, 4], dtype=np.int32)
    result, _ = RankedPolicy.prepare(y, t, scores, np.arange(4)).evaluate(counts, budgets=(0.3,))
    row = result["targeting"][0]
    assert row["n_selected"] == 6
    assert row["n_treatment"] == 2
    assert row["n_control"] == 4
    assert row["incremental_visits_per_1000"] == 1000.0


def test_tie_keys_only_change_targeting_not_qini():
    y, t, scores, keys = _fixture()
    a, _ = RankedPolicy.prepare(y, t, scores, keys).evaluate()
    b, _ = RankedPolicy.prepare(y, t, scores, keys[::-1]).evaluate()
    assert a["qini_coefficient"] == b["qini_coefficient"]


def test_bootstrap_pairs_models_and_resumes_exactly(tmp_path):
    y, t, scores, keys = _fixture()
    policies = {name: RankedPolicy.prepare(y, t, scores + i * t, keys)
                for i, name in enumerate(POLICIES)}
    first = tmp_path / "first"
    first.mkdir()
    partial = _bootstrap(policies, 4, 2027, first)
    resumed = _bootstrap(policies, 9, 2027, first)
    second = tmp_path / "second"
    second.mkdir()
    fresh = _bootstrap(policies, 9, 2027, second)
    np.testing.assert_array_equal(partial, resumed[:4])
    np.testing.assert_array_equal(resumed, fresh)
    assert fresh.shape == (9, len(POLICIES), len(METRIC_NAMES))
    # The random comparator is calculated from the same sampled rows each time.
    np.testing.assert_array_equal(fresh[:, 0, 1], fresh[:, 1, 1])
    np.testing.assert_array_equal(fresh[:, 0, 1], fresh[:, 2, 1])


def test_stratified_split_reuses_every_row_and_is_reproducible(tmp_path):
    y, t, _, _ = _fixture()
    labels = np.column_stack([t, y, y, t]).astype(np.uint8)
    train, test = _split(labels, tmp_path, 2026)
    assert len(test) == 120 and len(train) == 280
    np.testing.assert_array_equal(np.sort(np.r_[train, test]), np.arange(len(y)))
    train2, test2 = _split(labels, tmp_path, 2026)
    np.testing.assert_array_equal(train, train2)
    np.testing.assert_array_equal(test, test2)
    full = np.bincount(2 * t + y, minlength=4)
    held = np.bincount(2 * labels[test, 0] + labels[test, 1], minlength=4)
    np.testing.assert_allclose(held, full * 0.3, atol=1)


def test_x_learner_formula_and_checkpoint_predictions(tmp_path):
    # Small developer-only fixture checks model wiring, never produces project results.
    import lightgbm as lgb
    rng = np.random.default_rng(4)
    features = rng.normal(size=(400, 12)).astype(np.float32)
    y, t, _, _ = _fixture()
    labels = np.column_stack([t, y, y, t]).astype(np.uint8)
    train, test = _split(labels, tmp_path, 2026)
    parameters = {**FIXED_PARAMETERS, "n_estimators": 3, "min_child_samples": 10,
                  "n_jobs": 1, "random_state": 2026}
    preds, g = _policy_predictions(features, labels, train, test, tmp_path, parameters, 1)
    tau0 = np.load(tmp_path / "effect0_test.npy")
    tau1 = np.load(tmp_path / "effect1_test.npy")
    np.testing.assert_array_equal(preds["x_learner"], g * tau0 + (1 - g) * tau1)
    np.testing.assert_array_equal(preds["response"], np.load(tmp_path / "mu1_test.npy"))
    np.testing.assert_array_equal(preds["t_learner"],
                                  np.load(tmp_path / "mu1_test.npy") - np.load(tmp_path / "mu0_test.npy"))
    again, g_again = _policy_predictions(features, labels, train, test, tmp_path, parameters, 1)
    assert g == g_again
    np.testing.assert_array_equal(preds["x_learner"], again["x_learner"])
    assert lgb.Booster(model_file=str(tmp_path / "mu1.txt")).params["objective"] == "binary"
