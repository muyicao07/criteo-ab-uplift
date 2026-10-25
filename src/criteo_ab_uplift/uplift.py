"""Full-data, prespecified T-/X-learners and paired test-row bootstrap.

Training inputs and checkpoints stay in the ignored cache directory. Bootstrap
weights are exact multinomial row multiplicities, never Poisson approximations.
The weighted Qini implementation is algebraically identical to scikit-uplift's
``negative_effect=True`` metric, including aggregation of equal predictions.
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
POLICIES = ("x_learner", "t_learner", "response")
BUDGETS = (0.1, 0.3, 0.5)
CHUNK_ROWS = 250_000
FIXED_PARAMETERS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 2000,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "force_col_wise": True,
    "deterministic": True,
    "verbosity": -1,
}
METRIC_NAMES = ("qini", "random_per_1000", "u_10", "u_30", "u_50", "gain_10", "gain_30", "gain_50")
ALGORITHM_VERSION = 1


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_save(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values)
    os.replace(temporary, path)


def _area(x: np.ndarray, y: np.ndarray) -> float:
    # Explicit trapezoids work with both NumPy 1.x and NumPy 2.x.
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) * 0.5))


def _perfect_qini_area(n: int, n1: int, y1: int, y0: int) -> tuple[float, float]:
    """Perfect-minus-random area and baseline, from four contingency counts.

    scikit-uplift's perfect ordering is treated responders, all nonresponders,
    then control responders. Empty groups add zero-width trapezoids only.
    """
    n0 = n - n1
    if not n0 or not n1:
        return float("nan"), float("nan")
    end = y1 - y0 * n1 / n0
    x = np.asarray([0, y1, n - y0, n], dtype=np.float64)
    y = np.asarray([0, y1, y1, end], dtype=np.float64)
    baseline = 0.5 * n * end
    return _area(x, y) - baseline, baseline


@dataclass
class RankedPolicy:
    """One stable rank order, with original score ties retained for Qini."""

    order: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray
    group_ends: np.ndarray

    @classmethod
    def prepare(cls, outcome: np.ndarray, treatment: np.ndarray,
                scores: np.ndarray, tie_keys: np.ndarray) -> "RankedPolicy":
        if not (len(scores) == len(outcome) == len(treatment) == len(tie_keys)):
            raise ValueError("Scores, labels and tie keys must have the same length.")
        if not len(scores) or not np.isfinite(scores).all():
            raise ValueError("Policy scores must be finite and nonempty.")
        # Unique random integer keys break ties without perturbing model scores.
        order = np.lexsort((tie_keys, -scores))
        if len(order) < np.iinfo(np.int32).max:
            order = order.astype(np.int32)
        ordered_scores = scores[order]
        ends = np.r_[np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]), len(order) - 1]
        return cls(order, np.asarray(treatment[order], dtype=np.int8),
                   np.asarray(outcome[order], dtype=np.int8), ends)

    def evaluate(self, counts: np.ndarray | None = None,
                 budgets: tuple[float, ...] = BUDGETS,
                 return_curve: bool = False) -> tuple[dict, tuple | None]:
        """Evaluate an exact expanded sample without constructing its rows.

        A repeated boundary row is split to include exactly floor(k*n) copies.
        Qini uses the whole equal-score group, matching scikit-uplift; random
        tie keys affect targeting membership only, never the Qini definition.
        """
        weights = np.ones(len(self.order), dtype=np.int32) if counts is None else counts[self.order]
        if np.any(weights < 0):
            raise ValueError("Bootstrap multiplicities cannot be negative.")
        cumulative_n = np.cumsum(weights, dtype=np.int64)
        cumulative_t = np.cumsum(weights * self.treatment, dtype=np.int64)
        cumulative_y1 = np.cumsum(weights * self.treatment * self.outcome, dtype=np.int64)
        cumulative_y0 = np.cumsum(weights * (1 - self.treatment) * self.outcome, dtype=np.int64)
        n = int(cumulative_n[-1])
        n1 = int(cumulative_t[-1])
        y1, y0 = int(cumulative_y1[-1]), int(cumulative_y0[-1])
        if n == 0:
            raise ValueError("The resample must contain at least one row.")
        random = 1000 * (y1 / n1 - y0 / (n - n1)) if 0 < n1 < n else float("nan")
        result = {"n": n, "n_treatment": n1, "n_control": n - n1,
                  "random_per_1000": float(random), "targeting": []}
        for k in budgets:
            size = int(np.floor(k * n))
            if size <= 0 or size > n:
                raise ValueError("Each targeting budget must select between one and n rows.")
            boundary = int(np.searchsorted(cumulative_n, size, side="left"))
            excess = int(cumulative_n[boundary] - size)
            t, y = int(self.treatment[boundary]), int(self.outcome[boundary])
            selected_t = int(cumulative_t[boundary]) - excess * t
            selected_c = size - selected_t
            selected_y1 = int(cumulative_y1[boundary]) - excess * t * y
            selected_y0 = int(cumulative_y0[boundary]) - excess * (1 - t) * y
            u = (1000 * (selected_y1 / selected_t - selected_y0 / selected_c)
                 if selected_t and selected_c else float("nan"))
            result["targeting"].append({
                "k": k, "n_selected": size, "n_treatment": selected_t,
                "n_control": selected_c, "visit_treatment": selected_y1,
                "visit_control": selected_y0, "incremental_visits_per_1000": float(u),
                "gain_vs_random": float(u - random),
            })

        ends = self.group_ends
        nx = cumulative_n[ends]
        nt = cumulative_t[ends]
        nc = nx - nt
        q = cumulative_y1[ends].astype(np.float64)
        q -= cumulative_y0[ends] * np.divide(nt, nc, out=np.zeros(len(nt), dtype=np.float64), where=nc != 0)
        # Zero-weight score groups can be retained: their trapezoids have width 0.
        x = np.r_[0, nx]
        q = np.r_[0.0, q]
        perfect_area, baseline_area = _perfect_qini_area(n, n1, y1, y0)
        result["qini_coefficient"] = (float((_area(x, q) - baseline_area) / perfect_area)
                                      if perfect_area > 0 else float("nan"))
        curve = (x, q) if return_curve else None
        return result, curve


def _metric_vector(result: dict) -> np.ndarray:
    return np.asarray([result["qini_coefficient"], result["random_per_1000"]]
                      + [v["incremental_visits_per_1000"] for v in result["targeting"]]
                      + [v["gain_vs_random"] for v in result["targeting"]])


def _interval(values: np.ndarray) -> list[float]:
    if not np.isfinite(values).all():
        raise RuntimeError("An evaluation draw had an undefined metric; inspect treatment/control counts.")
    return [float(v) for v in np.quantile(values, [0.025, 0.975])]


def _predict_to_file(model, features: np.ndarray, indices: np.ndarray,
                     destination: Path, threads: int) -> np.ndarray:
    if destination.exists():
        existing = np.load(destination, mmap_mode="r")
        if existing.shape == (len(indices),):
            return existing
    temporary = destination.with_name(destination.name + ".tmp")
    values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float64, shape=(len(indices),))
    for start in range(0, len(indices), CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, len(indices))
        values[start:stop] = model.predict(features[indices[start:stop]], num_threads=threads)
    values.flush()
    del values
    os.replace(temporary, destination)
    return np.load(destination, mmap_mode="r")


def _fit_or_load(features: np.ndarray, indices: np.ndarray, target: np.ndarray,
                 directory: Path, name: str, objective: str, parameters: dict):
    import lightgbm as lgb

    path = directory / f"{name}.txt"
    if path.exists():
        LOGGER.info("Reusing completed model %s", name)
        return lgb.Booster(model_file=str(path))
    LOGGER.info("Fitting %s on %s rows (all rows in this training arm)", name, f"{len(indices):,}")
    # One arm is resident at a time. The full X.npy stays disk-backed.
    x_arm = np.ascontiguousarray(features[indices], dtype=np.float32)
    model_type = lgb.LGBMClassifier if objective == "binary" else lgb.LGBMRegressor
    model = model_type(objective=objective, **parameters)
    model.fit(x_arm, np.asarray(target))
    booster = model.booster_
    temporary = path.with_name(path.name + ".tmp")
    booster.save_model(str(temporary))
    os.replace(temporary, path)
    del model, x_arm
    gc.collect()
    return booster


def _ensure_identity(directory: Path, features_path: Path, labels_path: Path,
                     parameters: dict, seed: int) -> dict:
    import lightgbm as lgb
    import sklearn

    signature = {
        "algorithm_version": ALGORITHM_VERSION,
        "seed": seed,
        "test_size": 0.3,
        "parameters": parameters,
        "lightgbm": lgb.__version__, "sklearn": sklearn.__version__,
        "input_files": {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                        for p in (features_path, labels_path)},
    }
    identity_path = directory / "identity.json"
    if identity_path.exists():
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
        if previous != signature:
            raise RuntimeError("Uplift checkpoint configuration or input changed. "
                               "Move/remove cache/uplift before starting a new configuration.")
    else:
        _write_json(identity_path, signature)
    return signature


def _split(labels: np.ndarray, directory: Path, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    train_path, test_path = directory / "train_indices.npy", directory / "test_indices.npy"
    if not (train_path.exists() and test_path.exists()):
        LOGGER.info("Creating full-data 70/30 split, stratified by (treatment, visit)")
        strata = labels[:, 0] * 2 + labels[:, 1]
        indices = np.arange(len(labels), dtype=np.int32)
        train, test = train_test_split(indices, test_size=0.3, random_state=seed, stratify=strata)
        _atomic_save(train_path, train)
        _atomic_save(test_path, test)
        del indices, strata, train, test
        gc.collect()
    return np.load(train_path, mmap_mode="r"), np.load(test_path, mmap_mode="r")


def _policy_predictions(features: np.ndarray, labels: np.ndarray, train: np.ndarray,
                        test: np.ndarray, directory: Path, parameters: dict,
                        threads: int) -> tuple[dict[str, np.ndarray], float]:
    predictions = {name: directory / f"{name}_test.npy" for name in POLICIES}
    train_treatment = labels[train, 0]
    g = float(np.mean(train_treatment, dtype=np.float64))
    if all(p.exists() for p in predictions.values()):
        return {name: np.load(path, mmap_mode="r") for name, path in predictions.items()}, g
    control = np.asarray(train[train_treatment == 0])
    treated = np.asarray(train[train_treatment == 1])
    del train_treatment

    # Fit sequentially; mu_0 predicts the treated pseudo-outcome and vice versa.
    mu0 = _fit_or_load(features, control, labels[control, 1], directory, "mu0", "binary", parameters)
    mu0_test = _predict_to_file(mu0, features, test, directory / "mu0_test.npy", threads)
    mu0_treated = _predict_to_file(mu0, features, treated, directory / "mu0_treated.npy", threads)
    del mu0
    gc.collect()
    mu1 = _fit_or_load(features, treated, labels[treated, 1], directory, "mu1", "binary", parameters)
    mu1_test = _predict_to_file(mu1, features, test, directory / "mu1_test.npy", threads)
    mu1_control = _predict_to_file(mu1, features, control, directory / "mu1_control.npy", threads)
    del mu1
    gc.collect()

    pseudo0 = np.asarray(mu1_control - labels[control, 1], dtype=np.float32)
    effect0 = _fit_or_load(features, control, pseudo0, directory, "effect0", "regression_l2", parameters)
    del pseudo0
    effect0_test = _predict_to_file(effect0, features, test, directory / "effect0_test.npy", threads)
    del effect0
    gc.collect()
    pseudo1 = np.asarray(labels[treated, 1] - mu0_treated, dtype=np.float32)
    effect1 = _fit_or_load(features, treated, pseudo1, directory, "effect1", "regression_l2", parameters)
    del pseudo1
    effect1_test = _predict_to_file(effect1, features, test, directory / "effect1_test.npy", threads)
    del effect1, treated, control, mu0_treated, mu1_control
    gc.collect()

    # Künzel et al.: g(x)*tau_0(x) + (1-g(x))*tau_1(x).
    for name in POLICIES:
        if predictions[name].exists():
            continue
        temporary = predictions[name].with_name(predictions[name].name + ".tmp")
        values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float64, shape=(len(test),))
        for start in range(0, len(test), CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, len(test))
            if name == "x_learner":
                values[start:stop] = g * effect0_test[start:stop] + (1 - g) * effect1_test[start:stop]
            elif name == "t_learner":
                values[start:stop] = mu1_test[start:stop] - mu0_test[start:stop]
            else:
                values[start:stop] = mu1_test[start:stop]
        values.flush()
        del values
        os.replace(temporary, predictions[name])
    return {name: np.load(path, mmap_mode="r") for name, path in predictions.items()}, g


def _bootstrap(policies: dict[str, RankedPolicy], repetitions: int, seed: int,
               directory: Path) -> np.ndarray:
    n = len(next(iter(policies.values())).order)
    checkpoint = directory / "bootstrap_checkpoint.npz"
    draws = np.empty((repetitions, len(POLICIES), len(METRIC_NAMES)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    start = 0
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            saved_draws = saved["draws"]
            if int(saved["seed"]) != seed or int(saved["n"]) != n:
                raise RuntimeError("Bootstrap checkpoint seed or test size differs.")
            if saved_draws.shape[0] <= repetitions:
                start = len(saved_draws)
                draws[:start] = saved_draws
                rng.bit_generator.state = json.loads(str(saved["rng_state"]))
                LOGGER.info("Resuming paired bootstrap at draw %d/%d", start, repetitions)
    for b in range(start, repetitions):
        # Exactly n iid row indices implies Multinomial(n; 1/n, ..., 1/n).
        sampled_rows = rng.integers(0, n, size=n, dtype=np.int32)
        counts = np.bincount(sampled_rows, minlength=n).astype(np.int32)
        del sampled_rows
        for j, name in enumerate(POLICIES):
            result, _ = policies[name].evaluate(counts)
            draws[b, j] = _metric_vector(result)
        if (b + 1) % 20 == 0 or b + 1 == repetitions:
            temporary = checkpoint.with_name(checkpoint.name + ".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, draws=draws[:b + 1], seed=seed, n=n,
                                    rng_state=json.dumps(rng.bit_generator.state))
            os.replace(temporary, checkpoint)
            LOGGER.info("Completed paired test-row bootstrap %d/%d", b + 1, repetitions)
    return draws


def run_uplift(cache_dir: Path, output_dir: Path, seed: int = 2026,
               threads: int = 4, bootstrap_reps: int = 200) -> dict:
    """Fit all full-data learners and write prespecified visit-only evaluation."""
    from sklift.metrics import qini_auc_score

    started = time.monotonic()
    cache_dir, output_dir = Path(cache_dir), Path(output_dir)
    directory = cache_dir / "uplift"
    directory.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path, labels_path = cache_dir / "X.npy", cache_dir / "labels.npy"
    features = np.load(features_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    if features.dtype != np.float32 or features.shape != (len(labels), 12) or labels.shape[1] != 4:
        raise ValueError("Expected float32 X.npy (N,12) and labels.npy (N,4): T,visit,conversion,exposure.")
    if bootstrap_reps <= 0:
        raise ValueError("At least one bootstrap repetition is required.")
    parameters = {**FIXED_PARAMETERS, "random_state": seed, "n_jobs": threads}
    identity = _ensure_identity(directory, features_path, labels_path, parameters, seed)
    train, test = _split(labels, directory, seed)
    predictions, g = _policy_predictions(features, labels, train, test, directory, parameters, threads)
    outcome, treatment = labels[test, 1].astype(np.int8), labels[test, 0].astype(np.int8)
    # Integer permutation gives unique fixed-seed tie keys, shared by all models.
    tie_keys = np.random.default_rng(seed).permutation(len(test)).astype(np.int32)
    policies, points, curve_records = {}, {}, []
    for name in POLICIES:
        LOGGER.info("Evaluating %s, retaining exact score ties for Qini", name)
        policy = RankedPolicy.prepare(outcome, treatment, predictions[name], tie_keys)
        point, curve = policy.evaluate(return_curve=True)
        # The published point coefficient comes directly from scikit-uplift.
        # Signed labels avoid uint8 subtraction underflow in its perfect curve.
        reference = float(qini_auc_score(outcome, predictions[name], treatment, negative_effect=True))
        if not np.isclose(point["qini_coefficient"], reference, rtol=1e-10, atol=1e-12):
            raise RuntimeError("Weighted Qini disagrees with scikit-uplift.")
        point["qini_coefficient"] = reference
        policies[name], points[name] = policy, point
        x, q = curve
        # Display points only are compressed; Qini areas above use every score.
        keep = np.unique(np.rint(np.linspace(0, len(x) - 1, min(len(x), 2001))).astype(int))
        for i in keep:
            curve_records.append({"model": name, "fraction_targeted": float(x[i] / len(test)),
                                  "incremental_visits": float(q[i]),
                                  "qini_baseline": float(x[i] / len(test) * q[-1])})
        del x, q, curve
        gc.collect()
    del tie_keys, predictions
    draws = _bootstrap(policies, bootstrap_reps, seed + 1, directory)
    targeting_records, qini_records = [], []
    for j, name in enumerate(POLICIES):
        p = points[name]
        qini_ci, random_ci = _interval(draws[:, j, 0]), _interval(draws[:, j, 1])
        p["qini_ci"] = qini_ci
        p["random_ci"] = random_ci
        qini_records.append({"model": name, "qini_coefficient": p["qini_coefficient"],
                             "ci_lower": qini_ci[0], "ci_upper": qini_ci[1]})
        for z, row in enumerate(p["targeting"]):
            u_ci, gain_ci = _interval(draws[:, j, z + 2]), _interval(draws[:, j, z + 5])
            row["ci"] = u_ci
            row["gain_ci"] = gain_ci
            targeting_records.append({"model": name, **{k: v for k, v in row.items() if k not in ("ci", "gain_ci")},
                                      "ci_lower": u_ci[0], "ci_upper": u_ci[1],
                                      "random_per_1000": p["random_per_1000"],
                                      "random_ci_lower": random_ci[0], "random_ci_upper": random_ci[1],
                                      "gain_ci_lower": gain_ci[0], "gain_ci_upper": gain_ci[1]})
    pd.DataFrame(targeting_records).to_csv(output_dir / "uplift_targeting.csv", index=False)
    pd.DataFrame(qini_records).to_csv(output_dir / "uplift_qini.csv", index=False)
    pd.DataFrame(curve_records).to_csv(output_dir / "uplift_curves.csv", index=False)
    np.savez_compressed(output_dir / "uplift_bootstrap.npz", draws=draws,
                        models=np.asarray(POLICIES), metrics=np.asarray(METRIC_NAMES))
    split_counts = {}
    for split_name, indices in (("train", train), ("test", test)):
        strata = labels[indices, 0] * 2 + labels[indices, 1]
        split_counts[split_name] = {f"T={t},visit={y}": int(np.sum(strata == 2 * t + y))
                                    for t in (0, 1) for y in (0, 1)}
    result = {
        "primary_model": "x_learner", "outcome": "visit", "all_rows_used": True,
        "n_total": int(len(labels)), "n_train": int(len(train)), "n_test": int(len(test)),
        "split": {"test_fraction": 0.3, "stratification": ["treatment", "visit"],
                  "seed": seed, "counts": split_counts},
        "treatment_propensity_train": g, "parameters": parameters,
        "effect_objective": "regression_l2", "outcome_objective": "binary",
        "x_learner_formula": "g * tau_0 + (1-g) * tau_1",
        "bootstrap": {"repetitions": bootstrap_reps, "seed": seed + 1,
                      "method": "paired ordinary iid test-row bootstrap with exact integer multiplicities",
                      "ci": "95% percentile", "models_fixed": True,
                      "reselect_top_k_each_draw": True, "exact_floor_k_n": True,
                      "uncertainty_scope": "Test-sample uncertainty only; fitted models remain fixed."},
        "tie_break_seed": seed, "qini_definition": "sklift.metrics.qini_auc_score(negative_effect=True)",
        "unit": "incremental visits per 1,000 users selected by the policy and assigned to treatment "
                "(intention-to-treat, including users who were assigned but not exposed)",
        "models": points, "elapsed_seconds": time.monotonic() - started,
        "checkpoint_identity_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
    }
    _write_json(output_dir / "uplift_results.json", result)
    return result
