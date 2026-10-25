"""Full-data Lin adjustment with a two-pass, bounded-memory HC2 calculation.

Two separate arm regressions are an exact reparameterisation of an intercept,
treatment, centred covariates and all treatment-covariate interactions. The
contrast is evaluated at the full-sample covariate mean, treated as fixed.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
from scipy.stats import norm


def _inverse_for_contrast(gram: np.ndarray, contrast: np.ndarray) -> tuple[np.ndarray, dict]:
    """Use a rank-revealing symmetric inverse, without adding regularization."""
    gram = (gram + gram.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    tolerance = np.finfo("float64").eps * max(gram.shape) * float(eigenvalues[-1])
    if eigenvalues[0] < -100 * tolerance:
        raise ValueError("The accumulated regression Gram matrix is not positive semidefinite.")
    retained = eigenvalues > tolerance
    rank = int(retained.sum())
    if not rank:
        raise ValueError("The regression design has zero rank.")
    basis = eigenvectors[:, retained]
    if not np.allclose(basis @ (basis.T @ contrast), contrast, atol=1e-8, rtol=1e-8):
        raise ValueError("The adjusted outcome at the pooled covariate mean is not identifiable.")
    inverse = (basis / eigenvalues[retained]) @ basis.T
    condition = float(eigenvalues[-1] / eigenvalues[retained][0])
    note = None
    if rank < len(contrast):
        note = (f"Design rank is {rank}/{len(contrast)}. A Moore-Penrose inverse was used "
                "after checking the pooled-mean contrast is estimable; no ridge penalty was added.")
        warnings.warn(note, RuntimeWarning, stacklevel=2)
    elif condition > 1e10:
        note = f"The standardized Gram matrix has a large condition number ({condition:.3g})."
        warnings.warn(note, RuntimeWarning, stacklevel=2)
    return inverse, {"rank": rank, "columns": len(contrast), "condition_number": condition,
                     "rank_tolerance": tolerance, "numerical_note": note}


def lin_adjustment(cache_dir: Path, audit: dict, chunk_rows: int = 100_000) -> dict:
    """Return full-row visit/conversion Lin estimates and HC2 standard errors."""
    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer.")
    cache_dir = Path(cache_dir)
    features = np.load(cache_dir / "X64.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(cache_dir / "labels.npy", mmap_mode="r", allow_pickle=False)
    n = int(audit["n_total"])
    if features.shape != (n, 12) or labels.shape != (n, 4):
        raise ValueError("Regression cache dimensions do not match the full-data audit.")
    if features.dtype != np.dtype("float64"):
        raise ValueError("Lin adjustment requires the float64 feature cache.")
    means = np.asarray(audit["feature_means"], dtype="float64")
    stds = np.asarray(audit["feature_stds"], dtype="float64")
    if means.shape != (12,) or stds.shape != (12,) or not np.isfinite(means).all() or not np.isfinite(stds).all():
        raise ValueError("Invalid full-sample feature means or standard deviations.")
    scales = np.where(stds > 0, stds, 1.0)
    grams = np.zeros((2, 13, 13), dtype="float64")
    rhs = np.zeros((2, 13, 2), dtype="float64")
    counts = np.zeros(2, dtype="int64")
    contrast = np.zeros(13, dtype="float64")
    contrast[0] = 1

    def blocks():
        for start in range(0, n, chunk_rows):
            end = min(start + chunk_rows, n)
            z = np.empty((end - start, 13), dtype="float64")
            z[:, 0] = 1.0
            np.subtract(features[start:end], means, out=z[:, 1:])
            np.divide(z[:, 1:], scales, out=z[:, 1:])
            t = labels[start:end, 0]
            y = np.asarray(labels[start:end, 1:3], dtype="float64")
            yield z, t, y

    print("Lin adjustment: first full-data pass (normal equations for both outcomes)...", flush=True)
    for z, t, y in blocks():
        for arm in (0, 1):
            selected = t == arm
            za, ya = z[selected], y[selected]
            counts[arm] += len(za)
            grams[arm] += za.T @ za
            rhs[arm] += za.T @ ya
    expected_counts = np.array([int(next(group for group in audit["groups"]
                                          if int(group["treatment"]) == arm)["n"])
                                for arm in (0, 1)])
    if not np.array_equal(counts, expected_counts) or int(counts.sum()) != n:
        raise ValueError("Cached treatment counts disagree with the full-file SQL audit.")
    inverses, diagnostics, coefficients = [], [], []
    for arm in (0, 1):
        inverse, diagnostic = _inverse_for_contrast(grams[arm], contrast)
        if counts[arm] <= diagnostic["rank"]:
            raise ValueError("Each arm needs more observations than its regression rank for HC2.")
        inverses.append(inverse)
        diagnostics.append({"treatment": arm, **diagnostic})
        coefficients.append(inverse @ rhs[arm])
    variances = np.zeros((2, 2), dtype="float64")
    max_leverages = np.zeros(2, dtype="float64")
    rss = np.zeros((2, 2), dtype="float64")
    print("Lin adjustment: second full-data pass (HC2 residuals and leverage)...", flush=True)
    for z, t, y in blocks():
        for arm in (0, 1):
            selected = t == arm
            za, ya = z[selected], y[selected]
            if not len(za):
                continue
            residuals = ya - za @ coefficients[arm]
            leverage = np.einsum("ij,ij->i", za @ inverses[arm], za)
            if np.any(leverage < -1e-8) or np.any(leverage >= 1 - 1e-12):
                raise ValueError("HC2 is undefined or numerically unstable for a row with leverage near one.")
            leverage = np.maximum(leverage, 0.0)
            max_leverages[arm] = max(max_leverages[arm], float(leverage.max()))
            influence = za @ (inverses[arm] @ contrast)
            # c' (Z'Z)^-1 meat_HC2 (Z'Z)^-1 c, accumulated directly.
            weighted_square = influence**2 / (1.0 - leverage)
            variances[arm] += weighted_square @ (residuals**2)
            rss[arm] += np.sum(residuals**2, axis=0)
    z95 = float(norm.ppf(0.975))
    outcomes = {}
    for j, name in enumerate(("visit", "conversion")):
        g0, g1 = [next(group for group in audit["groups"] if int(group["treatment"]) == arm)
                  for arm in (0, 1)]
        p0, p1 = g0[f"{name}_sum"] / g0["n"], g1[f"{name}_sum"] / g1["n"]
        unadjusted = p1 - p0
        # Binary sample variance / n: p(1-p)/(n-1). This is intercept+T HC2.
        variance_unadjusted = p0 * (1 - p0) / (g0["n"] - 1) + p1 * (1 - p1) / (g1["n"] - 1)
        se_unadjusted = float(np.sqrt(variance_unadjusted))
        tau = float(coefficients[1][0, j] - coefficients[0][0, j])
        se = float(np.sqrt(variances[:, j].sum()))
        outcomes[name] = {
            "n": n, "tau_adjusted": tau, "se_adjusted": se,
            "tau_adj": tau, "se_adj": se,
            "ci_low": tau - z95 * se, "ci_high": tau + z95 * se,
            "tau_unadjusted": unadjusted, "se_unadjusted_hc2": se_unadjusted,
            "unadjusted_ci_low": unadjusted - z95 * se_unadjusted,
            "unadjusted_ci_high": unadjusted + z95 * se_unadjusted,
            "se_ratio": se / se_unadjusted if se_unadjusted > 0 else None,
            "adjusted_mean_control": float(coefficients[0][0, j]),
            "adjusted_mean_treated": float(coefficients[1][0, j]),
            "arm_hc2_variances": variances[:, j].tolist(),
            "arm_residual_sums_of_squares": rss[:, j].tolist(),
        }
    return {
        "n_total": n, "outcomes": outcomes,
        "method": "Lin fully interacted OLS, two-pass arm regressions, HC2 sandwich",
        "mean_handling": "Full-sample covariate mean treated as fixed",
        "unadjusted_method": "Intercept plus treatment OLS with HC2: s1^2/n1 + s0^2/n0",
        "feature_dtype": "float64", "accumulation_dtype": "float64",
        "chunk_rows": chunk_rows, "diagnostics": diagnostics,
        "max_leverage_by_arm": max_leverages.tolist(),
        "feature_means": means.tolist(), "feature_scales": scales.tolist(),
    }
