"""Aggregate-count inference for the complete released Criteo benchmark.

These calculations do not recover the original experiment's population effects.
The public data underwent nonuniform sub-sampling, so even precise
intervals describe the released-data model and do not remove selection bias.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import norm

Z_975 = float(norm.ppf(0.975))
OUTCOMES = ("visit", "conversion")
FEATURES = tuple(f"f{k}" for k in range(12))


def _integer(value: Any, name: str) -> int:
    try:
        result = int(value)
        if not math.isfinite(float(value)) or float(value) != result:
            raise ValueError
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite integer") from exc
    return result


def _validate_groups(groups: list[dict]) -> dict[int, dict[str, int]]:
    """Validate sufficient statistics, including each binary joint count."""
    if len(groups) != 2:
        raise ValueError("Expected exactly two treatment groups")
    result = {}
    for group in groups:
        t = _integer(group["treatment"], "treatment")
        if t not in (0, 1) or t in result:
            raise ValueError("Expected one control group (0) and one treatment group (1)")
        n = _integer(group["n"], "n")
        if n <= 0:
            raise ValueError("Both treatment groups need a positive sample size")
        clean = {"treatment": t, "n": n}
        for field in ("visit_sum", "conversion_sum", "exposure_sum"):
            count = _integer(group[field], field)
            if not 0 <= count <= n:
                raise ValueError(f"{field} must lie between zero and n")
            clean[field] = count
        for outcome in OUTCOMES:
            field = f"{outcome}_exposure_sum"
            joint = _integer(group[field], field)
            y, d = clean[f"{outcome}_sum"], clean["exposure_sum"]
            if not max(0, y + d - n) <= joint <= min(y, d):
                raise ValueError(f"{field} is inconsistent with the marginal counts")
            clean[field] = joint
        result[t] = clean
    return result


def _frequentist(n0: int, n1: int, y0: int, y1: int) -> dict:
    p0, p1 = y0 / n0, y1 / n1
    difference = p1 - p0
    variance0, variance1 = p0 * (1 - p0) / n0, p1 * (1 - p1) / n1
    se = math.sqrt(variance0 + variance1)
    hc2_se = (
        math.sqrt(p0 * (1 - p0) / (n0 - 1) + p1 * (1 - p1) / (n1 - 1))
        if min(n0, n1) > 1 else None
    )
    pooled = (y0 + y1) / (n0 + n1)
    pooled_se = math.sqrt(pooled * (1 - pooled) * (1 / n0 + 1 / n1))
    z = difference / pooled_se if pooled_se > 0 else 0.0
    lift = difference / p0 if p0 > 0 else None
    lift_se = (
        math.sqrt(variance1 / p0**2 + p1**2 * variance0 / p0**4)
        if p0 > 0
        else None
    )
    warnings = []
    if min(y0, n0 - y0, y1, n1 - y1) < 10:
        warnings.append(
            "Normal approximations can be poor with few events or non-events; "
            "Wald intervals can collapse at a boundary."
        )
    if p0 == 0:
        warnings.append("Relative lift and its delta-method interval are undefined with zero control events.")
    mde = (
        (Z_975 + float(norm.ppf(0.8)))
        * math.sqrt(p0 * (1 - p0) * (1 / n0 + 1 / n1))
        if 0 < p0 < 1
        else None
    )
    return {
        "control_n": n0,
        "treatment_n": n1,
        "control_events": y0,
        "treatment_events": y1,
        "control_rate": p0,
        "treatment_rate": p1,
        "difference": difference,
        "standard_error": se,
        "standard_error_method": "Conventional unpooled Bernoulli plug-in Wald standard error.",
        "hc2_standard_error": hc2_se,
        "hc2_ci95": ([difference - Z_975 * hc2_se, difference + Z_975 * hc2_se]
                     if hc2_se is not None else None),
        "hc2_interpretation": "HC2 from Y ~ 1 + T equals s1^2/n1 + s0^2/n0; use this SE when comparing with Lin adjustment.",
        "wald_ci95": [difference - Z_975 * se, difference + Z_975 * se],
        "pooled_z": z,
        "two_sided_p_value": float(2 * norm.sf(abs(z))),
        "relative_lift": lift,
        "relative_lift_standard_error": lift_se,
        "relative_lift_ci95": (
            [lift - Z_975 * lift_se, lift + Z_975 * lift_se]
            if lift is not None and lift_se is not None
            else None
        ),
        "approximate_mde_80pct_absolute": mde,
        "mde_assumptions": "Two-sided alpha=0.05, target power=0.80, local normal approximation at the released control rate.",
        "warnings": warnings,
    }


def _posterior(
    n0: int, n1: int, y0: int, y1: int, prior: float,
    draws: int, rng: np.random.Generator,
) -> dict:
    p0 = rng.beta(y0 + prior, n0 - y0 + prior, size=draws)
    p1 = rng.beta(y1 + prior, n1 - y1 + prior, size=draws)
    difference = p1 - p0
    positive = float(np.mean(difference > 0))
    return {
        "prior_alpha": prior,
        "prior_beta": prior,
        "posterior_mean_difference": (
            (y1 + prior) / (n1 + 2 * prior) - (y0 + prior) / (n0 + 2 * prior)
        ),
        "credible_interval95": np.quantile(difference, [0.025, 0.975]).tolist(),
        "probability_positive": positive,
        "probability_positive_mc_standard_error": math.sqrt(positive * (1 - positive) / draws),
        "expected_loss_treatment": float(np.maximum(-difference, 0).mean()),
        "expected_loss_control": float(np.maximum(difference, 0).mean()),
        "draws": draws,
        "loss_units": "Outcome probability per user; no monetary cost or value is assumed.",
    }


def _iv_analysis(groups: dict[int, dict[str, int]]) -> dict:
    g0, g1 = groups[0], groups[1]
    n0, n1 = g0["n"], g1["n"]
    d0, d1 = g0["exposure_sum"] / n0, g1["exposure_sum"] / n1
    first_stage = d1 - d0
    result = {
        "status": "ok",
        "effect_label": "CACE / ATT" if g0["exposure_sum"] == 0 else "LATE",
        "one_sided_noncompliance": g0["exposure_sum"] == 0,
        "interpretation": (
            "The Wald ratio describes this released sample under the listed IV assumptions; "
            "it does not estimate Criteo's actual advertising effect. "
            + (
                "Under one-sided noncompliance, all observed exposed users are compliers. "
                "With random assignment, they are representative of the complier population, "
                "so the CACE also equals the average effect of exposure on the exposed (ATT), "
                "subject to the assumptions and sampling caveat above."
                if g0["exposure_sum"] == 0 else
                "Control users have nonzero recorded exposure. With monotonicity, "
                "the ratio has a local average treatment effect (LATE) interpretation; "
                "no ATT interpretation is made."
            )
        ),
        "required_assumptions": [
            "Random assignment within the released sample, with positivity.",
            "Exclusion: assignment affects the outcome only through exposure.",
            "Relevance: treatment assignment increases exposure probability.",
            (
                "One-sided noncompliance: no control user is exposed."
                if g0["exposure_sum"] == 0 else
                "Monotonicity: no user's exposure is reduced by assignment to treatment."
            ),
            "SUTVA: no interference between users and a single version of exposure.",
        ],
        "control_exposure_rate": d0,
        "treatment_exposure_rate": d1,
        "first_stage": first_stage,
        "first_stage_standard_error": None,
        "first_stage_f_statistic": None,
        "weak_first_stage": None,
        "outcomes": {},
        "warnings": [
            "Covariate balance cannot prove randomization, exclusion, or valid IV identification.",
            "The estimate inherits any error in recorded exposure and the public-data sampling caveat.",
        ],
    }
    if min(n0, n1) < 2:
        result["status"] = "skipped_insufficient_sample"
        result["warnings"].append("At least two observations per arm are required for robust ratio uncertainty.")
        return result
    var_d = d1 * (1 - d1) / (n1 - 1) + d0 * (1 - d0) / (n0 - 1)
    result["first_stage_standard_error"] = math.sqrt(var_d)
    f_stat = first_stage**2 / var_d if var_d > 0 else None
    result["first_stage_f_statistic"] = f_stat
    result["weak_first_stage"] = bool(first_stage <= 0 or (f_stat is not None and f_stat < 10))
    if first_stage <= 0:
        result["status"] = "skipped_zero_or_negative_first_stage"
        result["warnings"].append("The first stage is not positive; a Wald ratio is not reported.")
        return result
    if result["weak_first_stage"]:
        result["warnings"].append(
            "First-stage F is below the heuristic threshold 10; ratio and normal interval can be unstable."
        )
    if var_d == 0:
        result["warnings"].append("Exposure is deterministic within each arm, so the first-stage F statistic is undefined.")
    for outcome in OUTCOMES:
        p0, p1 = g0[f"{outcome}_sum"] / n0, g1[f"{outcome}_sum"] / n1
        numerator = p1 - p0
        ratio = numerator / first_stage
        var_y = p1 * (1 - p1) / (n1 - 1) + p0 * (1 - p0) / (n0 - 1)
        cov_y_d = (
            (g1[f"{outcome}_exposure_sum"] / n1 - p1 * d1) / (n1 - 1)
            + (g0[f"{outcome}_exposure_sum"] / n0 - p0 * d0) / (n0 - 1)
        )
        ratio_var = max(0.0, (var_y + ratio**2 * var_d - 2 * ratio * cov_y_d) / first_stage**2)
        ratio_se = math.sqrt(ratio_var)
        exposed_n = g1["exposure_sum"]
        unexposed_n = n1 - exposed_n
        exposed_y = g1[f"{outcome}_exposure_sum"]
        naive = (
            exposed_y / exposed_n - (g1[f"{outcome}_sum"] - exposed_y) / unexposed_n
            if exposed_n > 0 and unexposed_n > 0
            else None
        )
        result["outcomes"][outcome] = {
            "wald_ratio": ratio,
            "robust_delta_standard_error": ratio_se,
            "normal_ci95": [ratio - Z_975 * ratio_se, ratio + Z_975 * ratio_se],
            "numerator_denominator_covariance": cov_y_d,
            "naive_exposed_minus_unexposed_within_treatment": naive,
            "naive_comparison_interpretation": "A descriptive association subject to exposure selection bias.",
        }
    return result


def analyze_groups(groups: list[dict], seed: int = 2026, posterior_draws: int = 100_000) -> dict:
    """Analyze aggregate counts without loading the full CSV into memory.

    Each arm requires treatment, n, visit_sum, conversion_sum, exposure_sum,
    visit_exposure_sum and conversion_exposure_sum. All intervals use probability
    units: multiply differences by 100 to express percentage points.
    """
    arms = _validate_groups(groups)
    draws = _integer(posterior_draws, "posterior_draws")
    if draws < 2:
        raise ValueError("posterior_draws must be at least two")
    rng = np.random.default_rng(seed)
    output = {
        "estimand": "Released-benchmark treatment-group outcome-rate difference.",
        "interval_scope": "IID binomial working model conditional on the released sample; sampling/selection bias is excluded.",
        "seed": int(seed),
        "outcomes": {},
        "limitations": [
            "Nonuniform sub-sampling prevents recovering Criteo's original incrementality level; effects describe the released sample.",
            "All intervals are unadjusted for multiple outcomes; visit is primary and conversion is exploratory.",
            "Rare conversion alone does not imply inadequate power; sample size, effect size and the decision threshold also matter.",
            "No independent historical data were available for an informative prior.",
        ],
    }
    for outcome in OUTCOMES:
        n0, n1 = arms[0]["n"], arms[1]["n"]
        y0, y1 = arms[0][f"{outcome}_sum"], arms[1][f"{outcome}_sum"]
        output["outcomes"][outcome] = {
            "frequentist": _frequentist(n0, n1, y0, y1),
            "bayesian": {
                "jeffreys": _posterior(n0, n1, y0, y1, 0.5, draws, rng),
                "uniform": _posterior(n0, n1, y0, y1, 1.0, draws, rng),
            },
        }
    output["iv_exploratory"] = _iv_analysis(arms)
    return output

