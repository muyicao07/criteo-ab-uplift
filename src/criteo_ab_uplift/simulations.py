"""Exact count simulations by sampling the full released finite population.

No artificial user outcomes are generated. Multivariate hypergeometric counts
are exactly equivalent to drawing released rows without replacement, including
joint visit/conversion labels when their eight treatment/outcome cell counts
are supplied. Ordinary Wald/Beta intervals are deliberately left uncorrected:
the experiment measures their coverage under this finite-population design.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import norm

from .statistics import OUTCOMES, Z_975, _integer, _validate_groups

FINITE_POPULATION_INTERPRETATION = (
    "Power and coverage are empirical rejection and coverage rates under repeated "
    "sampling without replacement from the full public sample, treated as a finite "
    "population whose difference in visit rates is the reference value."
)


def _p_values(y0: np.ndarray, y1: np.ndarray, n0: Any, n1: Any) -> np.ndarray:
    p0, p1 = y0 / n0, y1 / n1
    pooled = (y0 + y1) / (n0 + n1)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n0 + 1 / n1))
    z = np.divide(p1 - p0, se, out=np.zeros_like(pooled, dtype=float), where=se > 0)
    return 2 * norm.sf(np.abs(z))


def _rejection(y0: np.ndarray, y1: np.ndarray, n0: Any, n1: Any) -> np.ndarray:
    return _p_values(y0, y1, n0, n1) < 0.05


def _mc_summary(values: np.ndarray) -> dict:
    """Bernoulli Monte Carlo error; Wilson remains informative at 0 and 1."""
    n = int(values.size)
    p = float(values.mean())
    denominator = 1 + Z_975**2 / n
    center = (p + Z_975**2 / (2 * n)) / denominator
    half = Z_975 / denominator * math.sqrt(p * (1 - p) / n + Z_975**2 / (4 * n**2))
    return {
        "rate": p,
        "mc_standard_error": math.sqrt(p * (1 - p) / n),
        "mc_wilson_interval95": [max(0.0, center - half), min(1.0, center + half)],
        "repetitions": n,
    }


def _draw_cells(rng: np.random.Generator, population: np.ndarray, n: int, repetitions: int) -> np.ndarray:
    """Vectorized multivariate hypergeometric, allowing one pool per replicate.

    Sequential conditional hypergeometrics implement precisely the joint law.
    This also handles structural-zero cells without changing labels or counts.
    """
    population = np.asarray(population, dtype=np.int64)
    if population.ndim == 1:
        population = np.broadcast_to(population, (repetitions, population.size))
    if population.ndim != 2 or population.shape[0] != repetitions or np.any(population < 0):
        raise ValueError("Population counts must be nonnegative with one row per replication")
    remaining = population.sum(axis=1)
    if n < 0 or np.any(n > remaining):
        raise ValueError("Requested sample exceeds the remaining finite population")
    draws_left = np.full(repetitions, n, dtype=np.int64)
    sampled = np.zeros_like(population)
    for cell in range(population.shape[1] - 1):
        good = population[:, cell]
        bad = remaining - good
        active = (draws_left > 0) & (good > 0)
        # NumPy's hypergeometric API requires a nonempty good/bad total even
        # when zero draws remain, so zero/empty cases are handled explicitly.
        if np.any(active):
            sampled[active, cell] = rng.hypergeometric(good[active], bad[active], draws_left[active])
        draws_left -= sampled[:, cell]
        remaining -= good
    sampled[:, -1] = draws_left
    return sampled


def _population_blocks(arms: dict, joint_counts: list[dict] | None) -> list[dict]:
    """Validate joint cells, or state explicitly when only marginals are known."""
    if joint_counts is None:
        return [
            {
                "outcomes": (outcome,),
                "labels": np.array([[0], [1]], dtype=np.int64),
                "arms": {t: np.array([g["n"] - g[f"{outcome}_sum"], g[f"{outcome}_sum"]], dtype=np.int64)
                         for t, g in arms.items()},
            }
            for outcome in OUTCOMES
        ]
    cells = {0: np.zeros(4, dtype=np.int64), 1: np.zeros(4, dtype=np.int64)}
    seen = set()
    for record in joint_counts:
        t = _integer(record["treatment"], "treatment")
        visit = _integer(record["visit"], "visit")
        conversion = _integer(record["conversion"], "conversion")
        n = _integer(record["n"], "joint count")
        if t not in (0, 1) or visit not in (0, 1) or conversion not in (0, 1) or n < 0:
            raise ValueError("Joint cell labels must be binary and counts nonnegative")
        key = (t, visit, conversion)
        if key in seen:
            raise ValueError("Duplicate treatment/visit/conversion cell")
        seen.add(key)
        cells[t][2 * visit + conversion] = n
    labels = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    for t in (0, 1):
        if int(cells[t].sum()) != arms[t]["n"]:
            raise ValueError("Joint cells do not match the full treatment-group size")
        for outcome, count in zip(OUTCOMES, cells[t] @ labels):
            if int(count) != arms[t][f"{outcome}_sum"]:
                raise ValueError(f"Joint cells disagree with {outcome} marginal count")
    return [{"outcomes": OUTCOMES, "labels": labels, "arms": cells}]


def _analytic_power(
    p0: float, p1: float, n0: int, n1: int,
    population_n0: int | None = None, population_n1: int | None = None,
) -> float:
    """Normal approximation to the same pooled-z rejection rule.

    The critical threshold uses the pooled proportion under the alternative.
    An optional finite-population correction applies only to the distribution
    of the estimated alternative contrast, not the test's declared threshold.
    """
    pooled = (n0 * p0 + n1 * p1) / (n0 + n1)
    cutoff = Z_975 * math.sqrt(pooled * (1 - pooled) * (1 / n0 + 1 / n1))
    fpc0 = (population_n0 - n0) / (population_n0 - 1) if population_n0 is not None else 1.0
    fpc1 = (population_n1 - n1) / (population_n1 - 1) if population_n1 is not None else 1.0
    alternative_se = math.sqrt(fpc0 * p0 * (1 - p0) / n0 + fpc1 * p1 * (1 - p1) / n1)
    delta = p1 - p0
    if alternative_se == 0:
        return float(abs(delta) > cutoff)
    return float(norm.sf((cutoff - delta) / alternative_se) + norm.cdf((-cutoff - delta) / alternative_se))


def _posterior_bounds(
    counts0: np.ndarray, counts1: np.ndarray, n: int, prior: float,
    draws: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.empty(len(counts0)), np.empty(len(counts0))
    for start in range(0, len(counts0), 64):
        stop = min(len(counts0), start + 64)
        y0, y1 = counts0[start:stop, None], counts1[start:stop, None]
        p0 = rng.beta(y0 + prior, n - y0 + prior, size=(stop - start, draws))
        p1 = rng.beta(y1 + prior, n - y1 + prior, size=(stop - start, draws))
        intervals = np.quantile(p1 - p0, [0.025, 0.975], axis=1)
        lo[start:stop], hi[start:stop] = intervals
    return lo, hi


def run_simulations(
    groups: list[dict], seed: int = 2026, posterior_draws: int = 2_000, *,
    joint_counts: list[dict] | None = None, aa_reps: int = 1_000,
    power_reps: int = 500, peeking_reps: int = 1_000, coverage_reps: int = 1_000,
) -> dict:
    """Run the fixed full-data A/A, power, peeking and coverage design.

    Repetition keyword arguments exist for developer verification only. The
    project entry point uses the default replication counts. Sampling
    sizes and the 50 peeking looks never change, including in developer checks.
    """
    arms = _validate_groups(groups)
    repetitions = {}
    for name, value in (("aa", aa_reps), ("power", power_reps), ("peeking", peeking_reps), ("coverage", coverage_reps)):
        repetitions[name] = _integer(value, f"{name}_reps")
        if repetitions[name] < 2:
            raise ValueError("Each simulation requires at least two replications")
    posterior_draws = _integer(posterior_draws, "posterior_draws")
    if posterior_draws < 2:
        raise ValueError("posterior_draws must be at least two")
    blocks = _population_blocks(arms, joint_counts)
    n0_data, n1_data = arms[0]["n"], arms[1]["n"]
    if n0_data < 1_000_000:
        raise ValueError("The fixed peeking design requires at least 1,000,000 released control users")
    if min(n0_data, n1_data) < 10_000:
        raise ValueError("The fixed coverage design requires at least 10,000 released users in each arm")
    allocation = n1_data / (n0_data + n1_data)
    power_sizes = []
    for total in (10_000, 100_000, 1_000_000):
        n1 = min(total - 1, max(1, round(total * allocation)))
        n0 = total - n1
        if n0 > n0_data or n1 > n1_data:
            raise ValueError("Power sample sizes exceed the released finite population")
        power_sizes.append((total, n0, n1))
    # Independent reproducible streams mean posterior simulation does not alter
    # any row allocation, and both outcomes use the same actual sampled cells.
    seed_sequences = np.random.SeedSequence(seed).spawn(5)
    aa_rng, power_rng, peeking_rng, coverage_rng, posterior_rng = [np.random.default_rng(s) for s in seed_sequences]
    result = {
        "metadata": {
            "seed": int(seed),
            "repetitions": repetitions,
            "posterior_draws_per_replication": posterior_draws,
            "alpha": 0.05,
            "treatment_fraction": allocation,
            "control_population_n": n0_data,
            "treatment_population_n": n1_data,
            "sampling": "Exact multivariate hypergeometric draws without replacement from released-data cell counts.",
            "outcomes_jointly_sampled": joint_counts is not None,
            "outcome_pairing": (
                "Visit and conversion are paired on the same sampled rows, via their joint count cells."
                if joint_counts is not None else
                "Only marginal counts were supplied; each outcome is sampled separately. No cross-outcome paired comparison is claimed."
            ),
            "interpretation": FINITE_POPULATION_INTERPRETATION,
            "conversion_interpretation": "For secondary conversion analyses, the full public-sample conversion-rate difference is the analogous reference value.",
            "interval_design": "Wald and independent Beta posterior intervals use the ordinary binomial working model without finite-population correction; their empirical finite-population coverage is measured, not assumed.",
            "analytic_power_assumptions": "Two-sided pooled z-test, alpha=0.05, normal approximation at released arm rates. The IID baseline ignores finite-population correction; a second approximation applies it to the alternative variance. Rare-event approximations can be inaccurate.",
            "mde_assumptions": "Two-sided alpha=0.05, 80% target power, local normal approximation at the released control rate, IID baseline without finite-population correction.",
            "uncertainty": "Monte Carlo SE and Wilson intervals quantify replication error; posterior quantiles add numerical simulation error.",
            "aa_scope": "A/A checks the specified test under random splits of the fixed released control pool; it cannot establish randomisation of the original release.",
        },
        "aa_peeking": {},
        "power": {outcome: [] for outcome in OUTCOMES},
        "coverage": {outcome: [] for outcome in OUTCOMES},
    }
    for block in blocks:
        outcomes, labels, populations = block["outcomes"], block["labels"], block["arms"]
        control_cells, treatment_cells = populations[0], populations[1]
        control_events, treatment_events = control_cells @ labels, treatment_cells @ labels

        # Full control randomly partitioned once per replicate; the second
        # half is the complement, rather than an independent overlapping draw.
        half0, half1 = n0_data // 2, n0_data - n0_data // 2
        split0 = _draw_cells(aa_rng, control_cells, half0, repetitions["aa"])
        aa_y0 = split0 @ labels
        aa_y1 = (control_cells - split0) @ labels
        for column, outcome in enumerate(outcomes):
            p_values = _p_values(aa_y0[:, column], aa_y1[:, column], half0, half1)
            result["aa_peeking"][outcome] = {
                "control_population_n": n0_data,
                "control_population_rate": int(control_events[column]) / n0_data,
                "reference_difference": 0.0,
                "aa_control_n": half0,
                "aa_treatment_n": half1,
                "aa_false_positive": _mc_summary(p_values < 0.05),
                "aa_p_values": p_values.tolist(),
                "aa_mean_difference": float(np.mean(aa_y1[:, column] / half1 - aa_y0[:, column] / half0)),
                "aa_design": "The complete control pool is split into complementary random halves in every replication.",
                "total_n": 1_000_000,
                "control_n": 500_000,
                "treatment_n": 500_000,
                "looks": 50,
                "increment_per_arm": 10_000,
                "test": "Two-sided pooled z-test at alpha=0.05 without sequential correction.",
                "per_look": [],
            }

        # Each trajectory keeps its own finite pool. Every newly allocated
        # user is removed before allocating the other arm or the next look.
        remaining = np.broadcast_to(control_cells, (repetitions["peeking"], len(control_cells))).copy()
        total0 = np.zeros((repetitions["peeking"], len(outcomes)), dtype=np.int64)
        total1 = np.zeros_like(total0)
        ever = np.zeros_like(total0, dtype=bool)
        first = np.zeros_like(total0, dtype=np.int64)
        for look in range(1, 51):
            cells0 = _draw_cells(peeking_rng, remaining, 10_000, repetitions["peeking"])
            remaining -= cells0
            cells1 = _draw_cells(peeking_rng, remaining, 10_000, repetitions["peeking"])
            remaining -= cells1
            total0 += cells0 @ labels
            total1 += cells1 @ labels
            reject = _rejection(total0, total1, look * 10_000, look * 10_000)
            first[(first == 0) & reject] = look
            ever |= reject
            for column, outcome in enumerate(outcomes):
                result["aa_peeking"][outcome]["per_look"].append({
                    "look": look,
                    "per_arm_n": look * 10_000,
                    "fixed_look_false_positive": _mc_summary(reject[:, column]),
                    "ever_rejected": _mc_summary(ever[:, column]),
                })
        for column, outcome in enumerate(outcomes):
            result["aa_peeking"][outcome].update({
                "fixed_horizon_false_positive": _mc_summary(reject[:, column]),
                "naive_peeking_false_positive": _mc_summary(ever[:, column]),
                "additional_rejection_from_peeking": _mc_summary(ever[:, column] & ~reject[:, column]),
                "first_rejection_look_counts": {str(look): int(np.sum(first[:, column] == look)) for look in range(1, 51)},
                "never_rejected_count": int(np.sum(first[:, column] == 0)),
                "peeking_design": "Two disjoint cumulative groups from the remaining control pool; 50 correlated looks, 10,000 new users per arm per look.",
            })

        for total, n0, n1 in power_sizes:
            counts0 = _draw_cells(power_rng, control_cells, n0, repetitions["power"]) @ labels
            counts1 = _draw_cells(power_rng, treatment_cells, n1, repetitions["power"]) @ labels
            for column, outcome in enumerate(outcomes):
                p0, p1 = int(control_events[column]) / n0_data, int(treatment_events[column]) / n1_data
                mde = ((Z_975 + float(norm.ppf(0.8))) * math.sqrt(p0 * (1 - p0) * (1 / n0 + 1 / n1)) if 0 < p0 < 1 else None)
                result["power"][outcome].append({
                    "total_n": total,
                    "control_n": n0,
                    "treatment_n": n1,
                    "reference_control_rate": p0,
                    "reference_treatment_rate": p1,
                    "reference_difference": p1 - p0,
                    "empirical_power": _mc_summary(_rejection(counts0[:, column], counts1[:, column], n0, n1)),
                    "approximate_analytic_power": _analytic_power(p0, p1, n0, n1),
                    "approximate_finite_population_power": _analytic_power(p0, p1, n0, n1, n0_data, n1_data),
                    "control_variance_fpc": (n0_data - n0) / (n0_data - 1),
                    "treatment_variance_fpc": (n1_data - n1) / (n1_data - 1),
                    "approximate_mde80_absolute": mde,
                    "approximate_mde80_relative": mde / p0 if mde is not None else None,
                })

        for n in (500, 2_000, 10_000):
            counts0 = _draw_cells(coverage_rng, control_cells, n, repetitions["coverage"]) @ labels
            counts1 = _draw_cells(coverage_rng, treatment_cells, n, repetitions["coverage"]) @ labels
            for column, outcome in enumerate(outcomes):
                y0, y1 = counts0[:, column], counts1[:, column]
                p0, p1 = int(control_events[column]) / n0_data, int(treatment_events[column]) / n1_data
                delta = p1 - p0
                estimate0, estimate1 = y0 / n, y1 / n
                estimated_delta = estimate1 - estimate0
                se = np.sqrt((estimate0 * (1 - estimate0) + estimate1 * (1 - estimate1)) / n)
                wald_lo, wald_hi = estimated_delta - Z_975 * se, estimated_delta + Z_975 * se
                row = {
                    "per_arm_n": n,
                    "reference_control_rate": p0,
                    "reference_treatment_rate": p1,
                    "reference_difference": delta,
                    "wald_coverage": _mc_summary((wald_lo <= delta) & (delta <= wald_hi)),
                    "mean_wald_width": float(np.mean(wald_hi - wald_lo)),
                    "either_arm_zero_events": _mc_summary((y0 == 0) | (y1 == 0)),
                    "wald_interval_collapsed": _mc_summary(se == 0),
                }
                for prior_name, prior in (("jeffreys", 0.5), ("uniform", 1.0)):
                    posterior_lo, posterior_hi = _posterior_bounds(y0, y1, n, prior, posterior_draws, posterior_rng)
                    row[f"{prior_name}_coverage"] = _mc_summary((posterior_lo <= delta) & (delta <= posterior_hi))
                    row[f"mean_{prior_name}_width"] = float(np.mean(posterior_hi - posterior_lo))
                result["coverage"][outcome].append(row)
    return result
