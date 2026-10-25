"""Developer checks for inference formulas and exact finite-population draws."""

import json
import math
import unittest

import numpy as np
from scipy.stats import norm

from criteo_ab_uplift.simulations import _draw_cells, _population_blocks, run_simulations
from criteo_ab_uplift.statistics import _validate_groups, analyze_groups


def groups(y0=10, y1=20, d0=0, d1=50, joint0=0, joint1=15, n=100):
    return [
        {"treatment": 0, "n": n, "visit_sum": y0, "conversion_sum": 0,
         "exposure_sum": d0, "visit_exposure_sum": joint0, "conversion_exposure_sum": 0},
        {"treatment": 1, "n": n, "visit_sum": y1, "conversion_sum": 0,
         "exposure_sum": d1, "visit_exposure_sum": joint1, "conversion_exposure_sum": 0},
    ]


def full_count_fixture(identical_outcomes=False):
    # These are developer-only integer fixtures. They are never passed off as
    # observed Criteo results or distributed as project output.
    arms = []
    cells = []
    for t, n, visits, conversions in ((0, 1_000_001, 20_000, 1_000), (1, 3_000_003, 75_000, 4_500)):
        if identical_outcomes:
            conversions = visits
        exposure = 1_000_000 if t else 0
        joint_visit = 40_000 if t else 0
        joint_conversion = joint_visit if identical_outcomes else 3_500 if t else 0
        arms.append({"treatment": t, "n": n, "visit_sum": visits, "conversion_sum": conversions,
                     "exposure_sum": exposure, "visit_exposure_sum": joint_visit,
                     "conversion_exposure_sum": joint_conversion})
        counts = [n - visits, 0, visits - conversions, conversions]
        for key, count in enumerate(counts):
            cells.append({"treatment": t, "visit": key // 2, "conversion": key % 2, "n": count})
    return arms, cells


class StatisticsTests(unittest.TestCase):
    def test_difference_z_relative_lift_and_hc2_match_formulas(self):
        f = analyze_groups(groups(), posterior_draws=100)["outcomes"]["visit"]["frequentist"]
        self.assertAlmostEqual(f["difference"], 0.1)
        self.assertAlmostEqual(f["standard_error"], 0.05)
        self.assertAlmostEqual(f["hc2_standard_error"]**2, (0.1 * 0.9 + 0.2 * 0.8) / 99)
        self.assertAlmostEqual(f["wald_ci95"][0], 0.1 - norm.ppf(0.975) * 0.05)
        self.assertAlmostEqual(f["pooled_z"], 0.1 / math.sqrt(0.15 * 0.85 * 0.02))
        self.assertAlmostEqual(f["relative_lift"], 1.0)
        self.assertAlmostEqual(f["relative_lift_standard_error"], math.sqrt(0.52))

    def test_zero_events_finite_and_posterior_not_collapsed(self):
        result = analyze_groups(groups(y0=0, y1=0, joint1=0), posterior_draws=2000)
        f = result["outcomes"]["visit"]["frequentist"]
        self.assertEqual(f["two_sided_p_value"], 1.0)
        self.assertEqual(f["wald_ci95"], [0.0, 0.0])
        self.assertIsNone(f["relative_lift"])
        b = result["outcomes"]["visit"]["bayesian"]["jeffreys"]
        self.assertLess(b["credible_interval95"][0], 0)
        self.assertGreater(b["credible_interval95"][1], 0)
        json.dumps(result, allow_nan=False)

    def test_iv_hc2_delta_variance_uses_covariance(self):
        result = analyze_groups(groups(), posterior_draws=100)["iv_exploratory"]
        iv = result["outcomes"]["visit"]
        ratio = 0.1 / 0.5
        var_y, var_d = (0.2 * 0.8 + 0.1 * 0.9) / 99, 0.5 * 0.5 / 99
        covariance = (0.15 - 0.2 * 0.5) / 99
        expected = (var_y + ratio**2 * var_d - 2 * ratio * covariance) / 0.5**2
        self.assertAlmostEqual(iv["wald_ratio"], ratio)
        self.assertAlmostEqual(iv["robust_delta_standard_error"]**2, expected)
        self.assertEqual(result["effect_label"], "CACE / ATT")
        self.assertIn("all observed exposed users", result["interpretation"])

    def test_control_exposure_reports_late_using_both_first_stage_rates(self):
        result = analyze_groups(groups(d0=5, joint0=2), posterior_draws=100)["iv_exploratory"]
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["one_sided_noncompliance"])
        self.assertEqual(result["effect_label"], "LATE")
        self.assertAlmostEqual(result["first_stage"], 0.45)
        self.assertAlmostEqual(result["outcomes"]["visit"]["wald_ratio"], 0.1 / 0.45)
        self.assertTrue(any("Monotonicity" in a for a in result["required_assumptions"]))
        self.assertIn("no ATT interpretation", result["interpretation"])
        json.dumps(result, allow_nan=False)

    def test_first_stage_guard_and_invalid_counts(self):
        result = analyze_groups(groups(d1=0, joint1=0), posterior_draws=10)["iv_exploratory"]
        self.assertEqual(result["status"], "skipped_zero_or_negative_first_stage")
        self.assertFalse(result["outcomes"])
        with self.assertRaises(ValueError):
            analyze_groups(groups(joint1=30), posterior_draws=10)
        with self.assertRaises(ValueError):
            analyze_groups(groups(n=0), posterior_draws=10)

    def test_posterior_reproducible_with_only_approved_priors(self):
        first = analyze_groups(groups(), seed=7, posterior_draws=2000)
        self.assertEqual(first, analyze_groups(groups(), seed=7, posterior_draws=2000))
        self.assertEqual(set(first["outcomes"]["visit"]["bayesian"]), {"jeffreys", "uniform"})
        b = first["outcomes"]["visit"]["bayesian"]["jeffreys"]
        self.assertAlmostEqual(b["expected_loss_control"] - b["expected_loss_treatment"], b["posterior_mean_difference"], delta=0.006)


class FinitePopulationTests(unittest.TestCase):
    def test_multivariate_hypergeometric_moments_and_complements(self):
        cells = np.array([2, 1, 2, 1])
        draws = _draw_cells(np.random.default_rng(2026), cells, 3, 60_000)
        np.testing.assert_array_equal(draws.sum(axis=1), 3)
        self.assertTrue(np.all(draws <= cells))
        np.testing.assert_allclose(draws.mean(axis=0), 3 * cells / 6, atol=0.012)
        expected_var = 3 * (cells / 6) * (1 - cells / 6) * (6 - 3) / (6 - 1)
        np.testing.assert_allclose(draws.var(axis=0), expected_var, atol=0.012)
        complements = cells - draws
        np.testing.assert_array_equal(draws + complements, np.broadcast_to(cells, draws.shape))
        np.testing.assert_array_equal(_draw_cells(np.random.default_rng(2), cells, 6, 3), np.broadcast_to(cells, (3, 4)))

    def test_remaining_pools_and_structural_zero_cells(self):
        pools = np.array([[0, 2, 3, 0], [2, 0, 0, 3], [0, 0, 5, 0]])
        rng = np.random.default_rng(1)
        first = _draw_cells(rng, pools, 2, 3)
        rest = _draw_cells(rng, pools - first, 3, 3)
        np.testing.assert_array_equal(first + rest, pools)
        self.assertTrue(np.all(first[pools == 0] == 0))
        with self.assertRaises(ValueError):
            _draw_cells(rng, pools, 6, 3)

    def test_joint_counts_validate_every_marginal(self):
        arms, cells = full_count_fixture()
        block = _population_blocks(_validate_groups(arms), cells)[0]
        self.assertEqual(block["outcomes"], ("visit", "conversion"))
        cells[0]["n"] += 1
        with self.assertRaises(ValueError):
            _population_blocks(_validate_groups(arms), cells)

    def test_simulation_fixed_design_reproducibility_and_paired_outcomes(self):
        arms, cells = full_count_fixture(identical_outcomes=True)
        kwargs = dict(seed=8, joint_counts=cells, aa_reps=24, power_reps=24, peeking_reps=24,
                      coverage_reps=24, posterior_draws=128)
        result = run_simulations(arms, **kwargs)
        self.assertEqual(result, run_simulations(arms, **kwargs))
        self.assertTrue(result["metadata"]["outcomes_jointly_sampled"])
        for outcome in ("visit", "conversion"):
            aa = result["aa_peeking"][outcome]
            self.assertEqual(aa["aa_control_n"] + aa["aa_treatment_n"], 1_000_001)
            self.assertEqual(aa["looks"], 50)
            self.assertEqual(aa["per_look"][-1]["per_arm_n"], 500_000)
            fixed, peek = aa["fixed_horizon_false_positive"]["rate"], aa["naive_peeking_false_positive"]["rate"]
            self.assertGreaterEqual(peek, fixed)
            self.assertAlmostEqual(aa["additional_rejection_from_peeking"]["rate"], peek - fixed)
            self.assertEqual(sum(aa["first_rejection_look_counts"].values()) + aa["never_rejected_count"], 24)
            self.assertEqual([r["total_n"] for r in result["power"][outcome]], [10_000, 100_000, 1_000_000])
            self.assertEqual([r["per_arm_n"] for r in result["coverage"][outcome]], [500, 2_000, 10_000])
            for row in result["coverage"][outcome]:
                self.assertTrue(0 <= row["jeffreys_coverage"]["rate"] <= 1)
                self.assertTrue(0 <= row["uniform_coverage"]["rate"] <= 1)
        self.assertEqual(result["aa_peeking"]["visit"]["aa_p_values"], result["aa_peeking"]["conversion"]["aa_p_values"])
        self.assertEqual(result["aa_peeking"]["visit"]["per_look"], result["aa_peeking"]["conversion"]["per_look"])
        self.assertEqual(result["power"]["visit"], result["power"]["conversion"])
        json.dumps(result, allow_nan=False)

    def test_small_population_fails_instead_of_changing_design(self):
        with self.assertRaisesRegex(ValueError, "1,000,000"):
            run_simulations(groups(), aa_reps=2, power_reps=2, peeking_reps=2, coverage_reps=2)


if __name__ == "__main__":
    unittest.main()
