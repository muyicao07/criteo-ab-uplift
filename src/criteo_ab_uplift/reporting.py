"""Generate publishable tables, figures and prose exclusively from run outputs."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCES = """1. Diemert, E., Betlei, A., Renaudin, C. & Amini, M.-R. (2018). [A Large Scale Benchmark for Uplift Modeling](https://www.adkdd.org/papers/a-large-scale-benchmark-for-uplift-modeling/2018). AdKDD & TargetAd Workshop.
2. Deng, A., Xu, Y., Kohavi, R. & Walker, T. (2013). [Improving the sensitivity of online controlled experiments by utilizing pre-experiment data](https://doi.org/10.1145/2433396.2433413). WSDM.
3. Lin, W. (2013). [Agnostic notes on regression adjustments to experimental data: Reexamining Freedman's critique](https://doi.org/10.1214/12-AOAS583). Annals of Applied Statistics, 7(1), 295-318.
4. Angrist, J. D., Imbens, G. W. & Rubin, D. B. (1996). [Identification of causal effects using instrumental variables](https://doi.org/10.1080/01621459.1996.10476902). JASA, 91(434), 444-455.
5. Künzel, S. R., Sekhon, J. S., Bickel, P. J. & Yu, B. (2019). [Metalearners for estimating heterogeneous treatment effects using machine learning](https://doi.org/10.1073/pnas.1804597116). PNAS, 116(10), 4156-4165.
6. Radcliffe, N. J. (2007). [Using control groups to target on predicted lift: Building and assessing uplift model](https://www.research.ed.ac.uk/en/publications/using-control-groups-to-target-on-predicted-lift-building-and-ass/). Direct Marketing Analytics Journal, 14-21.
"""


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(x, digits=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "undefined"
    return f"{x:.{digits}f}"


def _pvalue(x):
    return "< 1e-300" if x < 1e-300 else f"{x:.3g}"


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def write_tables(output: Path):
    audit, ab, lin, sims = [load(output / f) for f in
                            ["data_audit.json", "ab_results.json", "regression_adjustment.json", "simulations.json"]]
    pd.DataFrame(audit["balance"]).to_csv(output / "feature_balance.csv", index=False)
    rows, bayes, iv_rows = [], [], []
    for outcome, result in ab["outcomes"].items():
        f = result["frequentist"]
        rows.append({"outcome": outcome, **{k: v for k, v in f.items() if isinstance(v, (float, int)) or v is None},
                     "ci_low": f["wald_ci95"][0], "ci_high": f["wald_ci95"][1],
                     "relative_lift_ci_low": None if f["relative_lift_ci95"] is None else f["relative_lift_ci95"][0],
                     "relative_lift_ci_high": None if f["relative_lift_ci95"] is None else f["relative_lift_ci95"][1]})
        for prior, b in result["bayesian"].items():
            bayes.append({"outcome": outcome, "prior": prior,
                          **{k: v for k, v in b.items() if isinstance(v, (float, int))},
                          "ci_low": b["credible_interval95"][0], "ci_high": b["credible_interval95"][1]})
    for outcome, row in ab["iv_exploratory"]["outcomes"].items():
        iv_rows.append({"outcome": outcome,
                        **{k: v for k, v in row.items() if isinstance(v, (float, int)) or v is None},
                        "ci_low": row["normal_ci95"][0], "ci_high": row["normal_ci95"][1]})
    pd.DataFrame(rows).to_csv(output / "frequentist.csv", index=False)
    pd.DataFrame(bayes).to_csv(output / "bayesian.csv", index=False)
    pd.DataFrame(iv_rows).to_csv(output / "iv.csv", index=False)
    pd.DataFrame([{"outcome": o, **v} for o, v in lin["outcomes"].items()]).to_csv(output / "adjustment.csv", index=False)
    for kind in ["power", "coverage"]:
        flattened = []
        for outcome, records in sims[kind].items():
            for row in records:
                flat = {"outcome": outcome}
                for key, value in row.items():
                    if isinstance(value, dict) and "rate" in value:
                        flat[key] = value["rate"]
                        flat[key + "_mc_low"], flat[key + "_mc_high"] = value["mc_wilson_interval95"]
                    elif not isinstance(value, (dict, list)):
                        flat[key] = value
                flattened.append(flat)
        pd.DataFrame(flattened).to_csv(output / f"{kind}.csv", index=False)
    aa_rows, peek_rows = [], []
    for outcome, row in sims["aa_peeking"].items():
        for method, key in [("A/A complementary control halves", "aa_false_positive"),
                            ("Fixed final look", "fixed_horizon_false_positive"),
                            ("Any naive peeking look", "naive_peeking_false_positive")]:
            s = row[key]
            aa_rows.append({"outcome": outcome, "method": method, "rejection_rate": s["rate"],
                            "mc_low": s["mc_wilson_interval95"][0], "mc_high": s["mc_wilson_interval95"][1],
                            "repetitions": s["repetitions"]})
        for look in row["per_look"]:
            peek_rows.append({"outcome": outcome, "look": look["look"], "per_arm_n": look["per_arm_n"],
                              "fixed_look": look["fixed_look_false_positive"]["rate"],
                              "ever_rejected": look["ever_rejected"]["rate"]})
    pd.DataFrame(aa_rows).to_csv(output / "aa_peeking.csv", index=False)
    pd.DataFrame(peek_rows).to_csv(output / "peeking_trajectory.csv", index=False)


def write_figures(output: Path):
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent / ".mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 12, "font.size": 10})
    figdir = output / "figures"
    figdir.mkdir(exist_ok=True)
    model_labels = {"x_learner": "X-learner", "t_learner": "T-learner", "response": "Response"}
    model_colors = {"x_learner": "#1f77b4", "t_learner": "#ff7f0e", "response": "#2ca02c"}

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figdir / name, bbox_inches="tight")
        plt.close(fig)

    balance = pd.read_csv(output / "feature_balance.csv")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(balance.feature, balance.smd, color="#247c95")
    ax.axhline(.1, ls="--", c="grey", label="Conventional |SMD| = 0.1 reference")
    ax.axhline(-.1, ls="--", c="grey")
    ax.set(title="Recorded covariate differences", ylabel="Standardized mean difference")
    ax.legend(fontsize=8, loc="lower right", framealpha=1)
    save(fig, "balance.png")

    f, b = pd.read_csv(output / "frequentist.csv"), pd.read_csv(output / "bayesian.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, outcome in zip(axes, ["visit", "conversion"]):
        ff, bb = f[f.outcome == outcome].iloc[0], b[b.outcome == outcome]
        values = [("Wald", ff.difference, ff.ci_low, ff.ci_high)] + [
            (r.prior, r.posterior_mean_difference, r.ci_low, r.ci_high) for r in bb.itertuples()]
        for j, (name, point, lo, hi) in enumerate(values):
            ax.hlines(j, 100 * lo, 100 * hi, lw=3)
            ax.scatter(100 * point, j)
        ax.set(yticks=range(len(values)), yticklabels=[x[0] for x in values],
               xlabel="Treatment difference (percentage points)", title=outcome.capitalize())
    save(fig, "average_effects.png")

    power, coverage = pd.read_csv(output / "power.csv"), pd.read_csv(output / "coverage.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for j, outcome in enumerate(["visit", "conversion"]):
        pp, cc = power[power.outcome == outcome], coverage[coverage.outcome == outcome]
        axes[0, j].plot(pp.total_n, pp.empirical_power, "o-", label="Empirical, without replacement")
        axes[0, j].plot(pp.total_n, pp.approximate_analytic_power, "--", label="IID normal approximation")
        axes[0, j].plot(pp.total_n, pp.approximate_finite_population_power, ":", label="Normal approximation with FPC")
        axes[0, j].set(xscale="log", ylim=(0, 1.05), title=outcome.capitalize(), ylabel="Rejection rate", xlabel="Total sample size")
        for col, name in [("wald_coverage", "Wald"), ("jeffreys_coverage", "Jeffreys"), ("uniform_coverage", "Uniform prior")]:
            axes[1, j].plot(cc.per_arm_n, cc[col], "o-", label=name)
        axes[1, j].axhline(.95, ls="--", c="grey")
        axes[1, j].set(xscale="log", ylim=(min(.8, float(cc.wald_coverage.min()) - .03), 1.02), ylabel="Coverage", xlabel="Sample size per arm")
        axes[0, j].legend(fontsize=7); axes[1, j].legend(fontsize=8)
    save(fig, "power_coverage.png")

    trajectory = pd.read_csv(output / "peeking_trajectory.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for ax, outcome in zip(axes, ["visit", "conversion"]):
        pp = trajectory[trajectory.outcome == outcome]
        ax.plot(pp.look, pp.ever_rejected, label="Rejected at least once")
        ax.plot(pp.look, pp.fixed_look, label="Rejected at this look")
        ax.axhline(.05, c="grey", ls="--")
        ax.set(title=outcome.capitalize(), xlabel="Sequential look", ylabel="False-positive rate")
        ax.legend(fontsize=8)
    save(fig, "peeking.png")

    curves = pd.read_csv(output / "uplift_curves.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, frame in curves.groupby("model", sort=False):
        ax.plot(frame.fraction_targeted, frame.incremental_visits,
                label=model_labels[model], color=model_colors[model])
    frame = curves[curves.model == curves.model.iloc[0]]
    ax.plot(frame.fraction_targeted, frame.qini_baseline, "--", c="grey", label="Random targeting")
    ax.set(xlabel="Fraction selected", ylabel="Qini cumulative incremental visits", title="Held-out Qini curves (descriptive)")
    ax.legend()
    save(fig, "qini.png")

    policies = pd.read_csv(output / "uplift_targeting.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, model in enumerate(["x_learner", "t_learner", "response"]):
        pp = policies[policies.model == model]
        xx = pp.k.to_numpy() + (j - 1) * .012
        color = model_colors[model]
        axes[0].scatter(xx, pp.incremental_visits_per_1000, label=model_labels[model], color=color)
        axes[0].vlines(xx, pp.ci_lower, pp.ci_upper, color=color)
        axes[1].scatter(xx, pp.gain_vs_random, label=model_labels[model], color=color)
        axes[1].vlines(xx, pp.gain_ci_lower, pp.gain_ci_upper, color=color)
    axes[0].axhline(policies.random_per_1000.iloc[0], ls="--", c="grey", label="Random targeting")
    axes[1].axhline(0, ls="--", c="grey")
    axes[0].set(title="Visit-rate contrast per 1,000 selected users", ylabel="Selected-group contrast")
    axes[1].set(title="Difference from random ranking", ylabel="Difference per 1,000 selected users")
    fig.suptitle("Descriptive comparisons in the test sample", fontsize=11)
    for ax in axes:
        ax.set_xlabel("Fraction selected"); ax.legend(fontsize=8)
    save(fig, "targeting.png")


def render_reports(root: Path, output: Path):
    """All outcome numbers are read back from the saved results, never typed in."""
    a, ab, lin, sim, manifest = [load(output / name) for name in
                               ["data_audit.json", "ab_results.json", "regression_adjustment.json", "simulations.json", "manifest.json"]]
    policies, qini = pd.read_csv(output / "uplift_targeting.csv"), pd.read_csv(output / "uplift_qini.csv")
    uplift = load(output / "uplift_results.json")
    rel = output.relative_to(root).as_posix()
    groups = {int(g["treatment"]): g for g in a["groups"]}
    balance = pd.read_csv(output / "feature_balance.csv")
    smds = balance.smd.to_numpy(dtype=float)
    randomisation_sd = float(np.sqrt(1 / groups[1]["n"] + 1 / groups[0]["n"]))
    z_abs = np.abs(smds) / randomisation_sd
    if not np.isfinite(z_abs).all():
        raise ValueError("Balance diagnostics require finite SMDs for every feature.")
    diagnostics = {
        "reference": "Approximate SMD standard deviation under simple random assignment: sqrt(1/n1 + 1/n0)",
        "n_treatment": groups[1]["n"], "n_control": groups[0]["n"],
        "randomisation_sd": randomisation_sd, "n_features": len(smds),
        "n_below_smd_0_1": int(np.sum(np.abs(smds) < .1)),
        "min_absolute_smd": float(np.min(np.abs(smds))), "max_absolute_smd": float(np.max(np.abs(smds))),
        "min_absolute_randomisation_z": float(np.min(z_abs)), "max_absolute_randomisation_z": float(np.max(z_abs)),
        "features": [{"feature": row.feature, "smd": float(row.smd),
                      "randomisation_z": float(row.smd / randomisation_sd)} for row in balance.itertuples()],
    }
    (output / "balance_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n")
    balance_range = f"{np.min(z_abs):.0f} to {np.max(z_abs):.0f}"
    scope = ("Criteo pooled several incrementality experiments and applied non-uniform sub-sampling for privacy. "
             "The sampling scheme is not documented, and the original incrementality level cannot be recovered. "
             "The average contrasts, IV estimates and targeting gains below describe the released sample; "
             "they are not estimates of Criteo's actual advertising effect.\n\n"
             f"All {len(smds)} features pass the conventional |SMD| < 0.1 screen, but this is not a test of randomisation "
             f"at this sample size. Under simple random assignment, an SMD would have an approximate standard deviation "
             f"of {randomisation_sd:.5f}. The observed absolute SMDs are {balance_range} such standard deviations across "
             "the features. These differences provide strong evidence that assignment is not independent of the "
             "recorded features in the released sample. Criteo's non-uniform privacy sub-sampling is a "
             "plausible explanation.\n\n"
             f"The change in the visit estimate from {100*lin['outcomes']['visit']['tau_unadjusted']:.2f} to "
             f"{100*lin['outcomes']['visit']['tau_adjusted']:.2f} percentage points after adjustment is consistent with "
             "imbalance in outcome-predictive covariates. Uplift evaluation compares treatment and control users "
             "within each selected group and uses the same assignments, so its results are descriptive of this sample.")
    frequentist_rows, bayes_rows, adjust_rows, iv_rows = [], [], [], []
    for outcome, result in ab["outcomes"].items():
        f = result["frequentist"]
        lo, hi = f["wald_ci95"]
        frequentist_rows.append([outcome, f"{f['control_rate']:.4%}", f"{f['treatment_rate']:.4%}",
                                f"{100*f['difference']:.4f}", f"[{100*lo:.4f}, {100*hi:.4f}]",
                                "undefined" if f["relative_lift"] is None else f"{f['relative_lift']:.2%}",
                                _pvalue(f['two_sided_p_value'])])
        for prior, b in result["bayesian"].items():
            blo, bhi = b["credible_interval95"]
            bayes_rows.append([outcome, prior, f"{b['probability_positive']:.5f}",
                               f"[{100*blo:.4f}, {100*bhi:.4f}]", f"{b['expected_loss_treatment']:.3e}",
                               f"{b['expected_loss_control']:.3e}"])
        rr = lin["outcomes"][outcome]
        adjust_rows.append([outcome, f"{100*rr['tau_unadjusted']:.4f}",
                            f"{100*rr['tau_adjusted']:.4f}",
                            f"[{100*rr['ci_low']:.4f}, {100*rr['ci_high']:.4f}]",
                            f"{rr['se_adjusted']:.6g}", f"{rr['se_unadjusted_hc2']:.6g}", f"{rr['se_ratio']:.4f}"])
    iv = ab["iv_exploratory"]
    for outcome, row in iv["outcomes"].items():
        lo, hi = row["normal_ci95"]
        iv_rows.append([outcome, f"{100*row['wald_ratio']:.4f}", f"[{100*lo:.4f}, {100*hi:.4f}]",
                        _fmt(None if row['naive_exposed_minus_unexposed_within_treatment'] is None else
                             100*row['naive_exposed_minus_unexposed_within_treatment'])])
    iv_text = ("Using assignment as an instrument for actual exposure, the Wald ratio has a complier/local average "
               "causal interpretation under: (i) random assignment within the released sample; "
               "(ii) exclusion, so assignment affects outcomes only through exposure; (iii) the compliance assumptions "
               "below; (iv) relevance; and (v) SUTVA, with no interference and a single version of exposure. ")
    if a["control_exposure_count"] == 0:
        iv_text += (f"The observed control exposure rate is {iv['control_exposure_rate']:.6%}. Under one-sided noncompliance, "
                    "all observed exposed users are compliers. With random assignment, they are representative of the "
                    "complier population, so the CACE also equals the average effect of exposure on the exposed (ATT), "
                    "subject to the assumptions and sampling caveat above. ")
    else:
        iv_text += (f"The control exposure rate is {iv['control_exposure_rate']:.6%}, so one-sided noncompliance does not hold. "
                    "The full difference in exposure rates is used as the denominator; monotonicity is an additional "
                    "assumption, and the estimate is labelled LATE without an ATT interpretation. ")
    iv_text += ("The estimate also depends on the accuracy of the exposure flag. The balance diagnostics show that "
                "assignment is not independent of the features in the released sample, so assumption (i) is doubtful "
                "and the IV estimate should be read as descriptive of this sample.")
    policy_rows = [[r.model, f"{r.k:.0%}", f"{r.incremental_visits_per_1000:.3f}",
                    f"[{r.ci_lower:.3f}, {r.ci_upper:.3f}]", f"{r.random_per_1000:.3f}",
                    f"{r.gain_vs_random:.3f}", f"[{r.gain_ci_lower:.3f}, {r.gain_ci_upper:.3f}]"]
                   for r in policies.itertuples()]
    aa_rows = []
    for outcome, r in sim["aa_peeking"].items():
        aa_rows.append([outcome, f"{r['aa_false_positive']['rate']:.3%}",
                        f"{r['fixed_horizon_false_positive']['rate']:.3%}",
                        f"{r['naive_peeking_false_positive']['rate']:.3%}"])
    target_unit = "incremental visits per 1,000 users selected by the policy and assigned to treatment (intention-to-treat, including users who were assigned but not exposed)"
    random_reference = policies.iloc[0]
    text = f"""# Criteo A/B Testing & Uplift Modeling

This project compares visit and conversion rates between treatment and control in Criteo's public experiment data, then evaluates uplift rankings for a limited advertising budget. The analysis uses all {a['n_total']:,} records. Visit is the primary outcome; conversion is secondary in the experiment analysis. Targeting models are evaluated on visit only.

The released data show substantial feature imbalance relative to randomisation variability. The estimates are therefore interpreted as descriptive comparisons within this sample. Tables and figures are saved in [{rel}/]({rel}/), with a summary in [CONCLUSION.md](CONCLUSION.md).

## Main results

{_table(['Outcome','Control rate','Treatment rate','Difference (pp)','Wald 95% CI (pp)','Relative lift','Two-sided p'], frequentist_rows)}

{_table(['Model','Normalized Qini coefficient','Bootstrap 95% CI'], [[r.model, f'{r.qini_coefficient:.6f}', f'[{r.ci_lower:.6f}, {r.ci_upper:.6f}]'] for r in qini.itertuples()])}

X-learner is the pre-specified primary model. T-learner and a response ranking serve as benchmarks. Model choice and hyperparameters were fixed before evaluating the test set.

## Data and scope

The [Criteo Uplift Prediction Dataset v2.1](https://huggingface.co/datasets/criteo/criteo-uplift) contains {groups[0]['n']:,} control users and {groups[1]['n']:,} treatment users ({a['treatment_fraction']:.4%} treated). The download is checked against the release's SHA-256 and expected row count before analysis.

{scope}

![Covariate balance]({rel}/figures/balance.png)

Maximum absolute SMD is {np.max(np.abs(smds)):.5f} (about {np.max(z_abs):.0f} randomisation standard deviations). The reference standard deviation is calculated as `sqrt(1/n_treatment + 1/n_control)`; feature-level values are saved in [balance_diagnostics.json]({rel}/balance_diagnostics.json). The anonymous features are treated as numeric predictors, so differences in their means provide only a partial description of imbalance. The original allocation probability is unavailable for a definitive sample-ratio mismatch test.

## Frequentist and Bayesian comparisons

Groups are defined by treatment assignment. Absolute differences are reported in percentage points (pp). Relative lift and its delta-method interval are available in [frequentist.csv]({rel}/frequentist.csv). The z-test uses pooled null variance, and the Wald interval uses unpooled variance. A p-value below numerical precision is displayed as `< 1e-300`. Expected losses use scientific notation; a zero Monte Carlo estimate does not imply that the true posterior loss is exactly zero.

{_table(['Outcome','Prior','P(difference > 0)','95% credible interval (pp)','Expected loss: treatment','Expected loss: control'], bayes_rows)}

Expected loss is measured in outcome probability per user. The Bayesian analysis uses uniform and Jeffreys Beta priors. No informative prior was used because no independent historical data were available. The coverage experiment compares interval behaviour in smaller samples.

![Intervals]({rel}/figures/average_effects.png)

## Full-data covariate adjustment

{_table(['Outcome','Unadjusted difference (pp)','Adjusted difference (pp)','Adjusted 95% CI (pp)','Adjusted HC2 SE','Unadjusted HC2 SE','SE ratio'], adjust_rows)}

The fully interacted Lin model is fitted as separate OLS regressions in each arm, evaluated at the pooled covariate mean. This mean is treated as fixed. Two passes through the data compute the regression estimates and HC2 variance in chunks. The unadjusted comparison uses the HC2 variance from a treatment-only regression on the same observations. The dataset has no pre-experiment outcomes for the original CUPED construction.

Adjustment standardises the comparison to a common covariate mean, changing the point estimate as well as its standard error. The shift is consistent with the feature imbalance described above. Adjusting for the recorded covariates does not recover the original experiment or establish a causal interpretation for the released sample.

## A/A, power, peeking and coverage

{_table(['Outcome','A/A rejection rate','Fixed final-look rejection','Any-look peeking rejection'], aa_rows)}

{sim['metadata']['interpretation']} For conversion, the full-sample conversion-rate difference is the reference value.

A/A splits the complete control pool into complementary random halves. Peeking adds disjoint groups from the remaining control pool at each look. Joint hypergeometric counts preserve the visit/conversion pairing. These simulations assess the statistical procedures under the specified sampling designs; they do not establish random assignment in the released dataset. Monte Carlo intervals, minimum detectable effects and analytical power approximations are included in the saved results.

Coverage is evaluated for ordinary Wald and independent Beta posterior intervals without a finite-population correction. The power figure includes the IID normal approximation and an approximation with a finite-population correction to the alternative variance. Both use the same pooled-z rejection rule.

![Peeking]({rel}/figures/peeking.png)
![Power and coverage]({rel}/figures/power_coverage.png)

## Exposure IV

{_table(['Outcome','Wald ratio (pp)','95% CI (pp)','Naive exposed/unexposed contrast (pp)'], iv_rows)}

First-stage exposure difference: {iv['first_stage']:.6f}; first-stage F statistic: {_fmt(iv['first_stage_f_statistic'], 2)}. The robust ratio variance includes covariance between the outcome and exposure contrasts.

{iv_text}

## Held-out uplift and targeting

The training split contains {manifest['configuration']['train_fraction']:.0%} of the data and is stratified by (treatment, visit), with seed {manifest['configuration']['seed']}. LightGBM uses the parameters in [configuration.json]({rel}/configuration.json), with no tuning or early stopping. The response baseline ranks users by the treated response model. X-learner combines its effect models using the training treatment fraction as a constant weight.

![Qini curves]({rel}/figures/qini.png)

The metric is expressed as {target_unit}. Each policy estimate is the treatment/control visit-rate difference in its top-k set, multiplied by 1,000. Random targeting uses the difference across the entire test set. Improvement is the policy estimate minus this random benchmark, on the same per-user scale. Given the covariate imbalance between assignment groups in the released data, these are descriptive ranking comparisons rather than identified causal targeting gains.

{_table(['Model','Selected','Policy increment','95% CI','Random increment','Improvement','95% CI of improvement'], policy_rows)}

The random-targeting estimate is {random_reference.random_per_1000:.3f} (bootstrap 95% CI [{random_reference.random_ci_lower:.3f}, {random_reference.random_ci_upper:.3f}]). It is shared across models and budgets because all comparisons use the same test resamples.

![Targeting]({rel}/figures/targeting.png)

Qini is calculated with `sklift.metrics.qini_auc_score(negative_effect=True)`, including its normalisation and grouping of tied scores. Ties at targeting cutoffs are broken with fixed seeded random keys.

The paired bootstrap draws a sample of the original test-set size with replacement. Each replicate reselects floor(k*n) rows, splits repeated-row counts at the cutoff where necessary, and recalculates the random baseline. All models use the same draws. The {manifest['configuration']['bootstrap_repetitions']} replicates give percentile intervals that are pointwise across models and budgets. The fitted models remain fixed, so these intervals cover test-sample uncertainty only. They do not account for bias from the released sample's assignment mechanism.

## Reproduce

With Python 3.12 installed, run from the project directory:

```bash
bash run.sh
```

On macOS, double-clicking `Run_Project.command` runs the same command. The script creates a virtual environment, installs the locked dependencies, downloads the data and runs the analysis. It then updates the reports and executes the notebooks from saved results. Existing data and model checkpoints are reused. Set `CRITEO_THREADS` to control the CPU thread count; the default is 4. The script also supports Linux.

Raw data and model caches are stored in `data/`, which is excluded from version control. Each run saves results in a separate folder. Numerical content in the README and conclusion is filled from the result files.

[manifest.json]({rel}/manifest.json) records analysis commit `{manifest['git']['commit']}`, the source fingerprint, package versions, data SHA-256, seeds, hardware, timings and sampled process-tree memory. Timing and memory measurements apply to that invocation. Reused results are marked, and the original uplift computation time is retained separately.

## Repository

- `sql/`: full-data validation and descriptive queries.
- `src/`: download, analysis, simulation, full-data adjustment, uplift and report generation.
- `notebooks/`: five notebooks presenting the saved analysis results.
- `results/`: generated numerical outputs, figures and manifests.
- `CONCLUSION.md`: summary of findings and limitations.

## Licenses and references

Project code: [MIT](LICENSE). Dataset: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), under the [Criteo data terms](https://ailab.criteo.com/criteo-uplift-prediction-dataset/). The dataset is downloaded separately and is not redistributed or relicensed by the code license.

{REFERENCES}
"""
    (root / "README.md").write_text(text, encoding="utf-8")
    visit = ab["outcomes"]["visit"]["frequentist"]
    main = policies[policies.model == "x_learner"]
    evidence = main[main.gain_ci_lower > 0]
    conclusion = ("X-learner's score exceeds the random-ranking benchmark at " +
                  ", ".join(f"{k:.0%}" for k in evidence.k) +
                  " selection fractions, with pointwise confidence intervals for the differences above zero.") if len(evidence) else (
                  "The pointwise confidence intervals do not establish a higher score than random ranking at the evaluated selection fractions.")
    page = f"""# Criteo A/B Testing & Uplift Modeling: Findings

The analysis uses all {a['n_total']:,} records in Criteo's public uplift dataset to compare treatment and control outcomes and assess targeting rankings. Visit is the primary outcome; conversion is secondary.

The treatment group's visit rate was {100*visit['difference']:.4f} percentage points higher than the control group's (95% Wald CI [{100*visit['wald_ci95'][0]:.4f}, {100*visit['wald_ci95'][1]:.4f}]), a relative difference of {100*visit['relative_lift']:.2f}%. Lin adjustment reduced the difference to {100*lin['outcomes']['visit']['tau_adjusted']:.4f} percentage points after standardising both arms to the pooled covariate mean. The adjusted-to-unadjusted HC2 standard error ratio was {lin['outcomes']['visit']['se_ratio']:.4f} for visit and {lin['outcomes']['conversion']['se_ratio']:.4f} for conversion. Bayesian estimates used uniform and Jeffreys Beta priors. No informative prior was used because no independent historical data were available.

Assignment is not balanced on recorded features in the released data: all {len(smds)} features differ by about {balance_range} randomisation standard deviations. This is consistent with the gap between raw and adjusted estimates; neither should be read as Criteo's advertising effect. Non-uniform privacy sub-sampling is a plausible source of the imbalance.

A/A splits of the control pool produced a visit rejection rate of {sim['aa_peeking']['visit']['aa_false_positive']['rate']:.2%}. Repeated significance checks increased the probability of at least one rejection to {sim['aa_peeking']['visit']['naive_peeking_false_positive']['rate']:.2%}. Power and coverage were evaluated by sampling without replacement from the public sample, using its treatment/control rate difference as the reference value.

The targeting comparison used a stratified training/test split and fixed LightGBM parameters. X-learner was selected as the primary model before test evaluation. {conclusion}

{_table(['Selected fraction','X-learner improvement over random','Pointwise 95% CI'], [[f'{r.k:.0%}', f'{r.gain_vs_random:.3f}', f'[{r.gain_ci_lower:.3f}, {r.gain_ci_upper:.3f}]'] for r in main.itertuples()])}

The metric is expressed in incremental visits per 1,000 users selected by the policy and assigned to treatment, including assigned users who were not exposed. Given the covariate imbalance between assignment groups, these scores are descriptive group contrasts, not identified causal gains. Bootstrap intervals reselect the top-k users while holding fitted models fixed and reflect test-sample uncertainty only.

The exposure IV estimate also relies on random assignment within the released sample, which the balance diagnostics call into question. Its Wald ratio should therefore be read descriptively. Unknown privacy sampling prevents recovery of the original advertising effect; without cost and revenue data, the ranking comparisons cannot be translated into ROI.

Results: [{rel}/]({rel}/). Methods and references: [README.md](README.md). Analysis commit: `{manifest['git']['commit']}`.
"""
    (root / "CONCLUSION.md").write_text(page, encoding="utf-8")


def execute_notebooks(root: Path, output: Path):
    import nbformat
    from nbclient import NotebookClient
    rel = output.relative_to(root).as_posix()
    common = f'''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
RESULTS = ROOT / {rel!r}
def table(name):
    frame = pd.read_csv(RESULTS / name)
    for column in frame.columns:
        if column == "two_sided_p_value":
            frame[column] = frame[column].map(lambda x: "< 1e-300" if x < 1e-300 else format(x, ".3g"))
        elif column.startswith("expected_loss_"):
            frame[column] = frame[column].map(lambda x: format(x, ".3e"))
    display(frame)
def figure(name):
    display(Image(filename=str(RESULTS / "figures" / name)))
def record(name):
    return json.loads((RESULTS / name).read_text())
'''
    specs = [
        ("01_data_checks", "Data checks", "The SQL queries audit the full dataset. Feature differences are compared with the approximate SMD variation under simple random assignment. The observed differences are too large to support randomisation within the released sample.",
         'a = record("data_audit.json")\ndisplay(pd.DataFrame(a["groups"]))\ntable("feature_balance.csv")\ndisplay(record("balance_diagnostics.json"))\nfigure("balance.png")'),
        ("02_frequentist", "Frequentist A/B and experiment diagnostics", "Assignment defines the comparison. The full-data Lin model includes all treatment interactions; adjusted and unadjusted SEs share the HC2 convention. Count simulations sample the released finite population without replacement.",
         'table("frequentist.csv")\ntable("adjustment.csv")\ntable("aa_peeking.csv")\ntable("power.csv")\nfigure("peeking.png")'),
        ("03_bayesian", "Bayesian comparison and coverage", "Posterior estimates use uniform and Jeffreys Beta priors. Coverage is evaluated against the full public-sample rate difference under sampling without replacement.",
         'table("bayesian.csv")\ntable("coverage.csv")\nfigure("average_effects.png")\nfigure("power_coverage.png")'),
        ("04_exposure_iv", "Exposure as an endogenous treatment", "The Wald ratio uses assignment as an instrument. The feature imbalance calls the random-assignment assumption into question, so the ratio is descriptive of the released sample. A CACE/ATT interpretation would require the assumptions listed below.",
         'iv = record("ab_results.json")["iv_exploratory"]\ndisplay(Markdown(iv["interpretation"]))\ndisplay(iv["required_assumptions"])\ntable("iv.csv")'),
        ("05_uplift", "Uplift rankings on the test sample", "X-learner is the pre-specified primary model; T-learner and response rankings are benchmarks. Covariate imbalance between assignment groups makes these descriptive ranking comparisons. Paired row bootstrap reselects the top-k users and holds fitted models fixed; it does not account for selection bias in the release.",
         'table("uplift_qini.csv")\ntable("uplift_targeting.csv")\nfigure("qini.png")\nfigure("targeting.png")'),
    ]
    folder = root / "notebooks"
    folder.mkdir(exist_ok=True)
    for name, title, description, code in specs:
        nb = nbformat.v4.new_notebook(cells=[
            nbformat.v4.new_markdown_cell(f"# {title}\n\n{description}\n\nResults are loaded from the saved analysis outputs."),
            nbformat.v4.new_code_cell(common), nbformat.v4.new_code_cell(code),
            nbformat.v4.new_code_cell('m = record("manifest.json")\ndisplay({k: m[k] for k in ["git", "data_sha256", "seeds", "hardware"]})')])
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
        print(f"Executing results-only notebook: {name}", flush=True)
        NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(root)}}).execute()
        nbformat.write(nb, folder / f"{name}.ipynb")
