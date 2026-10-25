# Criteo A/B Testing & Uplift Modeling: Findings

The analysis uses all 13,979,592 records in Criteo's public uplift dataset to compare treatment and control outcomes and assess targeting rankings. Visit is the primary outcome; conversion is secondary.

The treatment group's visit rate was 1.0342 percentage points higher than the control group's (95% Wald CI [1.0056, 1.0629]), a relative difference of 27.07%. Lin adjustment reduced the difference to 0.7734 percentage points after standardising both arms to the pooled covariate mean. The adjusted-to-unadjusted HC2 standard error ratio was 0.8971 for visit and 0.9885 for conversion. Bayesian estimates used uniform and Jeffreys Beta priors. No informative prior was used because no independent historical data were available.

Assignment is not balanced on recorded features in the released data: all 12 features differ by about 7 to 65 randomisation standard deviations. This is consistent with the gap between raw and adjusted estimates; neither should be read as Criteo's advertising effect. Non-uniform privacy sub-sampling is a plausible source of the imbalance.

A/A splits of the control pool produced a visit rejection rate of 5.50%. Repeated significance checks increased the probability of at least one rejection to 30.10%. Power and coverage were evaluated by sampling without replacement from the public sample, using its treatment/control rate difference as the reference value.

The targeting comparison used a stratified training/test split and fixed LightGBM parameters. X-learner was selected as the primary model before test evaluation. X-learner's score exceeds the random-ranking benchmark at 10%, 30%, 50% selection fractions, with pointwise confidence intervals for the differences above zero.

| Selected fraction | X-learner improvement over random | Pointwise 95% CI |
| --- | --- | --- |
| 10% | 51.316 | [47.972, 54.612] |
| 30% | 18.924 | [17.865, 20.026] |
| 50% | 9.391 | [8.913, 9.896] |

The metric is expressed in incremental visits per 1,000 users selected by the policy and assigned to treatment, including assigned users who were not exposed. Given the covariate imbalance between assignment groups, these scores are descriptive group contrasts, not identified causal gains. Bootstrap intervals reselect the top-k users while holding fitted models fixed and reflect test-sample uncertainty only.

The exposure IV estimate also relies on random assignment within the released sample, which the balance diagnostics call into question. Its Wald ratio should therefore be read descriptively. Unknown privacy sampling prevents recovery of the original advertising effect; without cost and revenue data, the ranking comparisons cannot be translated into ROI.

Results: [results/full-run/](results/full-run/). Methods and references: [README.md](README.md). Analysis commit: `6928958aac539e9314c5d992203a826e9d1e5037`.
