-- Exact arm counts, outcome/exposure cross-counts and covariate balance.
SELECT treatment, COUNT(*) AS n,
       SUM(visit) AS visit_sum, SUM(conversion) AS conversion_sum,
       SUM(exposure) AS exposure_sum,
       SUM(visit * exposure) AS visit_exposure_sum,
       SUM(conversion * exposure) AS conversion_exposure_sum,
       AVG(f0) AS mean_f0, STDDEV_SAMP(f0) AS std_f0,
       AVG(f1) AS mean_f1, STDDEV_SAMP(f1) AS std_f1,
       AVG(f2) AS mean_f2, STDDEV_SAMP(f2) AS std_f2,
       AVG(f3) AS mean_f3, STDDEV_SAMP(f3) AS std_f3,
       AVG(f4) AS mean_f4, STDDEV_SAMP(f4) AS std_f4,
       AVG(f5) AS mean_f5, STDDEV_SAMP(f5) AS std_f5,
       AVG(f6) AS mean_f6, STDDEV_SAMP(f6) AS std_f6,
       AVG(f7) AS mean_f7, STDDEV_SAMP(f7) AS std_f7,
       AVG(f8) AS mean_f8, STDDEV_SAMP(f8) AS std_f8,
       AVG(f9) AS mean_f9, STDDEV_SAMP(f9) AS std_f9,
       AVG(f10) AS mean_f10, STDDEV_SAMP(f10) AS std_f10,
       AVG(f11) AS mean_f11, STDDEV_SAMP(f11) AS std_f11
FROM data GROUP BY treatment ORDER BY treatment;
