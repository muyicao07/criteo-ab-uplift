-- Sufficient statistics for paired, finite-population outcome simulations.
SELECT treatment, visit, conversion, COUNT(*) AS n
FROM data
GROUP BY treatment, visit, conversion
ORDER BY treatment, visit, conversion;
