-- Executed against the typed, full-file `data` view; no LIMIT or sampling.
SELECT COUNT(*) AS n,
       COUNT(*) FILTER (WHERE f0 IS NULL OR NOT isfinite(f0)) AS invalid_f0,
       COUNT(*) FILTER (WHERE f1 IS NULL OR NOT isfinite(f1)) AS invalid_f1,
       COUNT(*) FILTER (WHERE f2 IS NULL OR NOT isfinite(f2)) AS invalid_f2,
       COUNT(*) FILTER (WHERE f3 IS NULL OR NOT isfinite(f3)) AS invalid_f3,
       COUNT(*) FILTER (WHERE f4 IS NULL OR NOT isfinite(f4)) AS invalid_f4,
       COUNT(*) FILTER (WHERE f5 IS NULL OR NOT isfinite(f5)) AS invalid_f5,
       COUNT(*) FILTER (WHERE f6 IS NULL OR NOT isfinite(f6)) AS invalid_f6,
       COUNT(*) FILTER (WHERE f7 IS NULL OR NOT isfinite(f7)) AS invalid_f7,
       COUNT(*) FILTER (WHERE f8 IS NULL OR NOT isfinite(f8)) AS invalid_f8,
       COUNT(*) FILTER (WHERE f9 IS NULL OR NOT isfinite(f9)) AS invalid_f9,
       COUNT(*) FILTER (WHERE f10 IS NULL OR NOT isfinite(f10)) AS invalid_f10,
       COUNT(*) FILTER (WHERE f11 IS NULL OR NOT isfinite(f11)) AS invalid_f11,
       COUNT(*) FILTER (WHERE treatment IS NULL OR treatment NOT IN (0, 1)) AS invalid_treatment,
       COUNT(*) FILTER (WHERE visit IS NULL OR visit NOT IN (0, 1)) AS invalid_visit,
       COUNT(*) FILTER (WHERE conversion IS NULL OR conversion NOT IN (0, 1)) AS invalid_conversion,
       COUNT(*) FILTER (WHERE exposure IS NULL OR exposure NOT IN (0, 1)) AS invalid_exposure
FROM data;
