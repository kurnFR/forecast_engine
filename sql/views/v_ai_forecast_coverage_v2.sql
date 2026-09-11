-- V2 forecast coverage QA
-- Makes forecast eligibility gaps visible instead of silently disappearing from
-- GM/CEO rollups. This is a QA/monitoring view, not a forecasting calculation.

CREATE OR REPLACE VIEW dwh_prod.v_ai_forecast_coverage_v2 AS
WITH expected AS (
    SELECT
        periode,
        regioncode
    FROM dwh_prod.mv_ai_region_monthly
    GROUP BY periode, regioncode
), forecasted AS (
    SELECT
        periode,
        regioncode
    FROM dwh_prod.forecast_sellin_eom
    GROUP BY periode, regioncode
), coverage AS (
    SELECT
        e.periode,
        COUNT(*) AS expected_region_count,
        COUNT(f.regioncode) AS forecast_region_count,
        COUNT(*) FILTER (WHERE f.regioncode IS NULL) AS missing_region_count,
        ARRAY_AGG(e.regioncode ORDER BY e.regioncode)
            FILTER (WHERE f.regioncode IS NULL) AS missing_regioncodes
    FROM expected e
    LEFT JOIN forecasted f
      ON e.periode = f.periode
     AND e.regioncode = f.regioncode
    GROUP BY e.periode
), hierarchy_qa AS (
    SELECT
        regioncode,
        COUNT(*) FILTER (WHERE is_active = TRUE) AS active_hierarchy_rows
    FROM dwh_prod.m_sales_org_hierarchy
    GROUP BY regioncode
)
SELECT
    c.periode,
    c.expected_region_count,
    c.forecast_region_count,
    c.missing_region_count,
    c.missing_regioncodes,
    COUNT(h.regioncode) FILTER (WHERE h.active_hierarchy_rows > 1)
        AS hierarchy_duplicate_region_count,
    CASE
        WHEN c.missing_region_count = 0
         AND COUNT(h.regioncode) FILTER (WHERE h.active_hierarchy_rows > 1) = 0
            THEN 'OK'
        WHEN COUNT(h.regioncode) FILTER (WHERE h.active_hierarchy_rows > 1) > 0
            THEN 'HIERARCHY_DUPLICATE'
        ELSE 'FORECAST_COVERAGE_GAP'
    END AS coverage_status
FROM coverage c
LEFT JOIN expected e
  ON c.periode = e.periode
LEFT JOIN hierarchy_qa h
  ON e.regioncode = h.regioncode
GROUP BY
    c.periode,
    c.expected_region_count,
    c.forecast_region_count,
    c.missing_region_count,
    c.missing_regioncodes;
