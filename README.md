# Sell-In End-of-Month Forecast Engine — V2

Production-oriented forecast engine for **monthly Sell-In value at region grain**.
The current design is locked to `regioncode × periode` and is intended to
forecast the current month's end-of-month Sell-In and achievement versus the
authoritative regional target.

## V2 contract

- **Forecast grain:** `regioncode × month`
- **History:** up to **36 months**; a region needs at least **24 closed months**
  to be eligible for the full model comparison
- **Calendar:** working-day progress comes from `networkeddays`
- **Target:** read directly from `dwh_prod.mv_ai_region_monthly`
- **Backtest checkpoints:** **WD4, WD7, WD10, WD15, WD20**
- **Leakage rule:** a checkpoint may use only data available on or before that
  working day; the target month itself must never be used to train its forecast
- **Candidate models:** Historical/Run-rate Baseline, ETS, SARIMA, XGBoost
- **Selection:** driven by leakage-safe checkpoint WAPE
- **Ensemble:** inverse-WAPE backtest weighting is used for P50 when valid
  checkpoint scores are available
- **Uncertainty:** P10 / P50 / P90 use out-of-sample residual quantiles, with a
  safe fallback when residual history is unavailable
- **Output:** upserted by `(regioncode, periode)`
- **QA:** source/mapping/calendar validation is required before production runs

## Current implementation status

### Implemented in V2

1. Region-month configuration and 36-month history window.
2. Direct region target extraction from `mv_ai_region_monthly`.
3. Complete monthly panel so missing sales months become explicit zero values.
4. Leakage-safe WD4/7/10/15/20 rolling backtest for baseline, ETS, SARIMA and XGBoost.
5. XGBoost target-month models are pooled across eligible regions and trained once
   per target month, rather than redundantly once per region.
6. Historical XGBoost features are aligned with the working-day calendar,
   including target-month total working days at prediction time.
7. WAPE, MAE, RMSE and bias diagnostics for model comparison.
8. Region-month prediction path with current-MTD baseline and closed-history
   ETS/SARIMA/XGBoost candidates.
9. Backtest-derived inverse-WAPE ensemble weighting for P50.
10. Out-of-sample residual q10/q90 collection for every candidate model.
11. Residual-calibrated P10/P50/P90 with non-negative forecast safeguards.
12. PostgreSQL output primary key changed to `(regioncode, periode)`.
13. Daily and target-source validation updated to the locked region-month contract,
    including duplicate target-key detection.
14. Formal region-alignment QA rejects actual regions missing from the authoritative
    target source and rejects target-only regions.
15. Calendar QA validates unique/consecutive dates, binary working-day flags,
    and enough working days for WD4/7/10/15/20.
16. Automated validation tests and GitHub Actions CI have been added.
17. A **strict leakage-safe interval backtest** now rebuilds OOS candidate forecasts
    and calibrates each target month's P10/P90 only from OOS residuals belonging to
    earlier target months for the same region and checkpoint. The target month's own
    residual is never available to its interval calibration.
18. The strict interval backtest is integrated into the training pipeline as
    `interval_backtest_results`, with interval width and normalized width diagnostics.

### Still required before production sign-off

- Add **checkpoint-safe MTD/run-rate features to XGBoost** if testing confirms that
  the history-only XGBoost candidate should compete directly with the MTD baseline.
- Add forecast reconciliation rules if forecasts are consumed together with a
  higher-level corporate aggregate.
- Add broader model/feature/output integration tests against representative
  synthetic fixtures before live database execution.
- Confirm the exact production column contract of `mv_ai_region_monthly` and
  `dimdate.networkeddays` against the live database.
- Run the full pipeline against representative production-like data and review
  interval coverage before accepting the P10/P90 calibration operationally.

## Interval backtest methodology

The production residual pool can legitimately use historical OOS residuals after
those target months have become past data. That is different from evaluating
whether an interval method would have worked at the time of a historical forecast.

For strict interval validation, the engine therefore performs a second rolling
pass:

1. Build candidate forecasts for each eligible `regioncode × target month ×
   checkpoint` using only information available at that checkpoint.
2. For target month `T`, restrict calibration data to target months `< T`.
3. Keep calibration within the same region and checkpoint.
4. Derive inverse-WAPE P50 weights from those earlier OOS rows only.
5. Derive each candidate's residual q10/q90 from those earlier OOS rows only.
6. Combine the candidate residual ranges around the leakage-safe P50.
7. Skip an interval when no prior OOS residual calibration exists rather than
   fabricating a validated interval for the first eligible target.

This makes historical interval coverage a genuine out-of-time diagnostic rather
than a retrospective in-sample calibration check.

## Data flow

```text
PostgreSQL
   │
   ├── sellinascend ── daily region Sell-In ──┐
   ├── dimdate (networkeddays) ────────────────┤
   └── mv_ai_region_monthly (target) ──────────┤
                                               ▼
                                    validation + monthly panel
                                               │
                              ┌────────────────┴────────────────┐
                              ▼                                 ▼
                       closed history                 current MTD + WD
                              │                                 │
                    ETS / SARIMA / XGBoost              run-rate baseline
                              └────────────────┬────────────────┘
                                               ▼
                                 backtest-weighted ensemble
                                               │
                                  OOS residual calibration
                                               │
                                               ▼
                                      P10/P50/P90 forecast
                                               │
                                               ▼
                                  forecast_sellin_eom
```

## Repository layout

```text
forecast_engine/
├── config.py
├── db.py
├── data/
│   ├── extract.py
│   ├── validation.py
│   └── aggregation.py
├── features/
│   ├── current_month.py
│   ├── historical.py
│   └── working_day.py
├── models/
│   ├── baseline.py
│   ├── ensemble.py
│   ├── ets.py
│   ├── sarima.py
│   └── xgboost_model.py
├── backtest/
│   ├── rolling.py
│   ├── rolling_intervals.py
│   ├── xgb_checkpoint.py
│   ├── intervals.py
│   ├── metrics.py
│   └── model_selection.py
├── forecast/
│   ├── train.py
│   └── predict.py
├── output/
│   └── postgres.py
├── tests/
│   ├── test_validation.py
│   └── test_rolling_intervals.py
├── .github/workflows/ci.yml
└── main.py
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Configure `PG_HOST`, `PG_PORT`, `PG_DB`, `PG_USER` and `PG_PASSWORD` in `.env`.

## Test

```bash
pytest -q
```

Tests are also executed automatically by GitHub Actions on pushes to `master`
and pull requests targeting `master`.

## Run

```bash
python main.py
```

The engine trains from closed history, evaluates historical checkpoints,
generates the current region-month forecast, and upserts the result to
`dwh_prod.forecast_sellin_eom`.

## Output schema

```sql
CREATE TABLE dwh_prod.forecast_sellin_eom (
    regioncode text NOT NULL,
    periode date NOT NULL,
    mtd_value numeric(23,4),
    elapsed_working_days int,
    remaining_working_days int,
    total_working_days int,
    forecast_baseline numeric(23,4),
    forecast_ets numeric(23,4),
    forecast_sarima numeric(23,4),
    forecast_xgboost numeric(23,4),
    forecast_p10 numeric(23,4),
    forecast_p50 numeric(23,4),
    forecast_p90 numeric(23,4),
    target_sellin numeric(23,4),
    achievement_pct_forecast numeric(9,4),
    generated_at timestamp NOT NULL DEFAULT now(),
    PRIMARY KEY (regioncode, periode)
);
```

## Safe update policy

Changes to `master` are applied sequentially using the current file/blob SHA.
No force-push or blind overwrite is used. Each write creates a normal Git
commit whose parent is the latest verified branch state.
