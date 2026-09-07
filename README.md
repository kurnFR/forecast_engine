# Sell-in End-of-Month Forecast Engine

Forecasts end-of-month sell-in achievement (Rp value vs `target_sellin`) for
every **region + branch**, and writes the result to
`dwh_prod.forecast_sellin_eom` in Postgres.

## What it does

1. **Extract** daily sell-in from `dwh_prod.sellinascend`, joined to a
   branch/region dimension, plus region-level monthly targets and a
   working-day calendar from `dwh_prod.dimdate`.
2. **Aggregate** to monthly per (region, branch); allocate region targets
   down to branches pro-rata (no branch-level target exists in the schema
   you shared — see `config.py` if you actually have one).
3. **Feature engineer**: month-to-date run-rate, working-days elapsed/
   remaining, lags, month-over-month growth, seasonality index.
4. **Model** each series four ways:
   - `baseline`: MTD value ÷ elapsed working days × total working days
   - `ets`: Holt exponential smoothing
   - `sarima`: seasonal ARIMA
   - `xgboost`: gradient boosting trained across all series' engineered features
5. **Backtest** ETS/SARIMA/naive on every historical month (rolling-origin)
   so you can see which model actually predicts well for which series.
6. **Ensemble** the four forecasts (weighted average, configurable in
   `config.py`), compute forecast achievement % vs target, and
   **upsert into Postgres**.

## ⚠️ Before you run this — check your schema assumptions

Your shared DDL fully defines `sellinascend` and `dimdate`, but the two
materialized views reference `v_sr_per_branch` and
`v_t_sellin_ascend_sellout_eska`, whose full column lists weren't included.
Everything the pipeline assumes about those two objects — plus the
`sellinascend.city ↔ vt_sr_per_rsmasw.kota` join used to attach a branch to
each invoice line — is documented and centralized at the top of
**`config.py`** in the `SOURCE` dict. If a query returns 0 rows or errors
out, that's the first place to look. In particular:

- If your real customer→branch mapping is different (e.g. by
  `"Customer Region"` or `"Customer Code"` instead of `city`), update
  `branch_join_col_fact` / `branch_join_col_dim`.
- If you *do* have a branch-level target table, swap out
  `data/aggregation.py::allocate_branch_targets` for a direct join instead
  of the pro-rata allocation.
- `"workingday(5)"` vs `"workingday(6)"` in `dimdate` — pick whichever one
  your team actually uses to mean "is a working day" (`SOURCE["working_day_col"]`).

## Setup

```bash
cd forecast_engine
python -m venv venv && source venv/bin/activate   # or your preferred env manager
pip install -r requirements.txt
cp .env.example .env   # then fill in your real Postgres credentials
```

## Run

```bash
python main.py
```

This trains on history, backtests, generates the current month's
end-of-month forecast for every (region, branch), and upserts it into
`dwh_prod.forecast_sellin_eom`.

Recommended: schedule this daily (cron / Airflow) so the forecast
tightens as the month progresses and more actuals come in.

## Output table

```sql
dwh_prod.forecast_sellin_eom (
    regioncode, branchcode, periode,
    mtd_value, elapsed_working_days, remaining_working_days, total_working_days,
    forecast_baseline, forecast_ets, forecast_sarima, forecast_xgboost, forecast_ensemble,
    branch_target_sellin, achievement_pct_forecast, generated_at,
    PRIMARY KEY (regioncode, branchcode, periode)
)
```

Re-running for the same month updates the row (upsert on
`regioncode, branchcode, periode`) so you always have the latest forecast
per month, plus history once the month rolls over.

## Project layout

```
forecast_engine/
├── config.py            # DB creds (.env) + all schema/join assumptions
├── db.py                # SQLAlchemy engine + query helpers
├── data/
│   ├── extract.py       # raw SQL pulls
│   ├── validation.py    # null/dupe/sanity checks
│   └── aggregation.py   # daily -> monthly, target allocation
├── features/
│   ├── current_month.py # MTD / run-rate
│   ├── historical.py    # lags, growth, seasonality
│   └── working_day.py   # working-day counting
├── models/
│   ├── baseline.py
│   ├── ets.py
│   ├── sarima.py
│   ├── xgboost_model.py
│   └── ensemble.py
├── backtest/
│   ├── rolling.py        # rolling-origin backtest
│   ├── metrics.py         # MAPE/MAE/RMSE/bias
│   └── model_selection.py # best model per series
├── forecast/
│   ├── train.py
│   └── predict.py
├── output/
│   └── postgres.py       # upsert into forecast_sellin_eom
└── main.py
```

## Tuning

- `config.FORECAST_CONFIG["grain"]` — change to `["regioncode"]` only, or
  add a third level, if you decide branch grain is too noisy for XGBoost/SARIMA.
- `config.MODEL_CONFIG["ensemble_weights_default"]` — reweight based on what
  `backtest_results` shows performs best in your data (e.g. if SARIMA
  consistently wins, raise its weight).
- Series with under `min_history_months` of history will only get a
  baseline + XGBoost forecast (ETS/SARIMA return `None` and are excluded
  from that series' ensemble automatically).
