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
19. **Checkpoint-aware XGBoost is now implemented.** Each WD4/7/10/15/20 model is
    trained from historical target-month examples using only information available
    through that checkpoint, including MTD Sell-In, elapsed/remaining working days,
    daily run-rate, projected EOM run-rate, authoritative target and MTD target
    achievement. The scored target month is excluded from model training.
20. Production prediction selects the latest available checkpoint model based on
    current working-day progress and falls back to the history-only XGBoost model
    when a checkpoint model is unavailable.
21. Regression tests cover checkpoint MTD leakage and exclusion of the scored target
    month from checkpoint training examples.

### Still required before production sign-off

- Add forecast reconciliation rules if forecasts are consumed together with a
  higher-level corporate aggregate.
- Add broader model/feature/output integration tests against representative
  synthetic fixtures before live database execution.
- Confirm the exact production column contract of `mv_ai_region_monthly` and
  `dimdate.networkeddays` against the live database.
- Run the full pipeline against representative production-like data and compare
  checkpoint-aware XGBoost versus the baseline/ETS/SARIMA candidates before
  accepting the new model selection behavior.
- Review strict interval coverage and interval width after the checkpoint-aware
  XGBoost forecasts are included in the candidate set.

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

## Checkpoint-aware XGBoost methodology

For a forecast made at checkpoint `C` in target month `T`, the XGBoost model receives
only information known by checkpoint `C`:

- closed-history lag and rolling features;
- target-month MTD Sell-In through checkpoint `C`;
- elapsed, remaining and total working days;
- observed daily run-rate and its EOM projection;
- the authoritative monthly target, which is known before the month starts;
- MTD target achievement; and
- the checkpoint identifier.

A separate pooled model is trained for each configured checkpoint. For every scored
target month `T`, its checkpoint model is trained only from historical target-month
examples with target month `< T`. This prevents both current-month actual leakage
and retrospective use of the scored month's outcome.

The production path uses the latest checkpoint model already reached by the current
month. If the current month has not reached WD4 or a checkpoint model is unavailable,
the existing history-only XGBoost model remains the safe fallback.

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
                              │                                 │
                              └──────── checkpoint XGBoost ────┘
                                               │
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
│   ├── test_rolling_intervals.py
│   └── test_xgb_checkpoint.py
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

## Code review notes (Claude, 2026-09-07)

Reviewed at commit `5907a30`. This is a genuine, substantial upgrade over the
V1 design — the review below is meant to get it to a state that actually
runs, not to relitigate the architecture. Nothing in the codebase was changed
as part of this review; only this section was added.

### 🔴 Blocking — the app cannot currently start or be tested

**1. `ImportError` in `backtest/rolling_intervals.py`.**
It imports `_xgb_checkpoint_features` from `backtest/xgb_checkpoint.py`, but
that function does not exist in that module (only `_as_key`,
`_checkpoint_date`, `_residual_stats`, `_history_features`, `_checkpoint_row`,
`_build_training_frame`, and `rolling_xgb_checkpoint_backtest` are defined
there). Confirmed by running `python -c "import backtest.rolling_intervals"`:

```
ImportError: cannot import name '_xgb_checkpoint_features' from 'backtest.xgb_checkpoint'
```

Because `forecast/train.py` imports `rolling_interval_backtest` from this
module at the top level, **`main.py` cannot run at all** right now — it fails
before any database connection is even attempted. It also means `pytest`
cannot collect the suite: `tests/test_rolling_intervals.py` fails to import,
which aborts collection for the *entire* run (confirmed: `pytest tests/`
reports "Interrupted: 1 error during collection", 0 of 15 tests executed).
**The GitHub Actions CI badge on this repo is red right now for this reason**
— every push and PR to `master` will fail the `test` job at the "Run tests"
step before a single test runs.

Fix options: either implement `_xgb_checkpoint_features` in
`xgb_checkpoint.py` (it looks like it was meant to build the current-month
feature row for a given checkpoint, similar to `_history_features` +
`_checkpoint_row` combined, for use by `rolling_intervals.py`), or have
`rolling_intervals.py` reuse `_history_features`/`_checkpoint_row` directly
if that covers the same need.

**2. Target-column mismatch crashes checkpoint XGBoost training once #1 is fixed.**
`train_xgboost_checkpoint()` in `models/xgboost_model.py` defaults to
`target_col="monthly_value"`. But the checkpoint training frames built by
`backtest/xgb_checkpoint.py` (`_checkpoint_row` → `_build_training_frame`)
name the realized value column `"actual"`, not `"monthly_value"` — confirmed
by inspecting `_checkpoint_row`'s return dict and by reproducing the crash
with synthetic data:

```
KeyError: ['monthly_value']
  at models/xgboost_model.py:_training_frame
     -> train_df.dropna(subset=cols + [target_col])
```

This hits two call sites with no override:
- `backtest/xgb_checkpoint.py::rolling_xgb_checkpoint_backtest` →
  `train_xgboost_checkpoint(train_frame)` (backtest path)
- `models/xgboost_model.py::train_xgboost_checkpoint_models` → same default,
  called from `forecast/train.py::run_training_pipeline` (production path)

So even after fixing the `ImportError`, training will still crash with
`KeyError: ['monthly_value']` on any dataset with enough history to reach
that step (reproduced locally with 30 months of synthetic history). Either
pass `target_col="actual"` explicitly at both call sites, or rename the key
in `_checkpoint_row`/`_history_features` to `monthly_value` for consistency
with the rest of the codebase (`to_monthly()` etc. all use `monthly_value`).

### 🟡 Worth checking

**3. `tests/test_xgb_checkpoint.py::test_checkpoint_training_frame_excludes_scored_target_month` fails against the current code**, independent of bugs #1/#2 — confirmed by running it in isolation. The fixture's `daily`/`calendar` frames only cover the target month (March 2026), but `_build_training_frame` needs daily + calendar coverage for *every* prior historical month back to the start of `monthly` (Jan 2024 in that fixture) to compute each historical checkpoint's MTD value. With only one month of calendar/daily data, every historical row's `_checkpoint_date()` lookup returns `None` and the training frame comes back empty, so `assert not frame.empty` fails. This is very likely just an under-specified fixture rather than a design problem — but it's a useful reminder that in production, `history_months=36` daily+calendar extraction (already done via `get_daily_sellin`/`get_calendar`) needs to actually deliver full-range data for this feature set to populate at all, not just the current month.

**4. `tests/test_historical_features.py::test_momentum_is_previous_month_growth` fails on float equality**: `assert march["mom_growth"] == 0.2` gets `0.19999999999999996`. Not a logic bug — `120/100 - 1` vs `150/120 - 1` chained through pandas `pct_change()` just isn't exactly representable — but worth switching to `pytest.approx(0.2)` so it isn't a recurring red herring in CI once #1/#2 are fixed.

**5. Silent fallback risk in ensemble weighting.** `models/ensemble.py::_row_backtest_weights` looks for `f"{model}_checkpoint_score"` columns (correctly produced by `backtest/model_selection.py::select_best_model` and merged into the prediction frame in `forecast/predict.py`). Right now, because `backtest_results` never successfully gets built (bugs #1/#2), this always silently falls through to `MODEL_CONFIG["ensemble_weights_default"]` with no warning logged. Once training actually completes end-to-end, it'd be worth adding a log line when the fallback path is taken, so a future silent misconfiguration (e.g. a renamed column) doesn't quietly degrade to static weights without anyone noticing.

**6. Runtime cost of the checkpoint backtest.** `rolling_xgb_checkpoint_backtest` and `build_oos_checkpoint_predictions` both retrain a fresh XGBoost model per (target month × checkpoint) combination for leakage-safety — with 36 months of history and 5 checkpoints that's up to ~180 model fits per run. Correct for the leakage guarantee, but worth timing against real production data volume; if `python main.py` needs to run daily, this is the part most likely to make that slow.

**7. Minor duplication.** `CANDIDATE_MODELS = ("baseline", "ets", "sarima", "xgboost")` is defined independently in both `models/ensemble.py` (tuple) and `backtest/model_selection.py` (list). Same four values in both places today; low risk, but consider importing from one location so they can't silently drift apart later.

### 🟢 Genuine improvements over the original V1 scaffold

- Dropping the branch grain and the `sellinascend.city ↔ v_sr_per_branch.kota`
  join guess entirely, and reading targets straight from
  `mv_ai_region_monthly`, removes the single riskiest unverified assumption
  in the V1 design. This is the right call.
- Using `dimdate.networkeddays` instead of guessing between
  `"workingday(5)"` and `"workingday(6)"` is a real fix, not a style choice.
- `WAPE` as the primary backtest metric instead of `MAPE` is more robust for
  months with a near-zero region total, where `MAPE` blows up.
- `complete_month_panel()` explicitly zero-filling missing months, and
  `validate_region_alignment()` rejecting silent region-code drift between
  actuals and targets, are exactly the kind of data-quality guardrails a V1
  scaffold like this needs before touching production data.
- The leakage discipline in the checkpoint design (both for point forecasts
  in `xgb_checkpoint.py` and for the P10/P50/P90 interval backtest in
  `rolling_intervals.py`) is conceptually sound and well-documented in the
  "Checkpoint-aware XGBoost methodology" section above — training strictly on
  `periode < target_month` and calibrating intervals strictly on
  `periode < T` for the same checkpoint is the correct walk-forward pattern.
  It just isn't reachable yet due to bugs #1/#2.
- Non-negative forecast guards (`max(pred, 0.0)`) throughout are a good
  defensive habit given this is a monetary forecast.
- Having `.github/workflows/ci.yml` at all, even though it's currently
  failing, is the right infrastructure to have in place before this goes
  near a real schedule.

### Suggested order of fixes

1. Implement `_xgb_checkpoint_features` (or remove the dependency) so the
   package imports again.
2. Fix the `target_col` mismatch in the two `train_xgboost_checkpoint(...)`
   call sites.
3. Re-run `pytest -q` and confirm all 15 tests pass (expect 2 more failures
   to fix per items #3–4 above).
4. Only then run `python main.py` against a real (or realistic synthetic)
   Postgres instance to sanity-check runtime and output rows before
   scheduling it anywhere.

## Code review notes, round 2 (Claude, 2026-09-10)

Reviewed at commit `82d66ca`. Nothing in the codebase was changed as part of
this review; only this section was added. This round was prompted by a real
traceback from running `python3 main.py` locally:

```
File ".../models/ensemble.py", line 50, in <module>
    def _residual_interval(row: pd.Series, weights: dict, p50: float) -> tuple[float, float]:
TypeError: 'type' object is not subscriptable
```

### ✅ Both blocking bugs from round 1 are confirmed fixed

- `_xgb_checkpoint_features` — the `ImportError` is gone; `backtest.rolling_intervals`, `forecast.predict`, and `main` all import cleanly now (verified by direct import).
- The `target_col` mismatch — `_build_training_frame()` in `backtest/xgb_checkpoint.py` now aliases `row["monthly_value"] = row["actual"]` before the row is used for training, so `train_xgboost_checkpoint(..., target_col="monthly_value")` no longer raises `KeyError`.
- **Full test suite: 17/17 passing** (up from 0/15 collectible last round — collection used to abort entirely). CI should be green again on the next push.
- `CANDIDATE_MODELS` has been centralized into `config.py` and most modules (`models/ensemble.py`, `backtest/ensemble.py`, `backtest/rolling_intervals.py`) now import it from there — addresses last round's minor duplication note, though see below, it's not fully done.

### 🔴 New finding: the traceback above is the *same bug class* recurring, and one more instance is still live

The `tuple[float, float]` the user hit in `models/ensemble.py` has already been
fixed in the current commit (now uses `from typing import Tuple` /
`Tuple[float, float]`, confirmed by reading the file — so pulling latest
should clear that specific error). But there is **one more, not-yet-fixed
occurrence of the identical problem**, and it sits directly on the
`main.py` import path:

```python
# data/validation.py, line 91 — no `from __future__ import annotations` in this file
def validate_calendar(calendar: pd.DataFrame, required_checkpoints: list[int] | None = None) -> pd.DataFrame:
```

`list[int]` needs Python 3.9+ (PEP 585) and `X | None` needs Python 3.10+
(PEP 604) *when evaluated eagerly*, which is exactly what happens here since
this file has no `from __future__ import annotations`. Traced the import
chain: `main.py` → `forecast/train.py` → `from data.validation import (...,
validate_calendar)` → this line executes at import time. **On whatever
Python version produced the original `tuple[float, float]` error (must be
< 3.9, since that's a PEP 585 feature too), this exact same `TypeError:
'type' object is not subscriptable` will reappear here** the moment the
previous fix is pulled — just one file later in the traceback. Confirmed
by scanning every `.py` file in the repo for `list[`, `dict[`, `tuple[`,
`set[`, or `X | None` outside files that opt into postponed evaluation via
`from __future__ import annotations`; this is the only remaining hit.

Two ways to close this out for good rather than one file at a time:
1. Add `from __future__ import annotations` to `data/validation.py` (and
   ideally to every module in the project, as cheap insurance against the
   next one of these), **or**
2. Rewrite the signature using `typing.Optional[typing.List[int]]`
   consistent with how `models/ensemble.py` and `models/xgboost_model.py`
   were just fixed, **or**
3. Pin and document a minimum Python version (3.10+) in `requirements.txt`
   / README and stop worrying about this class of bug entirely. Right now
   no minimum Python version is declared anywhere in the repo (no
   `runtime.txt`, `pyproject.toml`, `python_requires`, or README mention),
   so there's nothing stopping a contributor from reintroducing this same
   issue in a new file.

### 🟡 Smaller items

- **`CANDIDATE_MODELS` centralization is incomplete.** `config.py` now
  defines it once, and `models/ensemble.py`, `backtest/ensemble.py`, and
  `backtest/rolling_intervals.py` import it from there — but
  `backtest/model_selection.py:5` and `forecast/train.py:26` still each
  define their own local copy of the same four-model tuple/list instead of
  importing it. Low risk today since all four definitions currently agree,
  but it's the kind of thing that silently drifts later.
- **Operational note on data depth, not a bug**: `FORECAST_CONFIG` requires
  `min_history_months: 24` and `backtest_min_train_months: 24` — a region
  needs at least 24 closed months of history before it's eligible for a
  forecast at all (`run_training_pipeline` filters regions below that
  threshold out of `monthly` entirely). Worth double-checking that your
  real `sellinascend` history actually goes back that far for every region
  you care about forecasting — any region with less history will silently
  produce zero output rows rather than an error.
- The region-mapping join was changed to
  `sellinascend."Customer Area" = vt_sr_per_rsmasw.kota`, with a code
  comment stating it's been validated as one-to-one against production
  data. Good — this replaces the `city`/`kota` guess flagged in round 1
  with something you've apparently verified directly; just flagging that
  I can't independently confirm that from here, so it's worth keeping that
  validation claim honest as the mapping table evolves.

### 🟢 New since round 1, worth calling out

- `backtest/ensemble.py::audit_ensemble` is a nice addition: it compares the
  production ensemble against each candidate model on the *same* common
  out-of-sample (target month, checkpoint) pairs, so the "is the ensemble
  actually better than just picking the best single model" question has a
  fair, leakage-safe answer instead of an assumption.
- `models/ensemble.py::_row_backtest_weights` now logs when it falls back to
  static default weights instead of silently doing so — directly addresses
  the round-1 concern about that failure mode being invisible.
- Checkpoint-specific residual interval calibration
  (`{model}_wd{checkpoint}_residual_q10/q90`) with a documented fallback
  chain (checkpoint-specific → pooled → conservative spread-based) is a
  sensible way to keep P10/P90 honest as the month progresses.

### Suggested next step

Just the `data/validation.py` fix — either add
`from __future__ import annotations` there (fastest, matches nothing else
needs to change) or convert to `typing.Optional`/`typing.List` for
consistency with the two files already fixed this way. After that,
`python3 main.py` should get past import time; whether it completes
successfully end-to-end against your real Postgres instance is the next
thing to verify.

## Code review notes, round 3 (Claude, 2026-09-11)

Reviewed at commit `80d9c87`. Nothing in the codebase was changed as part
of this review; only this section was added. This round covers the new
`sql/views/*_v2.sql` layer — the deterministic V2 diagnostics/insight
contract that a downstream AI insight agent reads from.

### ✅ Round 2 issue confirmed fixed

`data/validation.py` now has `from __future__ import annotations` as its
first import. **Full test suite: 23/23 passing.** The Python-version class
of bug from rounds 1–2 appears fully closed out now.

### 🔴 Most important finding: GM/CEO uncertainty bounds are statistically invalid as written

`v_ai_gm_monthly_diagnostics_v2.sql` and `v_ai_ceo_monthly_diagnostics_v2.sql`
both compute `forecast_p10` and `forecast_p90` at the aggregate level with:

```sql
SUM(forecast_p10) AS forecast_p10,
SUM(forecast_p50) AS forecast_p50,
SUM(forecast_p90) AS forecast_p90,
```

**Summing quantiles across regions is not the same as the quantile of the
sum.** P50 is fine to sum (expectation is linear, so summing medians is a
reasonable approximation of the sum's median under typical conditions). P10
and P90 are not — they're tail bounds, and unless every region's forecast
error is perfectly positively correlated, some of the regions that miss low
will be offset by others that miss high. The true P10/P90 of a GM's *total*
sell-in is narrower than the sum of each region's individual P10/P90,
because independent (or even partially independent) errors partially
diversify away at the portfolio level. Summing the raw bounds instead
systematically **overstates uncertainty** at GM level, and — because the CEO
view aggregates on top of the already-summed GM/region numbers — that bias
compounds a second time at CEO level. In practice this means
`forecast_uncertainty_pct` will read wider than it actually is the higher
up the hierarchy you go, which is exactly the opposite of what you'd
intuitively expect (aggregates should usually look *more* certain, not
less).

Notably, this is the exact gap your own README already flagged as
outstanding, in "Still required before production sign-off": *"Add forecast
reconciliation rules if forecasts are consumed together with a higher-level
corporate aggregate."* The GM/CEO views were built before that item was
addressed, so they currently do the naive thing.

Ways to fix this, roughly in order of effort:
1. **Cheapest real fix**: treat regional forecast errors as independent and
   aggregate the *half-widths* in quadrature instead of summing the raw
   bounds — i.e. `gm_p50 ± sqrt(Σ (region_half_width)^2)` instead of
   `Σ region_p10 .. Σ region_p90`. Still an approximation (errors are
   probably not fully independent — e.g. a national sales holiday affects
   every region at once), but it's a one-line-per-view change that stops
   actively overstating uncertainty.
2. **Better**: keep the OOS residuals per region from the Python side
   (`backtest/rolling_intervals.py` already computes these) and combine
   them via a correlated bootstrap/Monte Carlo at GM/CEO grain in
   `forecast_engine` itself, writing native GM/CEO-level P10/P50/P90
   alongside the region-level ones into `forecast_sellin_eom` (or a sibling
   table) — this also naturally captures cross-region correlation instead
   of assuming independence.
3. **At minimum for now**: rename `forecast_uncertainty_pct` at GM/CEO
   level in the view comment/docs to make clear it's a conservative
   upper-bound approximation, not a calibrated interval — so nobody
   reading the CEO dashboard treats "42% uncertainty" as a validated
   statistical statement.

### 🟡 NULL handling: a region with no forecast can get silently mislabeled as CRITICAL

In `v_ai_region_monthly_insight_v2.sql`, every scenario/priority `CASE`
compares `achievement_pct_forecast` (or its GM/CEO equivalents) with `>=`.
If that value is `NULL` — which happens whenever `target_sellin` is zero or
missing, or whenever `forecast_p50` itself is `NULL` — every `WHEN`
condition evaluates to `NULL` (not true), so execution falls through to the
`ELSE` branch:

```sql
CASE
    WHEN b.achievement_pct_forecast >= 100 THEN 'TARGET_ACHIEVED'
    ...
    ELSE 'CRITICAL'
END AS performance_scenario
```

A region with genuinely missing data — not a real crisis, just no signal —
gets labeled `'CRITICAL'` (and separately, via the same NULL-passthrough
logic in the priority `CASE`, `'LOW'` priority). `'CRITICAL'`
performance + `'LOW'` priority is an internally inconsistent, confusing pair
for whatever narrates this to a CEO or GM. Worth adding an explicit
`WHEN achievement_pct_forecast IS NULL THEN 'NO_FORECAST_DATA'` branch (and
a matching `'REVIEW'` or similar priority) so missing data reads as missing
data, not as a false crisis signal.

Related and arguably more important: this view is driven `FROM
dwh_prod.forecast_sellin_eom`, and (per the V2 contract already documented
above) a region needs **24 closed months of history** to get a row there at
all. Any region below that threshold doesn't just get a NULL scenario — it
**doesn't appear in this view, or any of the GM/CEO rollups, at all**. If
even one region silently falls below the eligibility bar (new region, data
gap, etc.), nobody looking at the CEO dashboard has any way to know a
region went missing from the picture, since there's no explicit coverage
check anywhere in this SQL layer. A small `v_ai_forecast_coverage_v2` view
— total known regions from `mv_ai_region_monthly` vs. regions actually
present in `forecast_sellin_eom` for the current period — would make gaps
visible instead of silent.

### 🟡 Unverified join: `dwh_prod.m_sales_org_hierarchy`

Both the region view (for `regionname`) and the GM view (for
`gm_code`/`gm_name`, and for the region→GM rollup itself) join against
`dwh_prod.m_sales_org_hierarchy ... AND h.is_active = TRUE`. This table
wasn't part of any schema shared in earlier rounds, so I can't verify it
from here — but the failure mode to check for specifically is **fan-out**:
if any `regioncode` has more than one `is_active = TRUE` row in that table
(a data-entry duplicate, or a region mid-transition between GMs), the GM
view's `JOIN` will silently duplicate that region's `SUM(total_sellin)` /
`SUM(target_sellin)` into the aggregate, inflating the GM's (and therefore
the CEO's) numbers with no error raised anywhere. Worth adding a `UNIQUE`
constraint or partial unique index on `(regioncode) WHERE is_active`, or at
minimum a one-off `GROUP BY regioncode HAVING COUNT(*) > 1` sanity query
before trusting the rollups.

### 🟢 What's genuinely good here

- The core idea — one deterministic SQL contract
  (`v_ai_forecast_insight_input_v2`) that an LLM insight layer reads facts
  from and is explicitly told not to recalculate — is the right shape for
  keeping an LLM's output grounded in real numbers instead of letting it
  narrate its own arithmetic.
- Region → GM → CEO is a clean, consistent `UNION ALL` hierarchy with a
  stable column contract across all three levels — good for whatever
  consumes this next.
- Deriving `performance_scenario`/`forecast_scenario`/`priority` purely from
  the already-validated `forecast_p10/p50/p90` (rather than reintroducing a
  second, competing momentum/run-rate calculation in SQL) means there's
  exactly one forecasting method in the whole system, not two disagreeing
  ones. That's a real architectural win and avoids a whole class of "why do
  the two dashboards show different numbers" problems down the line.
- `shortfall_contribution_pct` and `largest_shortfall_region/gm` are useful,
  genuinely actionable additions for a GM/CEO reading this — "which region
  is driving the miss" is exactly the kind of thing raw percentages don't
  tell you on their own.

### If the goal is "more professional insight, more accurate forecast" — suggested priorities

Beyond the fixes above, in rough order of leverage:

1. **Fix the GM/CEO interval aggregation first** (see 🔴 above) — right now
   it's the one place where a number flowing into an executive-facing
   insight is measurably wrong, not just approximate.
2. **Track forecast accuracy as its own monitored metric.** You already
   compute WAPE per region per checkpoint during backtesting
   (`backtest/rolling.py`, `backtest/xgb_checkpoint.py`) — persist that
   history (e.g. `forecast_accuracy_history`) so a region's *recent, real*
   track record can be cited in its own insight ("this region's forecast
   has been within X% of actual for the last 3 months" vs. a region with
   a spotty history). That turns "trust me" into an auditable claim, which
   is a meaningfully more professional insight than a bare percentage with
   no stated confidence behind it — and it gives you a live signal for when
   a region's model quietly degrades and needs re-tuning.
3. **Add exogenous features to lift real accuracy** (carried over from
   round 1, still the highest-leverage unclaimed improvement): Indonesian
   holiday/Ramadan calendar, known promo/campaign flags, and sell-out or
   inventory signals as XGBoost features, not just sell-in's own history.
   Working-day-adjusted run-rate gets you far, but it can't see a demand
   spike or a promo push coming.
4. **Add a coverage view** (see 🟡 above) so "this region has no forecast"
   is a visible, queryable fact rather than a silent absence — this is
   cheap and directly protects the credibility of the CEO-level insight the
   first time a real gap happens.
5. **Consider native quantile models for P10/P90** (e.g. LightGBM/XGBoost
   quantile objectives, or NGBoost) instead of deriving intervals purely
   from historical point-forecast residuals — especially valuable for
   regions near the 24-month eligibility threshold, where the residual
   history used for calibration is itself thin.
