"""
Central configuration for the forecast engine.

IMPORTANT - read this before running anything.
Your shared DDL only fully defines `dwh_prod.sellinascend` and
`dwh_prod.dimdate`. The two materialized views reference two other
objects whose full column lists were NOT in what you shared:
    - dwh_prod.v_sr_per_branch   (branch/region dimension)
    - dwh_prod.v_t_sellin_ascend_sellout_eska (region-level monthly
      target vs actual sell-in/sell-out)

Everything below that touches those objects is a best-guess based on
the columns visible in the CREATE MATERIALIZED VIEW statements you
sent. If a query fails or returns 0 rows, the join/column name here
is almost certainly the thing to fix - check SOURCE below first.
"""
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": os.getenv("PG_PORT", "5432"),
    "dbname": os.getenv("PG_DB", "postgres"),
    "user": os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", ""),
}

# quote_plus URL-encodes the password so special chars like '@', '/', ':' in
# it don't break the connection-string parsing (SQLAlchemy treats '@' as the
# user:password@host delimiter).
SQLALCHEMY_URL = (
    f"postgresql+psycopg2://{DB_CONFIG['user']}:{quote_plus(DB_CONFIG['password'])}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

# ---------------------------------------------------------------------------
# SOURCE: every table/view/column name the pipeline touches, in one place.
#
# ASSUMPTIONS you should verify against your real DB:
#   1. dwh_prod.sellinascend has one row per invoice line, dated by
#      `invoicedate`, valued by "Line Total After Tax (Local)".
#   2. sellinascend.city can be joined to dwh_prod.vt_sr_per_rsmasw.kota to
#      attach branchcode/branchname/regioncode/regionname to each invoice
#      line. This is inferred from the vt_sr_per_rsmasw materialized view,
#      which is keyed by (kota, regioncode, sellingpointname). If your real
#      join key is different (e.g. "Customer Region" text <-> regionname
#      text, or "Customer Code" <-> a customer-to-branch mapping table you
#      have elsewhere), change branch_join_col_fact / branch_join_col_dim.
#   3. Monthly sell-in TARGETS only exist at REGION grain, in
#      v_t_sellin_ascend_sellout_eska (periode, regioncode, regionname,
#      totaltargetsellin, totalrealsellin, ...). There is no branch-level
#      target anywhere in the schema you shared, so branch targets are
#      allocated pro-rata from the region target using each branch's
#      trailing historical share of regional sell-in
#      (see data/aggregation.py::allocate_branch_targets). If you DO have
#      a branch-level target table, point BRANCH TARGET logic at it instead
#      and skip the allocation step.
#   4. dimdate has a working-day flag; two candidate columns exist in your
#      DDL - "workingday(5)" (text) and "workingday(6)" (float8). Default
#      below is (5); switch to (6) if that's the one your BI team actually
#      uses to mean "is a working day".
# ---------------------------------------------------------------------------

SOURCE = {
    "fact_table": "dwh_prod.sellinascend",
    "date_dim": "dwh_prod.dimdate",
    "branch_dim": "dwh_prod.vt_sr_per_rsmasw",       # deduped branch/region map
    "region_target_view": "dwh_prod.v_t_sellin_ascend_sellout_eska",
    "value_col": '"Line Total After Tax (Local)"',
    "invoice_date_col": "invoicedate",
    "branch_join_col_fact": "city",
    "branch_join_col_dim": "kota",
    "working_day_col": '"workingday(5)"',            # or '"workingday(6)"'
}

FORECAST_CONFIG = {
    "target_metric": "total_sellin_value",           # Rp value vs target_sellin
    "grain": ["regioncode", "branchcode"],            # hierarchical: region + branch
    "min_history_months": 6,                          # min months of closed history to backtest ETS/SARIMA
    "history_months_for_features": 12,
    "current_month_lookback_days": 400,               # how far back raw data is pulled (~13 months)
}

MODEL_CONFIG = {
    "ets": {"seasonal": None, "trend": "add"},
    "sarima": {"order": (1, 1, 1), "seasonal_order": (0, 1, 1, 12)},
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
    },
    # used when a series has both ETS/SARIMA and XGBoost predictions available;
    # missing model outputs are dropped and the rest re-normalized (see models/ensemble.py)
    "ensemble_weights_default": {"baseline": 0.2, "ets": 0.25, "sarima": 0.25, "xgboost": 0.3},
}

OUTPUT_CONFIG = {
    "forecast_table": "dwh_prod.forecast_sellin_eom",
    "write_mode": "upsert",
}
