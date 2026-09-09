"""Central configuration for the V2 Sell-In forecast engine.

V2 is intentionally region-month based. The forecast grain is
(regioncode, periode), history is up to 36 months, and monthly targets are
read directly from mv_ai_region_monthly. All forecast/backtest code should
consume these settings rather than embedding source-table assumptions.
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

SQLALCHEMY_URL = (
    f"postgresql+psycopg2://{DB_CONFIG['user']}:{quote_plus(DB_CONFIG['password'])}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

SOURCE = {
    "fact_table": os.getenv("FORECAST_FACT_TABLE", "dwh_prod.sellinascend"),
    "date_dim": os.getenv("FORECAST_DATE_DIM", "dwh_prod.dimdate"),
    "region_target_view": os.getenv("FORECAST_TARGET_TABLE", "dwh_prod.mv_ai_region_monthly"),
    "region_mapping_table": os.getenv("FORECAST_REGION_MAPPING_TABLE", "dwh_prod.vt_sr_per_rsmasw"),
    "mapping_fact_key_col": os.getenv("FORECAST_MAPPING_FACT_KEY_COL", '"Customer Area"'),
    "mapping_key_col": os.getenv("FORECAST_MAPPING_KEY_COL", "kota"),
    "mapping_region_col": os.getenv("FORECAST_MAPPING_REGION_COL", "regioncode"),
    "value_col": '"Line Total After Tax (Local)"',
    "invoice_date_col": "invoicedate",
    "region_col": "regioncode",
    "networked_days_col": "networkeddays",
}

FORECAST_CONFIG = {
    "target_metric": "sellin_value",
    "grain": ["regioncode"],
    "history_months": 36,
    "min_history_months": 24,
    "as_of": None,
    "backtest_checkpoints": [4, 7, 10, 15, 20],
    "backtest_min_train_months": 24,
    "forecast_horizon_months": 1,
}

CANDIDATE_MODELS = ("baseline", "ets", "sarima", "xgboost")

MODEL_CONFIG = {
    "ets": {
        "trend": "add",
        "seasonal": "add",
        "seasonal_periods": 12,
    },
    "sarima": {
        "order": (1, 1, 1),
        "seasonal_order": (0, 1, 1, 12),
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
    },
    "ensemble_weights_default": {
        "baseline": 0.20,
        "ets": 0.25,
        "sarima": 0.25,
        "xgboost": 0.30,
    },
}

BACKTEST_CONFIG = {
    "metric_primary": "wape",
    "metrics": ["mae", "wape", "rmse", "bias"],
    "minimum_observations": 24,
    "checkpoint_tolerance_days": 0,
}

OUTPUT_CONFIG = {
    "forecast_table": os.getenv("FORECAST_OUTPUT_TABLE", "dwh_prod.forecast_sellin_eom"),
    "write_mode": "upsert",
}
