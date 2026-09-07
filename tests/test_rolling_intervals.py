import pandas as pd

from backtest.rolling_intervals import _ensemble_with_prior_calibration


def test_interval_calibration_excludes_current_and_future_targets():
    prior = pd.DataFrame(
        {
            "periode": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "actual": [100.0, 120.0],
            "forecast_baseline": [90.0, 100.0],
            "forecast_ets": [95.0, 105.0],
            "forecast_sarima": [92.0, 108.0],
            "forecast_xgboost": [94.0, 106.0],
        }
    )
    target = pd.Series(
        {
            "forecast_baseline": 110.0,
            "forecast_ets": 111.0,
            "forecast_sarima": 109.0,
            "forecast_xgboost": 112.0,
        }
    )

    p10_a, p50_a, p90_a = _ensemble_with_prior_calibration(target, prior, 7)

    contaminated = pd.concat(
        [
            prior,
            pd.DataFrame(
                {
                    "periode": [pd.Timestamp("2026-03-01")],
                    "actual": [1000.0],
                    "forecast_baseline": [110.0],
                    "forecast_ets": [111.0],
                    "forecast_sarima": [109.0],
                    "forecast_xgboost": [112.0],
                }
            ),
        ],
        ignore_index=True,
    )
    p10_b, p50_b, p90_b = _ensemble_with_prior_calibration(target, contaminated.iloc[:2], 7)

    assert p50_a == p50_b
    assert p10_a == p10_b
    assert p90_a == p90_b


def test_first_target_without_prior_calibration_has_no_valid_interval():
    target = pd.Series(
        {
            "forecast_baseline": 100.0,
            "forecast_ets": 105.0,
            "forecast_sarima": 103.0,
            "forecast_xgboost": 101.0,
        }
    )
    p10, p50, p90 = _ensemble_with_prior_calibration(target, pd.DataFrame(), 4)
    assert pd.isna(p10)
    assert p50 > 0
    assert pd.isna(p90)
