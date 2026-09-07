import pandas as pd

from features.historical import build_historical_features


def test_momentum_feature_does_not_use_current_target():
    df = pd.DataFrame(
        {
            "regioncode": ["R1"] * 4,
            "periode": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"]),
            "monthly_value": [100.0, 120.0, 150.0, 180.0],
        }
    )
    features_a = build_historical_features(df, ["regioncode"])

    changed = df.copy()
    changed.loc[changed["periode"] == pd.Timestamp("2026-04-01"), "monthly_value"] = 9999.0
    features_b = build_historical_features(changed, ["regioncode"])

    row_a = features_a.loc[features_a["periode"] == pd.Timestamp("2026-04-01")].iloc[0]
    row_b = features_b.loc[features_b["periode"] == pd.Timestamp("2026-04-01")].iloc[0]
    assert row_a["mom_growth"] == row_b["mom_growth"]
    assert row_a["lag_1"] == row_b["lag_1"]
    assert row_a["rolling_mean_3"] == row_b["rolling_mean_3"]


def test_momentum_is_previous_month_growth():
    df = pd.DataFrame(
        {
            "regioncode": ["R1"] * 3,
            "periode": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"]),
            "monthly_value": [100.0, 120.0, 150.0],
        }
    )
    features = build_historical_features(df, ["regioncode"])
    march = features.loc[features["periode"] == pd.Timestamp("2026-03-01")].iloc[0]
    assert march["mom_growth"] == 0.2
