import pandas as pd
import pytest

from data.validation import DataValidationError, validate_forecast_output


def _valid_output():
    return pd.DataFrame(
        [
            {
                "regioncode": "ASWJWA1",
                "periode": "2026-09-01",
                "mtd_value": 7_481_451_991,
                "elapsed_working_days": 9,
                "remaining_working_days": 17,
                "total_working_days": 26,
                "forecast_p10": 13_528_163_824.58,
                "forecast_p50": 21_666_482_120.10,
                "forecast_p90": 25_916_951_052.19,
                "target_sellin": 24_700_000_000,
                "achievement_pct_forecast": 87.7185519032,
            }
        ]
    )


def test_valid_forecast_output_contract():
    result = validate_forecast_output(_valid_output())
    assert len(result) == 1
    assert result.loc[0, "regioncode"] == "ASWJWA1"


def test_rejects_duplicate_region_month():
    df = pd.concat([_valid_output(), _valid_output()], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_forecast_output(df)


def test_rejects_invalid_interval_order():
    df = _valid_output()
    df.loc[0, "forecast_p90"] = df.loc[0, "forecast_p50"] - 1
    with pytest.raises(DataValidationError, match="P50 must be <= P90"):
        validate_forecast_output(df)


def test_rejects_working_day_mismatch():
    df = _valid_output()
    df.loc[0, "remaining_working_days"] = 16
    with pytest.raises(DataValidationError, match="do not reconcile"):
        validate_forecast_output(df)


def test_rejects_achievement_formula_drift():
    df = _valid_output()
    df.loc[0, "achievement_pct_forecast"] = 90.0
    with pytest.raises(DataValidationError, match="achievement_pct_forecast"):
        validate_forecast_output(df)


def test_allows_zero_target_without_division_error():
    df = _valid_output()
    df.loc[0, "target_sellin"] = 0
    df.loc[0, "achievement_pct_forecast"] = 0
    result = validate_forecast_output(df)
    assert result.loc[0, "target_sellin"] == 0
