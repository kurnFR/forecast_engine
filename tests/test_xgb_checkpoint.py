import pandas as pd

from backtest.xgb_checkpoint import _build_training_frame, _checkpoint_row


def _fixtures():
    months = pd.date_range("2024-01-01", periods=26, freq="MS")
    monthly = pd.DataFrame(
        {
            "regioncode": ["R1"] * len(months),
            "periode": months,
            "monthly_value": [100.0 + i for i in range(len(months))],
        }
    )
    dates = pd.date_range("2026-03-02", "2026-03-31", freq="B")
    calendar = pd.DataFrame(
        {"date": dates, "is_working_day": 1}
    )
    daily = pd.DataFrame(
        {
            "regioncode": ["R1"] * 10,
            "invoice_date": dates[:10],
            "sellin_value": [10.0] * 10,
        }
    )
    targets = pd.DataFrame(
        {"regioncode": ["R1"], "periode": [pd.Timestamp("2026-03-01")], "target_sellin": [300.0]}
    )
    return monthly, daily, calendar, targets


def test_checkpoint_row_excludes_post_checkpoint_sales():
    monthly, daily, calendar, targets = _fixtures()
    target_month = pd.Timestamp("2026-03-01")
    row = _checkpoint_row(
        monthly, daily, calendar, targets, ["regioncode"],
        target_month, 4, ("R1",), 24,
    )
    checkpoint_date = calendar.loc[calendar["is_working_day"].astype(bool), "date"].iloc[3]
    expected = daily.loc[daily["invoice_date"] <= checkpoint_date, "sellin_value"].sum()
    assert row["mtd_value"] == expected
    assert row["mtd_value"] < daily["sellin_value"].sum()
    assert row["target_sellin"] == 300.0


def test_checkpoint_training_frame_excludes_scored_target_month():
    monthly, daily, calendar, targets = _fixtures()
    frame = _build_training_frame(
        monthly, daily, calendar, targets, ["regioncode"],
        pd.Timestamp("2026-03-01"), 4, 24,
    )
    assert not frame.empty
    assert frame["periode"].max() < pd.Timestamp("2026-03-01")
    assert (frame["checkpoint"] == 4).all()
