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

    dates = pd.date_range("2024-01-02", "2026-03-31", freq="B")
    calendar = pd.DataFrame({"date": dates, "is_working_day": 1})

    # One daily observation in every historical month makes each prior
    # target-month checkpoint snapshot valid for training.
    historical_dates = [
        dates[(dates >= month) & (dates < month + pd.offsets.MonthBegin(1))][0]
        for month in months[:-1]
    ]
    daily = pd.DataFrame(
        {
            "regioncode": ["R1"] * len(historical_dates),
            "invoice_date": historical_dates,
            "sellin_value": [5.0] * len(historical_dates),
        }
    )

    target_month = pd.Timestamp("2026-03-01")
    target_dates = dates[
        (dates >= target_month) & (dates < target_month + pd.offsets.MonthBegin(1))
    ][:10]
    daily = pd.concat(
        [
            daily,
            pd.DataFrame(
                {
                    "regioncode": ["R1"] * len(target_dates),
                    "invoice_date": target_dates,
                    "sellin_value": [10.0] * len(target_dates),
                }
            ),
        ],
        ignore_index=True,
    )

    targets = pd.DataFrame(
        {"regioncode": ["R1"], "periode": [target_month], "target_sellin": [300.0]}
    )
    return monthly, daily, calendar, targets


def test_checkpoint_row_excludes_post_checkpoint_sales():
    monthly, daily, calendar, targets = _fixtures()
    target_month = pd.Timestamp("2026-03-01")
    row = _checkpoint_row(
        monthly, daily, calendar, targets, ["regioncode"],
        target_month, 4, ("R1",), 24,
    )
    checkpoint_date = calendar.loc[
        (calendar["date"] >= target_month)
        & calendar["is_working_day"].astype(bool), "date"
    ].iloc[3]
    expected = daily.loc[
        (daily["invoice_date"] >= target_month)
        & (daily["invoice_date"] <= checkpoint_date), "sellin_value"
    ].sum()
    assert row["mtd_value"] == expected
    target_total = daily.loc[
        (daily["invoice_date"] >= target_month)
        & (daily["invoice_date"] < target_month + pd.offsets.MonthBegin(1)),
        "sellin_value",
    ].sum()
    assert row["mtd_value"] < target_total
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
