import pandas as pd
import pytest

from data.validation import (
    DataValidationError,
    validate_calendar,
    validate_region_alignment,
    validate_targets,
)


def test_calendar_accepts_consecutive_binary_working_days():
    dates = pd.date_range("2026-01-01", periods=21, freq="D")
    calendar = pd.DataFrame({"date": dates, "is_working_day": [int(d.weekday() < 5) for d in dates]})
    assert len(validate_calendar(calendar, required_checkpoints=[4, 7])) == 21


def test_calendar_rejects_duplicate_dates():
    calendar = pd.DataFrame({"date": ["2026-01-01", "2026-01-01"], "is_working_day": [1, 0]})
    with pytest.raises(DataValidationError, match="duplicate dates"):
        validate_calendar(calendar)


def test_calendar_rejects_missing_date():
    calendar = pd.DataFrame({"date": ["2026-01-01", "2026-01-03"], "is_working_day": [1, 1]})
    with pytest.raises(DataValidationError, match="missing/non-consecutive"):
        validate_calendar(calendar)


def test_calendar_rejects_invalid_working_day_flag():
    calendar = pd.DataFrame({"date": ["2026-01-01"], "is_working_day": [2]})
    with pytest.raises(DataValidationError, match="only 0/1"):
        validate_calendar(calendar)


def test_region_alignment_rejects_unmapped_region():
    daily = pd.DataFrame({"regioncode": ["A", "B"]})
    targets = pd.DataFrame({"regioncode": ["A"]})
    with pytest.raises(DataValidationError, match="mapping mismatch"):
        validate_region_alignment(daily, targets)


def test_targets_reject_duplicate_region_month():
    targets = pd.DataFrame({
        "regioncode": ["A", "A"],
        "periode": ["2026-01-01", "2026-01-01"],
        "target_sellin": [100, 110],
    })
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_targets(targets)
