import pandas as pd
import pytest

import insight.batch as batch
import insight.persistence as persistence


def test_verify_ai_input_freshness_accepts_matching_mtd(monkeypatch):
    monkeypatch.setattr(
        batch,
        "read_sql",
        lambda query, params: pd.DataFrame(
            [
                {
                    "regioncode": "ASWJWA1",
                    "forecast_mtd": 13972754201.0,
                    "ai_mtd": 13972754201.0,
                }
            ]
        ),
    )

    batch.verify_ai_input_freshness("2026-09-01")


def test_verify_ai_input_freshness_blocks_stale_mtd(monkeypatch):
    monkeypatch.setattr(
        batch,
        "read_sql",
        lambda query, params: pd.DataFrame(
            [
                {
                    "regioncode": "ASWJWA1",
                    "forecast_mtd": 16665419857.0,
                    "ai_mtd": 13972754201.0,
                }
            ]
        ),
    )

    with pytest.raises(RuntimeError, match="canonical MTD mismatch"):
        batch.verify_ai_input_freshness("2026-09-01")


class _Result:
    def mappings(self):
        return self

    def all(self):
        return [
            {
                "regioncode": "ASWJWA1",
                "periode": "2026-09-01",
                "total_sellin": 13972754201.0,
            }
        ]


class _Conn:
    def __init__(self, mismatch=False):
        self.mismatch = mismatch
        self.calls = 0

    def execute(self, statement, params):
        self.calls += 1
        if self.calls == 1:
            value = 12000000000.0 if self.mismatch else 13972754201.0
            return _ResultWithValue(value)
        return _Result()


class _ResultWithValue:
    def __init__(self, value):
        self.value = value

    def mappings(self):
        return self

    def all(self):
        return [{
            "regioncode": "ASWJWA1",
            "periode": "2026-09-01",
            "total_sellin": self.value,
        }]


class _Context:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        return False


class _Engine:
    def __init__(self, conn):
        self.conn = conn

    def connect(self):
        return _Context(self.conn)


def test_verify_region_persistence_blocks_stale_active_snapshot(monkeypatch):
    monkeypatch.setattr(
        persistence,
        "get_engine",
        lambda: _Engine(_Conn(mismatch=True)),
    )

    results = [{
        "input": {
            "hierarchy_level": "REGION",
            "entity_code": "ASWJWA1",
            "periode": "2026-09-01",
            "mtd_actual": 13972754201.0,
        },
        "insight": {},
    }]

    with pytest.raises(RuntimeError, match="total_sellin mismatch"):
        persistence.verify_region_persistence(results)
