import insight.batch as batch
import insight.persistence as persistence


class FakeConn:
    def __init__(self):
        self.sql = []

    def execute(self, statement, params=None):
        self.sql.append((str(statement), params))


class FakeBegin:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeEngine:
    def __init__(self):
        self.conn = FakeConn()

    def begin(self):
        return FakeBegin(self.conn)


def test_deactivate_failed_clears_region_gm_and_ceo(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(persistence, "get_engine", lambda: engine)

    persistence.deactivate_failed([
        {"input": {"hierarchy_level": "REGION", "periode": "2026-09-01", "entity_code": "ASWJWA1"}},
        {"input": {"hierarchy_level": "GM", "periode": "2026-09-01", "entity_code": "GM-COMJAWA"}},
        {"input": {"hierarchy_level": "CEO", "periode": "2026-09-01", "entity_code": "CEO"}},
    ])

    assert len(engine.conn.sql) == 3
    assert "ai_region_insight" in engine.conn.sql[0][0]
    assert engine.conn.sql[0][1] == {"p": "2026-09-01", "c": "ASWJWA1"}
    assert "ai_gm_insight" in engine.conn.sql[1][0]
    assert engine.conn.sql[1][1] == {"p": "2026-09-01", "c": "GM-COMJAWA"}
    assert "ai_insight" in engine.conn.sql[2][0]
    assert "CEO_SALES_SUMMARY" in engine.conn.sql[2][0]


def test_batch_invalidates_failed_rows_before_raising(monkeypatch):
    class FakeAgent:
        failures = [{
            "input": {
                "hierarchy_level": "REGION",
                "periode": "2026-09-01",
                "entity_code": "ASWJWA1",
            },
            "error": "forced",
        }]

        def __init__(self, period=None):
            pass

        def generate(self, continue_on_error=False):
            assert continue_on_error is True
            return []

    invalidated = []
    monkeypatch.setattr(batch, "InsightAgent", FakeAgent)
    monkeypatch.setattr(batch, "deactivate_failed", lambda failures: invalidated.extend(failures))

    try:
        batch.run("2026-09-01")
    except RuntimeError as exc:
        assert "1 failed row" in str(exc)
    else:
        raise AssertionError("batch.run() must surface failed rows")

    assert len(invalidated) == 1
    assert invalidated[0]["input"]["entity_code"] == "ASWJWA1"
