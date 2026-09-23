import pytest

import insight.batch as batch
import insight.engine as engine


def _rows():
    return [
        {"hierarchy_level": "REGION", "entity_code": "BAD", "periode": "2026-09-01"},
        {"hierarchy_level": "REGION", "entity_code": "GOOD"},
        {"hierarchy_level": "GM", "entity_code": "GM-COMJAWA", "largest_shortfall_regioncode": "GOOD"},
        {"hierarchy_level": "CEO", "entity_code": "CEO"},
    ]


def test_generate_continues_after_row_failure(monkeypatch):
    monkeypatch.setattr(engine, "load_input", lambda period=None: _rows())
    monkeypatch.setattr(engine.time, "sleep", lambda _: None)

    def fake_run_hermes(prompt, attempts=2):
        if 'LEVEL: REGION\nAUTHORITATIVE INPUT:\n{"hierarchy_level":"REGION","entity_code":"BAD"}' in prompt:
            raise RuntimeError("forced row failure")
        return {"ok": True}

    monkeypatch.setattr(engine, "run_hermes", fake_run_hermes)
    monkeypatch.setattr(
        engine,
        "validate",
        lambda row, raw: {
            "entity_code": row["entity_code"],
            "ai_insight_category": "AT_RISK",
            "ai_diagnosis": "diagnosis",
            "triggered_action_plan": "action",
            "priority": "MEDIUM",
        },
    )

    agent = engine.InsightAgent()
    results = agent.generate(continue_on_error=True)

    assert [x["input"]["entity_code"] for x in results] == ["GOOD", "GM-COMJAWA", "CEO"]
    assert len(agent.failures) == 1
    assert agent.failures[0]["input"]["entity_code"] == "BAD"
    assert "forced row failure" in agent.failures[0]["error"]


def test_batch_persists_successes_then_surfaces_failures(monkeypatch):
    class FakeAgent:
        failures = [{"input": {"hierarchy_level": "REGION", "entity_code": "BAD"}, "error": "forced"}]

        def __init__(self, period=None):
            pass

        def generate(self, continue_on_error=False):
            assert continue_on_error is True
            return [{"input": {"hierarchy_level": "REGION", "entity_code": "GOOD"}, "insight": {}}]

    persisted = []
    monkeypatch.setattr(batch, "InsightAgent", FakeAgent)
    monkeypatch.setattr(batch, "persist", lambda results: persisted.extend(results))

    with pytest.raises(RuntimeError, match="1 failed row"):
        batch.run("2026-09-01")

    assert len(persisted) == 1
    assert persisted[0]["input"]["entity_code"] == "GOOD"
