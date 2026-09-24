import sys

import pandas as pd

import main


def test_main_runs_ai_after_successful_forecast_write(monkeypatch):
    calls = []
    period = pd.Timestamp("2026-09-01")

    monkeypatch.setattr(main, "run_training_pipeline", lambda: "trained")
    monkeypatch.setattr(main, "run_prediction_pipeline", lambda trained: pd.DataFrame({"periode": [period]}))
    monkeypatch.setattr(main, "validate_forecast_output", lambda result: result)
    monkeypatch.setattr(main, "write_forecast", lambda result: calls.append(("forecast", len(result))))
    monkeypatch.setattr(main, "run_insight_batch", lambda value: calls.append(("insight", value)) or [{"input": {}}])
    monkeypatch.setattr(sys, "argv", ["main.py"])

    main.main()

    assert calls == [("forecast", 1), ("insight", "2026-09-01")]


def test_main_does_not_run_ai_in_dry_run(monkeypatch):
    calls = []

    monkeypatch.setattr(main, "run_training_pipeline", lambda: "trained")
    monkeypatch.setattr(
        main,
        "run_prediction_pipeline",
        lambda trained: pd.DataFrame({"periode": [pd.Timestamp("2026-09-01")]}),
    )
    monkeypatch.setattr(main, "validate_forecast_output", lambda result: result)
    monkeypatch.setattr(main, "run_insight_batch", lambda value: calls.append(value))
    monkeypatch.setattr(sys, "argv", ["main.py", "--dry-run"])

    main.main()

    assert calls == []
