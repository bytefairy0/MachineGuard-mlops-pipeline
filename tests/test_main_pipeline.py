import json
from pathlib import Path

from main_pipeline import evaluate_multi_task_metrics, save_metrics, validate_expected_outputs


def test_pipeline_generates_metrics_file():
    checks = validate_expected_outputs.fn()
    assert all(checks.values())

    metrics = evaluate_multi_task_metrics.fn()
    metrics_path = Path(save_metrics.fn(metrics))
    assert metrics_path.exists()
    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    assert "metrics" in payload
    assert "tasks" in payload["metrics"]
