from fastapi.testclient import TestClient

from src.api import main


def _payload() -> dict:
    return {
        "machine_id": "M001",
        "machine_type": "L",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551,
        "torque": 42.8,
        "tool_wear": 108,
        "machine_age_bin": "Mid",
    }


def test_health_endpoint_returns_core_fields():
    client = TestClient(main.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert "model_version" in body
    assert "loaded_artefacts" in body


def test_predict_endpoint_response_shape():
    client = TestClient(main.app)
    resp = client.post("/predict", json=_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["machine_id"] == "M001"
    assert "failure_probability" in body
    assert "failure_type" in body
    assert "health_regime" in body
    assert "predicted_tool_wear" in body
    assert "urgency_level" in body


def test_metrics_endpoint_updates_after_prediction():
    client = TestClient(main.app)
    pred = client.post("/predict", json=_payload())
    assert pred.status_code == 200

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    body = metrics.json()
    assert body["total_predictions"] >= 1
    assert "avg_failure_probability" in body
