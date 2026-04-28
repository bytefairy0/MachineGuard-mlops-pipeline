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


def test_predict_classification_endpoint():
    client = TestClient(main.app)
    resp = client.post("/predict/classification", json=_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert "failure_probability" in body["prediction"]
    assert "failure_type" in body["prediction"]


def test_predict_regression_endpoint():
    client = TestClient(main.app)
    resp = client.post("/predict/regression", json=_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert "predicted_tool_wear" in body["prediction"]


def test_predict_timeseries_csv_upload():
    client = TestClient(main.app)
    csv_payload = (
        "machine_id,machine_type,air_temperature,process_temperature,rotational_speed,torque,tool_wear,machine_age_bin\n"
        "M001,L,298.1,308.6,1551,42.8,108,Mid\n"
        "M001,L,298.4,309.0,1549,43.1,111,Mid\n"
    )
    resp = client.post(
        "/predict/timeseries",
        files={"file": ("series.csv", csv_payload, "text/csv")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert body["prediction"]["rows_processed"] == 2
