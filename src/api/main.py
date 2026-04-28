"""
MachineGuard+ FastAPI Service
Endpoints: /predict, /health, /feedback, /metrics
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
import json
import os
import logging
from datetime import datetime
from collections import deque
import uvicorn

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("machineguard")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MachineGuard+",
    description="Adaptive Predictive Maintenance Intelligence System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory state ───────────────────────────────────────────────────────────
MODEL_VERSION = "1.0.0"
FAILURE_THRESHOLD = 0.3          # probability threshold to run regression
feedback_buffer: List[Dict] = []
prediction_log: deque = deque(maxlen=1000)   # last 1000 predictions for drift

# Failure type mapping
FAILURE_LABELS = {0: "No Failure", 1: "TWF", 2: "HDF", 3: "PWF", 4: "OSF", 5: "RNF"}

# ── Model loading ─────────────────────────────────────────────────────────────
MODELS: Dict[str, Any] = {}

def load_models():
    """Load all saved model artefacts. Gracefully skip if not found (dev mode)."""
    artefacts = {
        "pca":        "models/pca.pkl",
        "kmeans":     "models/kmeans.pkl",
        "xgb_clf":    "models/xgb_classifier.pkl",
        "svm_clf":    "models/svm_classifier.pkl",
        "xgb_reg":    "models/xgb_regressor.pkl",
        "scaler":     "models/scaler.pkl",
        "rules":      "models/association_rules.json",
    }
    for name, path in artefacts.items():
        if os.path.exists(path):
            if path.endswith(".json"):
                with open(path) as f:
                    MODELS[name] = json.load(f)
            else:
                MODELS[name] = joblib.load(path)
            logger.info(f"Loaded artefact: {name}")
        else:
            logger.warning(f"Artefact not found (dev mode): {path}")

load_models()

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class SensorInput(BaseModel):
    """Raw sensor reading from one machine at one time step."""
    machine_id: str = Field(..., example="M001")
    machine_type: str = Field(..., example="L")          # L / M / H
    air_temperature: float = Field(..., example=298.1)   # Kelvin
    process_temperature: float = Field(..., example=308.6)
    rotational_speed: int   = Field(..., example=1551)   # rpm
    torque: float           = Field(..., example=42.8)   # Nm
    tool_wear: int          = Field(..., example=108)    # minutes
    machine_age_bin: Optional[str] = Field("Medium", example="High")  # Low/Medium/High

class PredictionResponse(BaseModel):
    machine_id: str
    timestamp: str
    failure_probability: float
    failure_type: str
    health_regime: str
    predicted_tool_wear: Optional[float]  # None if failure_prob < threshold
    urgency_level: Optional[str]
    recommendations: Optional[List[Dict]]  # None if rec. system not loaded
    triggered_rules: Optional[List[str]]   # association rules that fired
    model_version: str

class FeedbackInput(BaseModel):
    machine_id: str
    prediction_timestamp: str
    corrected_failure_type: str
    corrected_rul_hours: Optional[float] = None
    engineer_notes: Optional[str] = None

class MetricsResponse(BaseModel):
    model_version: str
    total_predictions: int
    feedback_buffer_size: int
    failure_rate_last_1000: float
    health_regime_distribution: Dict[str, int]
    avg_failure_probability: float
    drift_alert: bool

# ── Feature engineering helpers ───────────────────────────────────────────────

def engineer_features(inp: SensorInput) -> np.ndarray:
    """
    Produce the feature vector expected by the pipeline.
    In production this would use a window of prior readings;
    here we compute the single-point derived features.
    """
    type_enc = {"L": 0, "M": 1, "H": 2}.get(inp.machine_type, 0)
    temp_diff       = inp.process_temperature - inp.air_temperature
    power_W         = inp.rotational_speed * inp.torque * (2 * np.pi / 60)
    wear_rate       = inp.tool_wear / max(inp.rotational_speed, 1)
    torque_speed    = inp.torque / max(inp.rotational_speed, 1)
    high_wear_flag  = int(inp.tool_wear >= 150)
    thermal_overload = int(temp_diff > 10)

    features = np.array([[
        inp.air_temperature,
        inp.process_temperature,
        inp.rotational_speed,
        inp.torque,
        inp.tool_wear,
        type_enc,
        temp_diff,
        power_W,
        wear_rate,
        torque_speed,
        high_wear_flag,
        thermal_overload,
    ]], dtype=float)
    return features


def apply_pca(features: np.ndarray) -> np.ndarray:
    """Apply scaler + PCA on the raw 5 sensor columns."""
    pca_input = features[:, :5]
    if "scaler" in MODELS:
        pca_input = MODELS["scaler"].transform(pca_input)
    if "pca" in MODELS:
        return MODELS["pca"].transform(pca_input)
    return pca_input


def assign_cluster(pca_features: np.ndarray) -> tuple:
    """Return (cluster_id, health_regime_label)."""
    if "kmeans" not in MODELS:
        return 0, "Unknown"
    cluster = int(MODELS["kmeans"].predict(pca_features)[0])
    regime_map = {0: "Normal", 1: "Degraded", 2: "Critical"}
    return cluster, regime_map.get(cluster, f"Cluster-{cluster}")


def predict_failure(features: np.ndarray, cluster: int, pca_features: np.ndarray) -> tuple:
    """Return (failure_probability, failure_type_label)."""
    if "xgb_clf" not in MODELS:
        # Demo fallback — deterministic fake prediction
        prob = min(0.05 + features[0, 4] / 3000, 0.99)
        ftype = "No Failure" if prob < FAILURE_THRESHOLD else "HDF"
        return float(prob), ftype

    feat_with_cluster = np.hstack([features, [[cluster]], pca_features])
    proba = MODELS["xgb_clf"].predict_proba(feat_with_cluster)[0]
    pred_class = int(np.argmax(proba))
    fail_prob = float(1 - proba[0]) if len(proba) > 1 else float(proba[0])
    return fail_prob, FAILURE_LABELS.get(pred_class, "Unknown")


def estimate_tool_wear(features: np.ndarray, fail_prob: float) -> Optional[float]:
    """Return predicted tool wear, or None if below threshold."""
    if fail_prob < FAILURE_THRESHOLD:
        return None
    if "xgb_reg" not in MODELS:
        # Demo fallback centered around current wear and stress features
        wear_delta = features[0, 6] * 2.0 + features[0, 9] * 2500.0
        pred = max(0.0, min(300.0, features[0, 4] + wear_delta))
        return round(float(pred), 1)
    pred_wear = float(MODELS["xgb_reg"].predict(features)[0])
    return round(max(0.0, pred_wear), 1)


def get_urgency_level(predicted_wear: Optional[float]) -> Optional[str]:
    """Map predicted wear to urgency buckets used by the recommendation layer."""
    if predicted_wear is None:
        return None
    if predicted_wear >= 200:
        return "CRITICAL"
    if predicted_wear >= 150:
        return "HIGH"
    if predicted_wear >= 100:
        return "MEDIUM"
    return "LOW"


def match_rules(cluster: int, failure_type: str) -> List[str]:
    """Return association rules that match this cluster + failure type."""
    if "rules" not in MODELS:
        return []
    rules = MODELS["rules"]
    matched = []
    for rule in rules:
        if rule.get("cluster") == cluster and failure_type in rule.get("consequents", []):
            matched.append(rule.get("description", str(rule)))
    return matched[:3]   # top 3


def get_recommendations(failure_type: str, age_bin: str) -> Optional[List[Dict]]:
    """Content-based lookup from knowledge base JSON."""
    kb_path = "models/knowledge_base.json"
    if not os.path.exists(kb_path):
        return None
    with open(kb_path) as f:
        kb = json.load(f)
    if failure_type == "No Failure":
        return [{"message": "Machine is healthy"}]

    results = []
    for entry in kb:
        if entry.get("failure_type") == failure_type:
            score = 1.0 if entry.get("age_bin") == age_bin else 0.6
            results.append({**entry, "relevance_score": score})
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results[:3] if results else None

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"service": "MachineGuard+", "version": MODEL_VERSION, "status": "running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(sensor: SensorInput, background_tasks: BackgroundTasks):
    """
    Main inference endpoint.
    Accepts one sensor reading, runs the full pipeline, returns prediction.
    """
    ts = datetime.utcnow().isoformat()
    try:
        # 1. Feature engineering
        features = engineer_features(sensor)

        # 2. PCA
        pca_features = apply_pca(features)

        # 3. Clustering → health regime
        cluster, health_regime = assign_cluster(pca_features)

        # 4. Classification → failure type + probability
        fail_prob, failure_type = predict_failure(features, cluster, pca_features)

        # 5. Regression (conditional)
        predicted_tool_wear = estimate_tool_wear(features, fail_prob)
        urgency_level = get_urgency_level(predicted_tool_wear)

        # 6. Association rules
        triggered_rules = match_rules(cluster, failure_type)

        # 7. Recommendations (if rec system available)
        recs = get_recommendations(failure_type, sensor.machine_age_bin or "Medium")
        if recs and predicted_tool_wear is not None:
            for rec in recs:
                if isinstance(rec, dict):
                    rec["predicted_tool_wear"] = predicted_tool_wear
                    rec["urgency_level"] = urgency_level

        response = PredictionResponse(
            machine_id=sensor.machine_id,
            timestamp=ts,
            failure_probability=round(fail_prob, 4),
            failure_type=failure_type,
            health_regime=health_regime,
            predicted_tool_wear=predicted_tool_wear,
            urgency_level=urgency_level,
            recommendations=recs,
            triggered_rules=triggered_rules if triggered_rules else None,
            model_version=MODEL_VERSION,
        )

        # Log for drift monitoring
        background_tasks.add_task(
            log_prediction,
            sensor=sensor,
            failure_type=failure_type,
            fail_prob=fail_prob,
            health_regime=health_regime,
        )

        logger.info(
            f"[PREDICT] {sensor.machine_id} → {failure_type} "
            f"(p={fail_prob:.3f}, wear={predicted_tool_wear})"
        )
        return response

    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))


def log_prediction(sensor, failure_type, fail_prob, health_regime):
    prediction_log.append({
        "timestamp": datetime.utcnow().isoformat(),
        "machine_id": sensor.machine_id,
        "failure_type": failure_type,
        "fail_prob": fail_prob,
        "health_regime": health_regime,
        "air_temp": sensor.air_temperature,
        "process_temp": sensor.process_temperature,
        "rotational_speed": sensor.rotational_speed,
        "torque": sensor.torque,
        "tool_wear": sensor.tool_wear,
    })


@app.get("/health")
def health_check():
    """Returns current model version and loaded artefacts."""
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "loaded_artefacts": list(MODELS.keys()),
        "uptime_predictions": len(prediction_log),
        "feedback_buffer_size": len(feedback_buffer),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/feedback")
def submit_feedback(fb: FeedbackInput, background_tasks: BackgroundTasks):
    """
    Engineers submit corrected labels. When buffer hits threshold,
    a retraining job is triggered (Prefect flow in production).
    """
    RETRAIN_THRESHOLD = 50

    feedback_buffer.append({
        **fb.dict(),
        "received_at": datetime.utcnow().isoformat(),
    })
    logger.info(f"[FEEDBACK] {fb.machine_id} correction received. Buffer: {len(feedback_buffer)}")

    if len(feedback_buffer) >= RETRAIN_THRESHOLD:
        background_tasks.add_task(trigger_retraining)
        return {
            "status": "accepted",
            "buffer_size": len(feedback_buffer),
            "retraining_triggered": True,
            "message": f"Buffer hit {RETRAIN_THRESHOLD} — retraining job queued.",
        }

    return {
        "status": "accepted",
        "buffer_size": len(feedback_buffer),
        "retraining_triggered": False,
        "message": f"{RETRAIN_THRESHOLD - len(feedback_buffer)} more corrections needed to trigger retraining.",
    }


def trigger_retraining():
    """Stub — in production this kicks off the Prefect retraining flow."""
    logger.info(f"[RETRAIN] Retraining triggered with {len(feedback_buffer)} feedback samples.")
    # prefect_flow.run(feedback_data=list(feedback_buffer))
    # feedback_buffer.clear()


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    """Returns drift statistics and prediction distribution."""
    logs = list(prediction_log)
    total = len(logs)

    failure_rate = 0.0
    avg_prob = 0.0
    regime_dist: Dict[str, int] = {"Normal": 0, "Degraded": 0, "Critical": 0, "Unknown": 0}
    drift_alert = False

    if total > 0:
        failures = sum(1 for l in logs if l["failure_type"] != "No Failure")
        failure_rate = round(failures / total, 4)
        avg_prob = round(sum(l["fail_prob"] for l in logs) / total, 4)

        for l in logs:
            regime = l.get("health_regime", "Unknown")
            regime_dist[regime] = regime_dist.get(regime, 0) + 1

        # Simple drift heuristic: flag if avg failure prob spikes above 0.4
        recent = logs[-100:] if total >= 100 else logs
        recent_avg = sum(l["fail_prob"] for l in recent) / len(recent)
        drift_alert = recent_avg > 0.4

    return MetricsResponse(
        model_version=MODEL_VERSION,
        total_predictions=total,
        feedback_buffer_size=len(feedback_buffer),
        failure_rate_last_1000=failure_rate,
        health_regime_distribution=regime_dist,
        avg_failure_probability=avg_prob,
        drift_alert=drift_alert,
    )


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)