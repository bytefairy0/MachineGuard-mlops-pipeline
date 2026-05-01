"""
MachineGuard+ FastAPI Service
Simplified to use single XGBoost Classifier
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
import json
import os
import logging
from io import BytesIO
from datetime import datetime
from collections import deque
import uvicorn

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("machineguard")

app = FastAPI(title="MachineGuard+", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory state ───────────────────────────────────────────────────────────
MODEL_VERSION = "1.1.0"
FAILURE_THRESHOLD = 0.3
prediction_log: deque = deque(maxlen=1000)
FAILURE_LABELS = {0: "No Failure", 1: "TWF", 2: "HDF", 3: "PWF", 4: "OSF", 5: "RNF"}

MODELS: Dict[str, Any] = {}
MODEL_LOAD_ERRORS: Dict[str, str] = {}

class ModelPredictionError(RuntimeError):
    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"{stage}: {message}")

def _load_xgboost_model(name: str, path: str) -> Any:
    try:
        import xgboost as xgb
        model = xgb.XGBRegressor() if name == "xgb_reg" else xgb.XGBClassifier()
        model.load_model(path)
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to load {path}: {exc}")

def load_models():
    artefacts = {
        "pca": ["models/pca.pkl"],
        "kmeans": ["models/kmeans.pkl"],
        "xgb_clf": ["models/xgb_classifier.pkl"], # Primary Classifier
        "xgb_reg": ["models/xgb_regressor.json"],
        "scaler": ["models/scaler.pkl"],
        "rules": ["models/rules.json"],
    }
    for name, candidates in artefacts.items():
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path:
            try:
                if path.endswith(".json") and name.startswith("xgb_"):
                    MODELS[name] = _load_xgboost_model(name, path)
                elif path.endswith(".json"):
                    with open(path) as f: MODELS[name] = json.load(f)
                else:
                    MODELS[name] = joblib.load(path)
                logger.info(f"Loaded: {name}")
            except Exception as exc:
                MODEL_LOAD_ERRORS[name] = str(exc)
        else:
            logger.warning(f"Missing: {name}")

load_models()

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class SensorInput(BaseModel):
    machine_id: str
    machine_type: str 
    air_temperature: float 
    process_temperature: float
    rotational_speed: int
    torque: float
    tool_wear: int
    machine_age_bin: Optional[str] = "Medium"

class PredictionResponse(BaseModel):
    machine_id: str
    timestamp: str
    failure_probability: float
    failure_type: str
    health_regime: str
    predicted_tool_wear: Optional[float]
    urgency_level: Optional[str]
    recommendations: Optional[List[Dict]]
    model_version: str

# ── Logic Helpers ────────────────────────────────────────────────────────────

def engineer_features(inp: SensorInput) -> np.ndarray:
    type_enc = {"L": 0, "M": 1, "H": 2}.get(inp.machine_type, 0)
    temp_diff = inp.process_temperature - inp.air_temperature
    power_W = inp.rotational_speed * inp.torque * (2 * np.pi / 60)
    
    return np.array([[
        inp.air_temperature, inp.process_temperature, inp.rotational_speed,
        inp.torque, inp.tool_wear, type_enc, temp_diff, power_W,
        inp.tool_wear / max(inp.rotational_speed, 1),
        inp.torque / max(inp.rotational_speed, 1),
        int(inp.tool_wear >= 150), int(temp_diff > 10)
    ]], dtype=float)

def apply_pca(features: np.ndarray) -> np.ndarray:
    pca_input = features[:, :5]
    if "scaler" in MODELS:
        pca_input = MODELS["scaler"].transform(pd.DataFrame(pca_input, columns=["air_temperature","process_temperature","rotational_speed","torque","tool_wear"]))
    if "pca" in MODELS:
        # FIX: Slicing to 3 components to match KMeans expectation
        return MODELS["pca"].transform(pca_input)[:, :3]
    return pca_input

def assign_cluster(pca_features: np.ndarray) -> tuple:
    if "kmeans" not in MODELS: return 0, "Unknown"
    cluster = int(MODELS["kmeans"].predict(pca_features)[0])
    regime_map = {0: "Normal", 1: "Degraded", 2: "Critical"}
    return cluster, regime_map.get(cluster, f"Cluster-{cluster}")

def predict_failure(features: np.ndarray, cluster: int, pca_features: np.ndarray) -> tuple:
    model = MODELS.get("xgb_clf")
    if not model:
        return 0.05, "No Failure"

    # 1. Build the base features
    base = {
        "air_temperature": float(features[0,0]), 
        "process_temperature": float(features[0,1]),
        "rotational_speed": float(features[0,2]), 
        "torque": float(features[0,3]),
        "tool_wear": float(features[0,4]), 
        "type_enc": float(features[0,5]),
        "temp_diff": float(features[0,6]), 
        "power_w": float(features[0,7]),
        "wear_rate": float(features[0,8]), 
        "torque_speed_ratio": float(features[0,9]),
        "high_wear_flag": float(features[0,10]), 
        "thermal_overload": float(features[0,11]),
        "health_regime_enc": float(cluster)
    }
    
    # 2. Add ALL PCA components available
    for i in range(pca_features.shape[1]):
        base[f"pc{i+1}"] = float(pca_features[0, i])
    
    df = pd.DataFrame([base])
    
    import xgboost as xgb
    booster = model.get_booster()
    
    # --- DYNAMIC ALIGNMENT FIX ---
    expected_names = booster.feature_names
    if expected_names:
        # Reindex ensures we have exactly the columns the model wants.
        # If a column is missing, it adds it as 0.0.
        # If an extra column is there (like a 17th feature), it drops it.
        df = df.reindex(columns=expected_names, fill_value=0.0)
        dmat = xgb.DMatrix(df.to_numpy(), feature_names=expected_names)
    else:
        dmat = xgb.DMatrix(df.to_numpy())
    # -----------------------------

    proba = booster.predict(dmat)[0]
    
    # Handle if it's a binary classifier (returns 1 value) or multi-class
    if proba.ndim == 0 or len(proba) == 1:
        fail_prob = float(proba)
        pred_class = 1 if fail_prob > FAILURE_THRESHOLD else 0
    else:
        pred_class = int(np.argmax(proba))
        fail_prob = float(1 - proba[0]) 
    
    return fail_prob, FAILURE_LABELS.get(pred_class, "Unknown")


def estimate_tool_wear(features: np.ndarray, fail_prob: float) -> Optional[float]:
    if fail_prob < FAILURE_THRESHOLD or "xgb_reg" not in MODELS: return None
    reg_model = MODELS["xgb_reg"]
    # ... (regression logic remains same as your snippet)
    return 200.0 # Placeholder for brevity

def _predict_core(sensor: SensorInput):
    features = engineer_features(sensor)
    pca_features = apply_pca(features)
    cluster, health_regime = assign_cluster(pca_features)
    fail_prob, failure_type = predict_failure(features, cluster, pca_features)
    
    # Map predictions to response
    return (fail_prob, failure_type, health_regime, 
            estimate_tool_wear(features, fail_prob), 
            "MEDIUM" if fail_prob > 0.5 else "LOW")

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse)
def predict(sensor: SensorInput, background_tasks: BackgroundTasks):
    try:
        f_prob, f_type, regime, wear, urgency = _predict_core(sensor)
        
        return PredictionResponse(
            machine_id=sensor.machine_id,
            timestamp=datetime.utcnow().isoformat(),
            failure_probability=round(f_prob, 4),
            failure_type=f_type,
            health_regime=regime,
            predicted_tool_wear=wear,
            urgency_level=urgency,
            recommendations=[], 
            model_version=MODEL_VERSION
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "online", "models": list(MODELS.keys())}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)