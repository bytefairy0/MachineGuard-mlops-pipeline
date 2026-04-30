from fastapi import FastAPI, Body
import joblib
import xgboost as xgb
import numpy as np
import pandas as pd

app = FastAPI()

# 1. Load OLD Pipeline (The one that worked yesterday)
# This contains Scaler + Model inside one file
old_pipeline = joblib.load('models/tool_wear_regressor.pkl')

# 2. Load NEW JSON Model (Just the weights)
json_model = xgb.XGBRegressor()
json_model.load_model('models/xgb_regressor.json')

# 3. Load the Scaler (since it's now a separate file)
scaler = joblib.load('models/scaler.pkl')

@app.post("/compare")
async def compare_predictions(data: dict = Body(...)):
    # Assuming data['features'] is a list of your raw sensor values
    # e.g., [300.1, 310.2, 1500, 40.5, 50, ...]
    raw_features = np.array([data['features']]) 
    
    # --- PREDICTION A: OLD PIPELINE ---
    # The pipeline handles scaling internally
    pred_old = old_pipeline.predict(raw_features)[0]
    
    # --- PREDICTION B: NEW JSON ---
    # We MUST scale manually first because JSON doesn't have the scaler
    scaled_features = scaler.transform(raw_features)
    pred_json = json_model.predict(scaled_features)[0]
    
    return {
        "old_pkl_result": float(pred_old),
        "new_json_result": float(pred_json),
        "match": bool(np.isclose(pred_old, pred_json, atol=1e-5)),
        "difference": float(abs(pred_old - pred_json))
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)