import joblib
import xgboost as xgb

# Load the old pickle
#model = joblib.load('models/xgb_isFailure.pkl')
#model = joblib.load('models/xgb_failureTypes.pkl')


# Save it properly as a JSON
# If it's a scikit-learn wrapper:
#model.get_booster().save_model('models/xgb_isFailure.json') 
#model.get_booster().save_model('models/xgb_failureTypes.json') 
# If it's a raw booster:
# model.save_model('models/xgb_isFailure.json')

# 1. Load the pickle file
model = joblib.load('models/tool_wear_regressor.pkl')

# 2. Extract the underlying booster and save to JSON
if hasattr(model, "get_booster"):
    # This is for the Scikit-Learn wrapper (XGBRegressor)
    model.get_booster().save_model('models/xgb_regressor.json')
    print("Success: Saved via Booster")
else:
    # This is for a direct XGBoost object
    model.save_model('models/xgb_regressor.json')
    print("Success: Saved directly")