# MachineGuard+ Project Information and Report Help

Last updated from current repository state: 2026-05-02

This file is a detailed but precise project report reference for MachineGuard+. It combines the current notebooks, cleaned datasets, saved model artifacts, generated plots, API code, Streamlit dashboard, pipeline code, tests, and JSON metrics into one report-friendly document.

## 1. Executive Summary

MachineGuard+ is an end-to-end predictive maintenance system for industrial machines. It uses machine sensor readings to identify failure risk, predict the likely failure type, segment operating conditions into health regimes, estimate tool wear, discover interpretable fault patterns, and recommend maintenance actions.

The system is not only a notebook experiment. The repository also contains a deployable FastAPI service, a Streamlit dashboard, a Prefect-based pipeline, Docker files, model artifacts, automated tests, and quality/metric outputs.

Main project idea:

```text
Raw machine sensor data
  -> EDA and feature engineering
  -> PCA dimensionality reduction
  -> K-Means health-regime clustering
  -> Classification for failure prediction
  -> Regression for tool-wear estimation
  -> Time-series trend analysis
  -> Association-rule mining
  -> Recommendation system
  -> FastAPI + Streamlit + MLOps pipeline
```

The project is built around the AI4I 2020 Predictive Maintenance dataset. The dataset is highly imbalanced: 9,661 records are non-failures and only 339 records are failures. Because of that imbalance, F1-score and class-wise behavior are more meaningful than accuracy alone.

## 2. Business Problem

Factories want to reduce unexpected machine downtime. A maintenance engineer needs answers to questions such as:

- Is this machine likely to fail?
- What type of failure is most likely?
- Is this machine operating normally, degrading, or in a critical state?
- How worn is the tool?
- What maintenance action should be taken now?
- Which sensor patterns explain the risk?

MachineGuard+ answers these questions through multiple connected ML tasks:

- Classification predicts machine failure and failure type.
- Regression estimates tool wear or life consumed.
- PCA compresses correlated sensor readings.
- Clustering creates health regimes.
- Time-series analysis checks whether temporal features improve prediction.
- Association rules expose frequent fault patterns.
- Recommendation logic maps failure type and machine age to maintenance actions.

## 3. Repository Structure

Important project files and folders:

```text
data/raw_data/predictive_maintenance_data.csv
data/cleaned_data/eda_data.csv
data/cleaned_data/clustered_data.csv

notebooks/eda.ipynb
notebooks/clustering.ipynb
notebooks/classification.ipynb
notebooks/Tool_Wear_Regression.ipynb
notebooks/Time_series.ipynb
notebooks/Association_rules.ipynb
notebooks/MachineGuard_Recommender.ipynb

plots/
models/
outputs/

src/api/main.py
streamlit_app.py
main_pipeline.py
scripts/deepchecks_runner.py
tests/

Dockerfile
docker-compose.yml
requirements.txt
requirements.runtime.txt
RUN_AND_DEPLOY.md
PROJECT_LEARNING_GUIDE.md
MACHINEGUARD_PROJECT_INFO.md
```

## 4. Dataset Overview

Dataset: AI4I 2020 Predictive Maintenance

Raw dataset shape:

```text
Rows:    10,000
Columns: 14
```

Raw columns:

| Column | Meaning |
|---|---|
| UDI | Sequential record ID; used as a time/order proxy |
| Product ID | Product identifier |
| Type | Machine/product quality tier: L, M, H |
| Air temperature [K] | Ambient air temperature |
| Process temperature [K] | Process temperature during machining |
| Rotational speed [rpm] | Machine spindle speed |
| Torque [Nm] | Motor/mechanical torque |
| Tool wear [min] | Tool usage/wear in minutes |
| Machine failure | Binary target, 1 means failure |
| TWF | Tool Wear Failure flag |
| HDF | Heat Dissipation Failure flag |
| PWF | Power Failure flag |
| OSF | Overstrain Failure flag |
| RNF | Random Failure flag |

Machine type distribution:

| Type | Count | Share |
|---|---:|---:|
| L | 6,000 | 60.00% |
| M | 2,997 | 29.97% |
| H | 1,003 | 10.03% |

Failure distribution:

| Class | Count | Share |
|---|---:|---:|
| No failure | 9,661 | 96.61% |
| Failure | 339 | 3.39% |

Failure type counts:

| Failure type | Count |
|---|---:|
| HDF | 115 |
| OSF | 98 |
| PWF | 95 |
| TWF | 46 |
| RNF | 19 |

Important note: failure type counts add to more than 339 because a small number of rows contain more than one failure flag.

## 5. Raw Sensor Statistics

Main raw sensor statistics:

| Feature | Mean | Std | Min | Q1 | Median | Q3 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Air temperature | 300.005 | 2.000 | 295.300 | 298.300 | 300.100 | 301.500 | 304.500 |
| Process temperature | 310.006 | 1.484 | 305.700 | 308.800 | 310.100 | 311.100 | 313.800 |
| Rotational speed | 1538.776 | 179.284 | 1168.000 | 1423.000 | 1503.000 | 1612.000 | 2886.000 |
| Torque | 39.987 | 9.969 | 3.800 | 33.200 | 40.100 | 46.800 | 76.600 |
| Tool wear | 107.951 | 63.654 | 0.000 | 53.000 | 108.000 | 162.000 | 253.000 |

Distribution notes from EDA:

- Air temperature and process temperature are roughly smooth and centered around their means.
- Rotational speed is strongly right-skewed, with skewness 1.993 and kurtosis 7.393.
- Torque is nearly symmetric, with skewness -0.010.
- Tool wear is broad and almost flat across its operating range, with skewness 0.027.

Outlier summary using the IQR rule:

| Feature | Lower fence | Upper fence | Outliers | Outlier percent |
|---|---:|---:|---:|---:|
| Air temperature | 293.50 | 306.30 | 0 | 0.00% |
| Process temperature | 305.35 | 314.55 | 0 | 0.00% |
| Rotational speed | 1139.50 | 1895.50 | 418 | 4.18% |
| Torque | 12.80 | 67.20 | 69 | 0.69% |
| Tool wear | -110.50 | 325.50 | 0 | 0.00% |

## 6. EDA Findings

Sensor correlation matrix:

| Pair | Correlation | Interpretation |
|---|---:|---|
| Air temperature vs process temperature | 0.876 | Very strong positive thermal relationship |
| Rotational speed vs torque | -0.875 | Very strong inverse mechanical relationship |
| Tool wear vs other raw sensors | Around 0.000 to 0.014 | Tool wear is mostly independent from instant sensor values |

These correlations justify PCA because the five raw sensors contain redundant structure. PCA can compress the thermal relationship and the mechanical relationship into fewer independent components.

Top current correlations with `machine_failure` from `eda_data.csv`:

| Feature | Abs correlation |
|---|---:|
| hdf | 0.576 |
| osf | 0.531 |
| pwf | 0.523 |
| twf | 0.363 |
| torque_speed_ratio | 0.206 |
| torque | 0.191 |
| power_W | 0.176 |
| high_wear_flag | 0.174 |
| torque_roll_std_5 | 0.132 |
| torque_delta | 0.132 |
| wear_rate | 0.130 |
| temp_diff | 0.112 |
| tool_wear_roll_mean_5 | 0.106 |
| tool_wear | 0.105 |

Important leakage note:

The columns `twf`, `hdf`, `pwf`, `osf`, and `rnf` are failure label flags. They are useful for analysis and target construction, but they should not be used as ordinary model features for real inference because they directly encode the answer.

## 7. Feature Engineering

The EDA notebook cleans column names and creates `data/cleaned_data/eda_data.csv`.

Cleaned/engineered dataset shape:

```text
Rows:    10,000
Columns: 40
Nulls:   0
```

Column-name cleaning:

- Removed units and brackets.
- Converted names to lowercase.
- Replaced spaces with underscores.
- Example: `Air temperature [K]` becomes `air_temperature`.

Interaction features:

| Feature | Formula or rule | Purpose |
|---|---|---|
| type_enc | L=0, M=1, H=2 | Numeric machine type |
| temp_diff | process_temperature - air_temperature | Thermal stress gap |
| power_W | torque * rotational_speed * 2*pi/60 | Mechanical power proxy |
| wear_rate | tool_wear / rotational_speed | Wear per rotational speed unit |
| torque_speed_ratio | torque / rotational_speed | Overload-at-low-speed signal |
| high_wear_flag | tool_wear in high wear zone | Binary wear risk flag |
| thermal_overload | temp_diff > 10 | Binary thermal-stress flag |

Engineered feature statistics:

| Feature | Mean | Std | Min | Median | Max |
|---|---:|---:|---:|---:|---:|
| temp_diff | 10.001 | 1.001 | 7.600 | 9.800 | 12.100 |
| power_W | 6279.745 | 1067.418 | 1148.441 | 6271.027 | 10469.923 |
| wear_rate | 0.071 | 0.043 | 0.000 | 0.070 | 0.185 |
| torque_speed_ratio | 0.027 | 0.009 | 0.001 | 0.027 | 0.064 |
| high_wear_flag | 0.104 | 0.305 | 0.000 | 0.000 | 1.000 |
| thermal_overload | 0.446 | 0.497 | 0.000 | 0.000 | 1.000 |

Time-series features:

The EDA notebook treats `udi` as a chronological index and creates 20 rolling/lag/delta features for the five sensor columns:

- Rolling mean with window 5.
- Rolling standard deviation with window 5.
- First difference/delta.
- Lag-1 previous value.

Examples:

```text
air_temperature_roll_mean_5
air_temperature_roll_std_5
air_temperature_delta
air_temperature_lag1
process_temperature_roll_mean_5
rotational_speed_delta
torque_roll_std_5
tool_wear_lag1
```

## 8. PCA

Notebook: `notebooks/clustering.ipynb`

PCA is applied to the five raw sensor columns:

```text
air_temperature
process_temperature
rotational_speed
torque
tool_wear
```

Before PCA, the sensor columns are standardized with `StandardScaler`.

Explained variance from the current clustering notebook:

| Component | Explained variance | Cumulative variance |
|---|---:|---:|
| PC1 | 38.21% | 38.21% |
| PC2 | 36.82% | 75.03% |
| PC3 | 19.99% | 95.02% |
| PC4 | 2.53% | 97.55% |
| PC5 | 2.45% | 100.00% |

The notebook keeps 3 components because they explain about 95.0% of total variance.

PCA interpretation from notebook output:

| Component | Most influential feature | Loading |
|---|---|---:|
| PC1 | Air temperature | 0.506 |
| PC2 | Torque | 0.507 |
| PC3 | Tool wear | 1.000 |

Saved PCA columns in `clustered_data.csv`:

| Component | Mean | Std | Min | Median | Max |
|---|---:|---:|---:|---:|---:|
| PC1 | 0.000 | 1.382 | -4.960 | -0.052 | 6.658 |
| PC2 | -0.000 | 1.357 | -7.121 | 0.047 | 4.272 |
| PC3 | -0.000 | 1.000 | -1.768 | -0.002 | 2.235 |

## 9. K-Means Clustering and Health Regimes

Notebook: `notebooks/clustering.ipynb`

The clustering notebook uses PCA output as input to K-Means. It evaluates different `k` values using elbow and silhouette plots, then chooses `k=3` to match the domain idea of three health regimes:

- Normal
- Degraded
- Critical

Current clustering summary:

```text
Input features:       5 sensor columns
PCA components kept:  3
Variance retained:    95.0%
K-Means clusters:     3
Notebook silhouette:  0.2802
Pipeline silhouette:  0.3815
```

The notebook silhouette uses the 3-component clustering setup. The automated pipeline metric in `outputs/metrics.json` computes silhouette on a narrower PC space, so its value differs.

Health regime distribution:

| Health regime | Count | Failure count | Failure rate |
|---|---:|---:|---:|
| Normal | 3,999 | 87 | 2.18% |
| Degraded | 2,094 | 46 | 2.20% |
| Critical | 3,907 | 206 | 5.27% |

Cluster sensor means:

| Regime | Air temp | Process temp | Rotational speed | Torque | Tool wear |
|---|---:|---:|---:|---:|---:|
| Normal | 298.264 | 308.740 | 1473.311 | 43.323 | 104.873 |
| Degraded | 300.120 | 310.100 | 1793.786 | 26.680 | 109.206 |
| Critical | 301.725 | 311.250 | 1469.108 | 43.704 | 110.429 |

Failure types per regime:

| Regime | TWF | HDF | PWF | OSF | RNF |
|---|---:|---:|---:|---:|---:|
| Normal | 13 | 0 | 31 | 48 | 5 |
| Degraded | 12 | 0 | 31 | 0 | 3 |
| Critical | 21 | 115 | 33 | 50 | 11 |

Key interpretation:

- Critical contains all HDF cases in the clustered data and has the highest failure rate.
- Degraded has high rotational speed and lower torque, suggesting a distinct operating regime rather than simply a more dangerous regime.
- Normal and Degraded have similar failure rates, while Critical is clearly riskier.

## 10. Classification

Notebook: `notebooks/classification.ipynb`

The classification notebook implements a two-stage failure prediction pipeline:

Stage 1:

- Binary classification: no failure vs failure.

Stage 2:

- Failure type classification among TWF, HDF, PWF, OSF, RNF for rows predicted/known as failures.

Final pipeline:

- Combines Stage 1 and Stage 2 into a 6-class output:
  - No Failure
  - TWF
  - HDF
  - PWF
  - OSF
  - RNF

Features used by XGBoost:

```text
air_temperature
process_temperature
rotational_speed
torque
tool_wear
type_enc
temp_diff
power_w
wear_rate
torque_speed_ratio
high_wear_flag
thermal_overload
health_regime_enc
pc1
pc2
```

Features used by SVM:

```text
pc1
pc2
```

The SVM setup is intentionally lower-dimensional so the decision boundary can be visualized in principal-component space.

### 10.1 Stage 1 Binary Classification Results

XGBoost Stage 1:

| Metric | Value |
|---|---:|
| F1-macro | 0.8641 |
| ROC-AUC | 0.9802 |
| Accuracy | 0.98 |
| Failure recall | 0.81 |
| Failure precision | 0.68 |

SVM Stage 1:

| Metric | Value |
|---|---:|
| F1-macro | 0.6028 |
| ROC-AUC | 0.8642 |
| Accuracy | 0.86 |
| Failure recall | 0.84 |
| Failure precision | 0.17 |

Interpretation:

- XGBoost is much better balanced.
- SVM catches many failures but produces many more false positives, reflected by low failure precision.

### 10.2 Stage 2 Failure-Type Results

XGBoost Stage 2:

| Metric | Value |
|---|---:|
| F1-macro | 0.9660 |
| Accuracy | 0.97 |
| Weighted F1 | 0.97 |

Per-class highlights:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| TWF | 1.00 | 0.89 | 0.94 | 9 |
| HDF | 1.00 | 0.96 | 0.98 | 28 |
| PWF | 1.00 | 1.00 | 1.00 | 13 |
| OSF | 0.89 | 1.00 | 0.94 | 16 |
| RNF | 0.00 | 0.00 | 0.00 | 0 |

SVM Stage 2:

| Metric | Value |
|---|---:|
| F1-macro | 0.5371 |
| Accuracy | 0.67 |
| Weighted F1 | 0.68 |

Interpretation:

- XGBoost is substantially stronger for failure-type recognition.
- RNF has zero support in the current Stage 2 test split, so its score is not informative there.

### 10.3 End-to-End 6-Class Classification

XGBoost full pipeline:

| Metric | Value |
|---|---:|
| Accuracy | 0.9800 |
| F1-macro | 0.7169 |

XGBoost per-class final results:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| No Failure | 0.99 | 0.99 | 0.99 | 1932 |
| TWF | 0.09 | 0.09 | 0.09 | 11 |
| HDF | 0.89 | 0.89 | 0.89 | 28 |
| PWF | 0.81 | 1.00 | 0.90 | 13 |
| OSF | 0.58 | 0.94 | 0.71 | 16 |
| RNF | 0.00 | 0.00 | 0.00 | 0 |

SVM full pipeline:

| Metric | Value |
|---|---:|
| Accuracy | 0.8465 |
| F1-macro | 0.2504 |

SVM per-class final results:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| No Failure | 0.99 | 0.86 | 0.92 | 1932 |
| TWF | 0.00 | 0.00 | 0.00 | 11 |
| HDF | 0.13 | 0.79 | 0.23 | 28 |
| PWF | 0.16 | 0.69 | 0.26 | 13 |
| OSF | 0.06 | 0.31 | 0.10 | 16 |
| RNF | 0.00 | 0.00 | 0.00 | 0 |

Final classification conclusion:

XGBoost is the best classifier in the current notebooks. It wins on the final 6-class F1-macro score, binary ROC-AUC, and failure-type classification stability.

### 10.4 Cross-Validation Results

Cross-validation F1-macro:

| Model | F1-macro mean | Std |
|---|---:|---:|
| XGBoost Stage 1 binary | 0.9890 | 0.0018 |
| XGBoost Stage 2 type | 0.9282 | 0.0787 |
| SVM Stage 1 binary | 0.7807 | 0.0082 |
| SVM Stage 2 type | 0.3817 | 0.0595 |

## 11. Regression: Tool Wear / Life Consumed

Notebook: `notebooks/Tool_Wear_Regression.ipynb`

The regression notebook predicts `Life_Consumed`, a normalized target derived from tool wear:

```text
Life_Consumed = tool_wear / max_observed_tool_wear
```

Current target statistics:

| Statistic | Value |
|---|---:|
| Count | 9,991 |
| Mean | 0.4270 |
| Std | 0.2514 |
| Min | 0.0000 |
| Q1 | 0.2095 |
| Median | 0.4269 |
| Q3 | 0.6403 |
| Max | 1.0000 |

Regression feature set:

```text
air_temperature
process_temperature
rotational_speed
torque
Torque_roll10
Process_Temp_roll10
temp_diff_new
temp_diff_cum
type_enc
health_regime_enc
PC1
PC2
PC3
```

Train/test split:

| Split | Samples | Mean target | Std target |
|---|---:|---:|---:|
| Train | 7,992 | 0.4288 | 0.2519 |
| Test | 1,999 | 0.4199 | 0.2495 |

Hyperparameter search:

```text
5 folds
20 candidates
100 total fits
Best CV RMSE: 0.002765
```

Best XGBoost regressor parameters:

| Parameter | Value |
|---|---|
| model__subsample | 0.8 |
| model__n_estimators | 300 |
| model__max_depth | 6 |
| model__learning_rate | 0.05 |
| model__colsample_bytree | 1.0 |

Regression evaluation:

| Split | R2 | RMSE |
|---|---:|---:|
| Train | 0.999985 | 0.000989 |
| Test | 0.999888 | 0.002642 |

Feature importances:

| Feature | Importance |
|---|---:|
| PC3 | 0.997791 |
| Process_Temp_roll10 | 0.000472 |
| health_regime_enc | 0.000463 |
| air_temperature | 0.000441 |
| process_temperature | 0.000363 |
| temp_diff_cum | 0.000275 |
| PC1 | 0.000098 |
| temp_diff_new | 0.000027 |
| Torque_roll10 | 0.000021 |
| rotational_speed | 0.000014 |
| torque | 0.000014 |
| PC2 | 0.000013 |
| type_enc | 0.000008 |

Interpretation:

PC3 dominates because PCA found PC3 to be almost entirely tool-wear driven. The regression result is therefore extremely strong, but it should be described carefully: it is excellent for estimating normalized life consumed from features that include a tool-wear-heavy principal component.

## 12. Time-Series Analysis

Notebook: `notebooks/Time_series.ipynb`

The time-series notebook compares classification performance with raw sensor features versus time-series features.

Current result:

| Approach | F1-score macro |
|---|---:|
| Raw sensors only | 0.792084 |
| Time-series features | 0.796436 |

Interpretation:

Time-series features provide a small but positive improvement. This supports the feature-engineering decision to include rolling statistics, lag values, and delta/change features.

The automated pipeline also stores a time-series proxy metric:

```text
forecast_rmse_proxy: 18.322177
```

This proxy compares rolling tool-wear behavior against lagged tool wear. It is not the same as the notebook classification F1 result; it is a lightweight pipeline monitoring metric.

## 13. Association Rules

Notebook: `notebooks/Association_rules.ipynb`

The association-rule task mines frequent rule patterns separately by health regime.

Current rule summary:

| Regime | Rule count | Avg confidence | Max confidence | Avg lift | Max lift |
|---|---:|---:|---:|---:|---:|
| Normal | 15 | 0.789 | 1.000 | 74.185 | 84.635 |
| Degraded | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| Critical | 16 | 0.793 | 1.000 | 50.964 | 79.735 |
| All regimes | 31 | 0.791 | 1.000 | 62.200 | 84.635 |

Important notebook insight:

Normal and Critical regimes produce sharp, deterministic patterns. Degraded does not produce valid deterministic rules under the current mining settings, which suggests it is a more mixed or high-variance operating state.

Example pattern types found in `models/rules.json`:

- Low rotational speed plus extreme tool wear plus PWF often implies OSF-related patterns.
- Extreme torque and extreme tool wear appear in high-lift rules.
- Rules are stored under regime keys: `Normal`, `Degraded`, and `Critical`.

Saved artifact:

```text
models/rules.json
```

## 14. Recommendation System

Notebook: `notebooks/MachineGuard_Recommender.ipynb`

The recommendation system uses a structured knowledge base of maintenance actions. It maps failure type and age bin to repair instructions, parts, urgency, repair manual IDs, and downtime.

Saved artifact:

```text
models/knowledge_base.json
```

Knowledge base size:

```text
Total entries: 15
Failure types covered: TWF, HDF, PWF, OSF, RNF
Age bins covered: Young, Mid, Old
Coverage pairs: 15
```

Coverage table:

| Failure type | Young | Mid | Old |
|---|---:|---:|---:|
| HDF | 1 | 1 | 1 |
| OSF | 1 | 1 | 1 |
| PWF | 1 | 1 | 1 |
| RNF | 1 | 1 | 1 |
| TWF | 1 | 1 | 1 |

Estimated downtime by failure type:

| Failure type | Min hours | Mean hours | Max hours |
|---|---:|---:|---:|
| HDF | 1.0 | 3.33 | 6.0 |
| OSF | 0.5 | 2.67 | 5.0 |
| PWF | 2.0 | 4.83 | 8.0 |
| RNF | 1.0 | 2.17 | 3.5 |
| TWF | 1.5 | 2.50 | 4.0 |

Recommendation behavior:

- If failure type is `No Failure`, the system returns a healthy-machine message.
- If failure type is invalid, the notebook recommender raises a validation error.
- If RUL is critically low, urgency can be elevated.
- For valid failure types, top ranked maintenance actions are returned with part codes and manual IDs.

## 15. Current Saved Model Artifacts

Files currently present in `models/`:

| Artifact | Purpose |
|---|---|
| scaler.pkl | Standardizes raw sensor values before PCA |
| pca.pkl | PCA transformer |
| kmeans.pkl | K-Means clustering model |
| xgb_classifier.pkl | Main XGBoost classifier |
| svm_pipeline.pkl | SVM comparison pipeline |
| tool_wear_regressor.pkl | Tool-wear/life-consumed regressor |
| xgb_regressor.json | Native XGBoost regressor format |
| xgb_regressor.pkl | Pickled regressor variant |
| label_encoder.pkl | Label encoding artifact |
| rules.json | Association-rule artifact |
| knowledge_base.json | Maintenance recommendation knowledge base |

## 16. FastAPI Application

Main file:

```text
src/api/main.py
```

Service name:

```text
MachineGuard+
```

API version:

```text
1.0.0
```

Core endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Root service status |
| `/health` | GET | Model version, loaded artifacts, buffer status |
| `/predict` | POST | Full prediction pipeline |
| `/predict/classification` | POST | Classification-specific prediction |
| `/predict/regression` | POST | Regression/tool-wear-specific prediction |
| `/predict/timeseries` | POST | CSV upload using latest row and trend |
| `/feedback` | POST | Collect corrected labels from engineers |
| `/metrics` | GET | Prediction log metrics and drift heuristic |

Input schema for a single sensor reading:

```text
machine_id
machine_type
air_temperature
process_temperature
rotational_speed
torque
tool_wear
machine_age_bin
```

API inference flow:

```text
SensorInput
  -> engineer_features()
  -> apply_pca()
  -> assign_cluster()
  -> predict_failure()
  -> estimate_tool_wear()
  -> get_urgency_level()
  -> match_rules()
  -> get_recommendations()
  -> PredictionResponse
```

API feature engineering:

- `temp_diff`
- `power_W`
- `wear_rate`
- `torque_speed_ratio`
- `high_wear_flag`
- `thermal_overload`
- `type_enc`

Failure threshold:

```text
FAILURE_THRESHOLD = 0.3
```

Urgency mapping:

| Predicted wear | Urgency |
|---:|---|
| >= 200 | CRITICAL |
| >= 150 | HIGH |
| >= 100 | MEDIUM |
| < 100 | LOW |

The API has compatibility fallbacks for older XGBoost wrappers by using the underlying booster when needed.

## 17. Streamlit Dashboard

Main file:

```text
streamlit_app.py
```

Dashboard purpose:

- Provide interactive sensor input.
- Run the same core prediction logic as the FastAPI app.
- Show model/artifact status.
- Display prediction history.
- Provide a visual gallery for plots.
- Provide project-flow context.

Dashboard tabs:

| Tab | Purpose |
|---|---|
| Live Prediction | Manual sensor input and prediction dashboard |
| Batch Trend | Batch/trend-style prediction workflow |
| Visual Gallery | Displays plots from `plots/` |
| Project Flow | Explains the pipeline flow |

Scenario presets:

| Preset | Meaning |
|---|---|
| Normal operation | Low-risk example |
| Watch closely | Medium-risk example |
| Likely failure | High-risk example |

The dashboard uses Altair for sensor visualizations and Streamlit metrics/cards for prediction display.

## 18. Prefect Pipeline and MLOps Layer

Main file:

```text
main_pipeline.py
```

Flow name:

```text
machineguard-multi-ml-pipeline
```

Pipeline tasks:

- Optionally execute key notebooks using Papermill.
- Validate expected data/model artifacts.
- Compute multi-task metrics.
- Save metrics to `outputs/metrics.json`.

Expected notebook execution plan:

| Notebook | Expected output |
|---|---|
| eda.ipynb | data/cleaned_data/eda_data.csv |
| clustering.ipynb | data/cleaned_data/clustered_data.csv |
| Association_rules.ipynb | models/rules.json |
| MachineGuard_Recommender.ipynb | models/knowledge_base.json |

Artifact validation checks:

- `eda_data.csv`
- `clustered_data.csv`
- `pca.pkl`
- `kmeans.pkl`
- `scaler.pkl`
- `xgb_classifier.pkl`
- `rules.json`
- `knowledge_base.json`

Current `outputs/metrics.json` timestamp:

```text
2026-04-29T20:22:41.122354+00:00
```

Automated metric summary:

| Task | Metric | Value |
|---|---|---:|
| Classification baseline | Accuracy | 0.9660 |
| Classification baseline | F1 | 0.4914 |
| Classification improved | Accuracy | 0.9605 |
| Classification improved | F1 | 0.1937 |
| Regression baseline | RMSE | 65.3427 |
| Regression improved | RMSE | 65.3427 |
| Clustering | Silhouette score | 0.3815 |
| Time series | Forecast RMSE proxy | 18.3222 |
| Association | Rule count | 31 |
| Association | Avg confidence | 0.7909 |
| Association | Avg lift | 62.2003 |
| Recommendation | Entries | 15 |
| Recommendation | Coverage pairs | 15 |

Important report note:

The notebook classification results are the stronger experimental results. The automated pipeline metrics are lighter operational checks and may use different target construction/model alignment, so they should not be presented as the main model-performance claim without explanation.

## 19. Deepchecks / Quality Report

Current file:

```text
outputs/deepchecks_report.json
```

Current result:

| Check group | Passed |
|---|---|
| Integrity | false |
| Performance | true |
| Model info | true |

Interpretation:

The quality-report output indicates performance and model-info checks passed, but at least one integrity check failed. In a final defense/report, this should be presented as a known monitoring result that needs inspection, not ignored.

## 20. Tests

Test files:

```text
tests/test_api_main.py
tests/test_main_pipeline.py
tests/conftest.py
```

API tests verify:

- `/health` returns core fields.
- `/predict` returns the expected response shape.
- `/metrics` updates after prediction.
- `/predict/classification` returns classification fields.
- `/predict/regression` returns regression fields.
- `/predict/timeseries` accepts a CSV upload and returns row count/trend output.

Pipeline tests verify:

- Expected artifacts exist.
- Multi-task metrics can be generated.
- Metrics can be saved to a JSON file.

## 21. Docker and Deployment

Deployment-related files:

```text
Dockerfile
docker-compose.yml
RUN_AND_DEPLOY.md
requirements.runtime.txt
requirements.txt
```

The project can be run as an API service and pipeline service through Docker Compose. The FastAPI app is the model-serving layer, while the pipeline handles artifact validation and metric generation.

## 22. Plot Inventory and Report Interpretation

The `plots/` directory contains 35 PNG files. These are the current visual outputs to reference in the final project report.

### EDA and Class Distribution Plots

`class_distribution.png`

- Size: 1200 x 600.
- Shows binary machine-failure imbalance and failure-type distribution.
- Main statistic: 96.61% no failure vs 3.39% failure.
- Failure type counts: HDF 115, OSF 98, PWF 95, TWF 46, RNF 19.
- Report use: demonstrate why accuracy alone is misleading.

`01_class_distribution.png`

- Size: 1035 x 583.
- Classification-focused class distribution plot.
- Reinforces imbalance before model evaluation.

### PCA and Clustering Plots

`pre_pca_correlation.png`

- Size: 667 x 489.
- Shows strong raw sensor correlations before PCA.
- Key stats: air/process temperature correlation 0.876, rotational speed/torque correlation -0.875.
- Report use: justify dimensionality reduction.

`pca_scree.png`

- Size: 1290 x 495.
- Shows individual and cumulative explained variance.
- Key stats: PC1 38.21%, PC2 36.82%, PC3 19.99%, cumulative first 3 PCs 95.02%.
- Report use: justify keeping 3 PCA components.

`pca_loadings.png`

- Size: 764 x 390.
- Shows feature loading contribution to principal components.
- Key interpretation: PC1 thermal/temperature axis, PC2 mechanical torque-speed axis, PC3 tool-wear axis.

`pca_2d_scatter.png`

- Size: 1389 x 495.
- Shows records projected into PC1-PC2 space, colored by failure or type.
- Report use: visualize how failures distribute in reduced sensor space.

`elbow.png`

- Size: 790 x 390.
- Shows K-Means inertia over candidate k values.
- Report use: support cluster selection.

`silhouette_scores.png`

- Size: 790 x 390.
- Shows silhouette score over candidate k values.
- Key notebook statistic: best k is 3 with silhouette about 0.2803.

`silhouette_plot.png`

- Size: 889 x 590.
- Shows per-cluster silhouette samples for final k=3.
- Report use: explain separation quality and cluster cohesion.

`cluster_pca_scatter.png`

- Size: 1489 x 592.
- Shows final Normal, Degraded, and Critical clusters in PCA space.
- Key stats: Normal 3,999 rows, Degraded 2,094 rows, Critical 3,907 rows.

`cluster_sensor_profiles.png`

- Size: 1990 x 495.
- Shows mean sensor profiles for each health regime.
- Key stats: Critical has highest air/process temperature; Degraded has highest rotational speed and lowest torque.

`failure_type_per_regime.png`

- Size: 989 x 490.
- Shows failure-type distribution inside each health regime.
- Key stats: Critical contains all 115 HDF rows and the largest total failures.

### Classification Plots

`02_confusion_matrices.png`

- Size: 1645 x 770.
- Compares confusion matrices for model stages or model variants.
- Report use: show where classifiers confuse failure classes.

`confusion_matrix_comparison.png`

- Size: 2400 x 900.
- Large comparison of final confusion matrices.
- Key conclusion: XGBoost is much cleaner than SVM in the final 6-class pipeline.

`xgb_confusion_matrix.png`

- Size: 1200 x 900.
- XGBoost confusion matrix.
- Key final stats: accuracy 0.9800 and F1-macro 0.7169 for full 6-class pipeline.

`svm_confusion_matrix.png`

- Size: 1200 x 900.
- SVM confusion matrix.
- Key final stats: accuracy 0.8465 and F1-macro 0.2504 for full 6-class pipeline.

`03_model_comparison.png`

- Size: 1185 x 734.
- Compares model F1/score values.
- Main conclusion: XGBoost outperforms SVM across binary, type, and full-pipeline settings.

`perclass_f1_comparison.png`

- Size: 1500 x 750.
- Compares F1 by class.
- Important detail: minority classes, especially TWF and RNF, remain difficult because of very low support.

`04_feature_importance.png`

- Size: 1034 x 583.
- Classification feature importance plot.
- Report use: discuss which engineered/raw features drive failure prediction.

`xgb_feature_importance.png`

- Size: 1200 x 900.
- XGBoost-specific feature importance.
- Report use: supports interpretability of the selected classifier.

`feature_importance.png`

- Size: 1334 x 1010.
- General feature importance plot, likely from regression or classification depending on notebook context.
- If used in report, label it with the exact notebook section to avoid ambiguity.

`05_cv_scores.png`

- Size: 1185 x 583.
- Cross-validation score plot.
- Key stats: XGBoost Stage 1 CV F1 0.9890 +/- 0.0018; XGBoost Stage 2 CV F1 0.9282 +/- 0.0787.

`cv_comparison.png`

- Size: 1050 x 750.
- Cross-validation comparison.
- Report use: show model stability.

`cv_boxplot.png`

- Size: 1050 x 750.
- Cross-validation distribution/boxplot.
- Report use: show variance across folds.

`06_pc_decision_space.png`

- Size: 1935 x 772.
- Decision space in principal component coordinates.
- Report use: visually explain SVM-style PC1/PC2 decision boundaries.

`svm_decision_boundary.png`

- Size: 1350 x 1050.
- SVM decision boundary plot.
- Important interpretation: useful for visualization, but final SVM performance is weaker than XGBoost.

`failure_type_final.png`

- Size: 2383 x 982.
- Final failure-type visualization.
- Report use: summarize multi-class failure-type behavior.

### Regression Plots

`actual_vs_predicted.png`

- Size: 1933 x 740.
- Shows predicted vs actual regression target.
- Key stats from notebook: test R2 0.999888 and test RMSE 0.002642.

`residual_plots.png`

- Size: 1936 x 740.
- Shows regression residual behavior.
- Report use: confirm residuals are very small in the current regression setup.

`learning_curve.png`

- Size: 1335 x 733.
- Shows training behavior as sample size changes.
- Report use: discuss whether more data is likely to improve model fit.

`test_comparison.png`

- Size: 1650 x 750.
- Regression test comparison plot.
- Report use: show prediction tracking on the held-out chronological test segment.

### Time-Series Pattern Plots

`Tool_wear_trend.png`

- Size: 991 x 451.
- Shows tool-wear trend over ordered records.
- Report use: justify treating `UDI` as a time/order proxy.

`pre_failure_Spike_pattern.png`

- Size: 1017 x 451.
- Shows pre-failure spike pattern.
- Report use: explain why deltas and rolling windows can help.

`Torque_Shock_pattern.png`

- Size: 1008 x 374.
- Shows torque shock behavior.
- Key related finding: time-series features improve F1-macro from 0.792084 to 0.796436.

### Association Plot

`association.png`

- Size: 1589 x 590.
- Shows rule strength/behavior by health regime.
- Key stats: 31 total rules, average confidence 0.7909, average lift 62.2003.
- Regime detail: Normal 15 rules, Degraded 0 rules, Critical 16 rules.

## 23. Main Results to Put in Final Report

Recommended concise results table:

| Task | Best/current result |
|---|---|
| EDA | 10,000 rows, no missing values, 3.39% failure rate |
| PCA | 3 components retain 95.02% variance |
| Clustering | k=3 health regimes; Critical failure rate 5.27% |
| Classification | XGBoost full pipeline accuracy 0.9800, F1-macro 0.7169 |
| Binary failure detection | XGBoost Stage 1 F1-macro 0.8641, ROC-AUC 0.9802 |
| Failure-type classification | XGBoost Stage 2 F1-macro 0.9660 |
| Regression | XGBoost test R2 0.999888, RMSE 0.002642 |
| Time series | F1 improves from 0.792084 to 0.796436 |
| Association rules | 31 rules, avg confidence 0.7909, avg lift 62.2003 |
| Recommendation | 15 KB entries covering 5 failure types x 3 age bins |
| API | FastAPI endpoints for full, classification, regression, time-series, feedback, metrics |
| MLOps | Prefect pipeline, Docker, tests, metrics JSON, Deepchecks output |

## 24. Suggested Report Narrative

Use this order in the final written report:

1. Introduce predictive maintenance and why it matters.
2. Describe the AI4I dataset and severe class imbalance.
3. Present EDA findings: no missing values, correlated sensors, imbalance, outliers.
4. Explain feature engineering: physics-inspired, rolling, lag, and delta features.
5. Explain PCA: why it is justified and why 3 PCs were kept.
6. Explain clustering: k=3 regimes and failure-rate-based regime labels.
7. Explain classification: two-stage design, XGBoost vs SVM, final 6-class result.
8. Explain regression: life consumed/tool wear estimation and dominant PC3 effect.
9. Explain time-series analysis: small but positive F1 improvement.
10. Explain association rules: interpretable high-confidence patterns by regime.
11. Explain recommendation system: knowledge-base coverage and practical actions.
12. Explain deployment: FastAPI, Streamlit, Prefect, Docker, tests, metrics.
13. Discuss limitations and future improvements.

## 25. Limitations

Important limitations to mention honestly:

- The dataset is synthetic, so real factory deployment would require validation on live sensor data.
- The failure class is highly imbalanced, especially RNF and TWF.
- RNF has very low support and often receives zero meaningful evaluation support in splits.
- Some regression features are strongly related to tool wear through PC3, so the very high regression score should be interpreted carefully.
- The API currently computes single-point features at inference time; rolling features in production would need historical machine readings.
- The Deepchecks integrity output is currently false and should be investigated.
- The automated pipeline metric for classification differs from the notebook result, likely due to different evaluation/model-alignment logic.

## 26. Future Work

Recommended improvements:

- Add real streaming history per machine so API inference can compute rolling and lag features exactly like training.
- Improve minority-class handling for TWF and RNF using resampling, class weights, or anomaly/few-shot strategies.
- Calibrate failure probabilities with Platt scaling or isotonic calibration.
- Add clearer model cards for each saved artifact.
- Investigate and fix Deepchecks integrity failure.
- Align automated pipeline metrics with notebook evaluation logic.
- Add drift monitoring beyond the current simple average-probability heuristic.
- Add persistent feedback storage instead of in-memory feedback buffer.
- Add CI execution for tests, linting, and pipeline smoke checks.

## 27. One-Paragraph Project Summary

MachineGuard+ is an end-to-end predictive maintenance system for industrial machines using the AI4I 2020 dataset. It cleans and engineers sensor data, reduces correlated sensors through PCA, clusters operating states into Normal, Degraded, and Critical regimes, predicts failure risk and failure type using XGBoost and SVM models, estimates tool wear with regression, analyzes time-series improvements, mines association rules for interpretable fault patterns, and recommends maintenance actions through a structured knowledge base. The strongest current classifier is the XGBoost full pipeline with 0.9800 accuracy and 0.7169 F1-macro, while PCA keeps 3 components explaining 95.02% variance and clustering identifies a Critical regime with a 5.27% failure rate. The project also includes FastAPI endpoints, a Streamlit dashboard, Prefect orchestration, Docker support, automated tests, saved model artifacts, plots, and monitoring outputs.

