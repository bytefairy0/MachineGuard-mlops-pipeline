# MachineGuard+ — Project Reference Guide
### AI221 Machine Learning Engineering | GIKI
> **Course Instructor:** Dr. Ali Imran Sandhu
> **Dataset:** AI4I 2020 Predictive Maintenance (Kaggle)
> **Domain:** Industrial IoT / Smart Manufacturing

---

## Table of Contents
1. [What is this project?](#1-what-is-this-project)
2. [The Big Picture — How the pipeline works](#2-the-big-picture--how-the-pipeline-works)
3. [Dataset Deep Dive](#3-dataset-deep-dive)
4. [What we found in EDA](#4-what-we-found-in-eda)
5. [All ML Tasks — What, Why, How](#5-all-ml-tasks--what-why-how)
   - [Feature Engineering](#51-feature-engineering-done-in-eda-notebook)
   - [PCA](#52-pca--dimensionality-reduction)
   - [Clustering](#53-k-means-clustering--health-segmentation)
   - [Classification](#54-classification--failure-type-prediction)
   - [Regression](#55-regression--remaining-useful-life)
   - [Time Series](#56-time-series-analysis)
   - [Association Rules](#57-association-rules)
   - [Recommendation System](#58-recommendation-system)
6. [Notebook Structure & File Flow](#6-notebook-structure--file-flow)
7. [Prompts for Each Task](#7-prompts-for-each-task)
8. [MLOps Layer](#8-mlops-layer)
9. [Tech Stack](#9-tech-stack)
10. [Key Things to Remember](#10-key-things-to-remember)

---

## 1. What is this project?

MachineGuard+ is an **end-to-end machine learning pipeline** for predictive maintenance in industrial manufacturing. The core idea is simple: instead of waiting for a machine to break down (reactive maintenance) or replacing parts on a fixed schedule (preventive maintenance), we use sensor data to **predict failures before they happen**.

A real factory manager wakes up every day asking four questions:
- Is any machine about to fail?
- If yes — what kind of failure is it?
- How many hours do I have left before it breaks?
- Which machines in my entire fleet need urgent attention right now?

MachineGuard+ answers all four using a sequential ML pipeline. Each stage's output feeds meaningfully into the next — this is **not** a collection of random models thrown together to tick course requirements. Every single component maps to a real industrial need.

The course (AI221) requires you to demonstrate: classification, regression, dimensionality reduction, clustering, time series analysis, association rules, and a recommendation system — all wired together in one coherent workflow with a FastAPI backend, Prefect orchestration, Docker containerisation, and a GitHub Actions CI/CD pipeline.

---

## 2. The Big Picture — How the pipeline works

```
Raw Sensor Data (CSV)
        ↓
[1] Feature Engineering
    → Rolling stats, lag features, physics-inspired interactions
        ↓
[2] PCA (Dimensionality Reduction)
    → Compress 5 correlated sensors into 2-3 principal components
        ↓
[3] K-Means Clustering
    → Assign each machine a health regime: Normal / Degraded / Critical
        ↓
[4] Association Rules (FP-Growth)
    → Mine fault patterns per health cluster
    → e.g. {High Torque ∧ High Tool Wear} → {OSF} with confidence 0.85
        ↓
[5] Classification (XGBoost + SVM)
    → Predict failure type (TWF / HDF / PWF / OSF / RNF / No Failure)
        ↓  [only if failure probability > threshold]
[6] Regression (XGBoost)
    → Estimate Remaining Useful Life in operating hours
        ↓
[7] Recommendation System
    → Return top 3 repair actions + required spare parts
        ↓
FastAPI Service → Streamlit Dashboard → MLOps Loop
```

The pipeline is **sequential and dependent**. You cannot skip PCA and go straight to clustering. You cannot do regression without classification telling you a failure is coming. This sequential dependency is a design feature, not a limitation.

---

## 3. Dataset Deep Dive

**Dataset:** AI4I 2020 Predictive Maintenance Dataset (synthetic, from Kaggle)
**URL:** kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification

### Raw Columns

| Column | Type | Description |
|--------|------|-------------|
| `UDI` | int | Sequential record ID — acts as our time index |
| `Product ID` | str | Machine identifier (drop before ML) |
| `Type` | str | Machine quality tier: L (Low), M (Medium), H (High) |
| `Air temperature [K]` | float | Ambient air temperature |
| `Process temperature [K]` | float | Temperature at machining point |
| `Rotational speed [rpm]` | int | Spindle rotation speed |
| `Torque [Nm]` | float | Motor torque |
| `Tool wear [min]` | int | Cumulative tool usage in minutes |
| `Machine failure` | binary | 0 = no failure, 1 = failure |
| `TWF` | binary | Tool Wear Failure |
| `HDF` | binary | Heat Dissipation Failure |
| `PWF` | binary | Power Failure |
| `OSF` | binary | Overstrain Failure |
| `RNF` | binary | Random Failure |

### Class Distribution (actual counts from data)

```
Total records     : 10,000
No failure (0)    : 9,661  (96.6%)
Failure (1)       :   339  ( 3.4%)

Breakdown by failure type:
  HDF (Heat Dissipation)  : 115
  OSF (Overstrain)        :  98
  PWF (Power Failure)     :  95
  TWF (Tool Wear)         :  46
  RNF (Random)            :  19

Machine type distribution:
  L (Low quality)    : 6,000
  M (Medium quality) : 2,997
  H (High quality)   : 1,003
```

> **⚠️ Critical imbalance:** Only 3.4% of records are failures. Every classifier you train must handle this — use `class_weight='balanced'` or `scale_pos_weight` in XGBoost, use stratified train/test splits, and evaluate with F1-macro NOT accuracy.

### Engineered Dataset (after EDA notebook)
After feature engineering, the dataset has **41 columns** including 5 raw sensors, 6 physics-inspired features, 20 time-series features, and several binary flags. Saved as `data_engineered.csv`.

After clustering, PCA components and health regime labels are added. Saved as `data_clustered.csv`.

---

## 4. What we found in EDA

These are the actual findings from analysing the data — important for understanding every downstream decision.

### Sensor correlations
```
air_temperature  ↔  process_temperature  =  +0.876  (very high)
torque           ↔  rotational_speed     =  -0.875  (very high, negative)
tool_wear        ↔  everything else      =  ~0.00   (independent)
```
This is the primary justification for PCA. Two strong correlated pairs mean PCA will find clean principal components. Expect PC1 to capture the **thermal axis** (temperatures) and PC2 to capture the **mechanical axis** (torque/RPM).

### Feature correlations with machine_failure
```
failure_mode_count    : 0.933  ← LABEL DERIVED, exclude from features!
hdf                   : 0.576  ← LABEL DERIVED
osf                   : 0.531  ← LABEL DERIVED
pwf                   : 0.523  ← LABEL DERIVED
twf                   : 0.363  ← LABEL DERIVED
torque_speed_ratio    : 0.206  ← best engineered feature
torque                : 0.191
power_W               : 0.176
high_wear_flag        : 0.174
wear_rate             : 0.130
temp_diff             : 0.112
tool_wear             : 0.105
```

### Key takeaways
- `failure_mode_count`, `twf`, `hdf`, `pwf`, `osf`, `rnf` are all derived from `machine_failure` — **including them in features causes data leakage**. Always exclude.
- `torque_speed_ratio` and `power_W` are the most useful engineered features
- `thermal_overload` flag has 44.6% positive rate — it's very common, not very discriminating alone
- `high_wear_flag` (top 10% wear) has 10.4% positive rate — more selective and useful

---

## 5. All ML Tasks — What, Why, How

### 5.1 Feature Engineering *(done in EDA notebook)*

**What:** Transform raw sensor readings into richer features that expose degradation signals not visible in point-in-time readings.

**Why:** Raw sensors alone miss trends. A machine at 200 Nm torque means nothing — but torque rising 20 Nm over the last 5 readings is a red flag. Also, domain physics give us meaningful combinations (Power = Torque × ω).

**Features created:**

| Feature | Formula | Meaning |
|---------|---------|---------|
| `temp_diff` | process_temp − air_temp | Thermal stress on machine |
| `power_W` | torque × (RPM × 2π/60) | Actual mechanical power output |
| `wear_rate` | tool_wear / RPM | Wear per unit of rotational input |
| `torque_speed_ratio` | torque / RPM | Detects overload at low speed |
| `high_wear_flag` | tool_wear ≥ 90th percentile | Binary: critical wear zone |
| `thermal_overload` | temp_diff > 10K | Binary: thermal stress flag |
| `*_roll_mean_5` | 5-step rolling mean | Smoothed trend per sensor |
| `*_roll_std_5` | 5-step rolling std | Volatility / instability signal |
| `*_delta` | first-order difference | Rate of change per sensor |
| `*_lag1` | 1-step lag | Previous reading as feature |

**What to use where:**
- PCA → 5 raw sensors only
- Clustering → PCA output
- Classification → sensors + engineered features + cluster label + PCA components
- Regression → sensors + time-series features + cluster label + PCA components

---

### 5.2 PCA — Dimensionality Reduction

**What:** Compress the 5 correlated sensor columns into 2–3 uncorrelated principal components.

**Why:** SVM performance degrades with correlated inputs. Also, PCA removes noise and gives us a clean 2D/3D space for clustering visualisation. The high sensor correlations (0.876, -0.875) mean PCA is genuinely useful here — not box-ticking.

**How:**
1. Select only: `air_temperature`, `process_temperature`, `rotational_speed`, `torque`, `tool_wear`
2. Apply `StandardScaler` (mandatory — RPM ~1500, temps ~300K, completely different scales)
3. Fit `PCA(n_components=5)` first to see all variance
4. Plot scree + cumulative variance → select n where cumulative ≥ 90%
5. Refit with chosen n_components
6. Plot loadings heatmap to interpret each PC
7. Scatter PC1 vs PC2 coloured by `machine_failure` and by `type`

**Expected outcome:** ~2 components should explain ≥90% variance given the strong correlations.

**Output:** PCA components `PC1`, `PC2` (etc.) added to dataset.

---

### 5.3 K-Means Clustering — Health Segmentation

**What:** Unsupervised segmentation of machines into health regimes without using failure labels.

**Why:** In real industrial practice, you first define health zones from operational patterns, then you label them. Clustering gives you a cluster label as an additional engineered feature for the supervised models. It also enables fleet-level monitoring — you can say "15% of your fleet is in the Critical zone right now."

**How:**
1. Input: PCA output (PC1, PC2, ...) — NOT raw sensors
2. Run K-Means for k=2 to 10, record inertia and silhouette score
3. Plot elbow + silhouette → select k (domain says k=3)
4. Fit final K-Means with k=3
5. Map raw cluster IDs → health regimes by failure rate:
   - Highest failure rate cluster → **Critical**
   - Middle → **Degraded**
   - Lowest → **Normal**
6. Encode: Normal=0, Degraded=1, Critical=2 → `Health_regime_enc`
7. Validate: plot failure rate per regime (should increase Critical > Degraded > Normal)
8. Visualise clusters in PCA space with centroids marked

**Output:** `Health_regime` (string) and `Health_regime_enc` (int) columns added to dataset. Saved as `data_clustered.csv`.

---

### 5.4 Classification — Failure Type Prediction

**What:** Predict which type of failure a machine will have (multi-class).

**Why:** Knowing *that* a failure is coming isn't enough. A maintenance engineer needs to know *what kind* — HDF requires checking cooling systems, TWF means replacing the cutting tool, OSF means reviewing load parameters. Different failures require completely different repair actions.

**Target construction:**
```python
# Create single multi-class target
df['failure_type'] = 0  # No failure
df.loc[df['twf']==1, 'failure_type'] = 1
df.loc[df['hdf']==1, 'failure_type'] = 2
df.loc[df['pwf']==1, 'failure_type'] = 3
df.loc[df['osf']==1, 'failure_type'] = 4
df.loc[df['rnf']==1, 'failure_type'] = 5
```

**Features to use:**
```
air_temperature, process_temperature, rotational_speed, torque, tool_wear,
type_enc, temp_diff, power_W, wear_rate, torque_speed_ratio,
high_wear_flag, thermal_overload, Health_regime_enc, PC1, PC2
```

**Features to EXCLUDE (data leakage):**
```
udi, type, machine_failure, twf, hdf, pwf, osf, rnf, failure_mode_count,
Product ID, all *_roll_* and *_lag* (use only if doing pure time-series approach)
```

**Models:**
- **XGBoost Classifier** — use `scale_pos_weight` or `class_weight` to handle imbalance. Produces feature importances. Strong baseline for tabular data.
- **SVM (RBF kernel)** — apply only on PCA-reduced features (PC1, PC2). Use `class_weight='balanced'`. Tune C and gamma via GridSearchCV.

**Evaluation:**
- Stratified 80/20 train/test split
- F1-macro (primary metric — accounts for imbalance)
- ROC-AUC (macro one-vs-rest)
- Confusion matrix per class
- Compare XGBoost vs SVM side by side

---

### 5.5 Regression — Remaining Useful Life

**What:** Predict how many operating steps remain before the next failure event.

**Why:** Binary alarm ("failure coming") is useful. A number ("42 hours left") is actionable. The regression module lets maintenance teams schedule repairs without unnecessary emergency shutdowns.

**RUL label engineering** (since AI4I has no explicit RUL):
```python
# Sort by UDI, then for each record compute:
# RUL = steps until next machine_failure == 1
df = df.sort_values('udi').reset_index(drop=True)
df['RUL'] = 0
failure_indices = df[df['machine_failure']==1].index.tolist()

for i in range(len(df)):
    future_failures = [f for f in failure_indices if f >= i]
    if future_failures:
        df.loc[i, 'RUL'] = future_failures[0] - i
    else:
        df.loc[i, 'RUL'] = 0  # or drop these rows
```

**Features to use:**
```
air_temperature, process_temperature, rotational_speed, torque, tool_wear,
type_enc, temp_diff, power_W, wear_rate, torque_speed_ratio,
tool_wear_roll_mean_5, tool_wear_delta, torque_roll_std_5,
Health_regime_enc, PC1, PC2
```

**Model:** XGBoost Regressor

**Evaluation:** RMSE, MAE, R²

**Pipeline logic:** This module only runs when the classifier predicts failure probability > 0.3. Don't run it on every record — that wastes computation and makes no logical sense.

---

### 5.6 Time Series Analysis

**What:** Use the sequential nature of UDI-ordered records to extract temporal patterns.

**Why:** Machines don't fail instantaneously. They degrade gradually — torque creeps up, tool wear accumulates, temperature differential widens. Point-in-time sensor readings miss this trend. Rolling and lag features capture the *trajectory* not just the current state.

**What's already done in EDA notebook:**
- `*_roll_mean_5` — smoothed trend over 5 steps
- `*_roll_std_5` — volatility signal (instability before failure)
- `*_delta` — rate of change (spike = rapid degradation)
- `*_lag1` — previous step value

**What to do in the time-series notebook:**
1. Plot `tool_wear_roll_mean_5` over `udi` for failed vs non-failed machines — show divergence
2. Plot `torque_roll_std_5` near failure events — show volatility spike
3. Plot `torque_delta` and `tool_wear_delta` in the N steps before failure
4. If a sequence model is needed: use sliding window of size 10 on UDI, create input shape (samples, 10, n_features), train LSTM or use as XGBoost features

**Key insight to communicate:** In this pipeline, time series analysis is primarily a **feature engineering step**. The rolling/lag features feed into the classifiers rather than being a standalone model. The "time series task" is satisfied by demonstrating that temporal features materially improve classification performance.

---

### 5.7 Association Rules

**What:** Mine frequent co-occurring sensor patterns that lead to specific failure types.

**Why:** Explainability. When XGBoost says "HDF incoming," the maintenance technician asks "why?" Association rules surface the exact sensor state pattern that historically caused it — a human-readable root cause alongside the model prediction.

**How:**
1. Filter to each health cluster separately (run FP-Growth within each regime)
2. Discretise sensors into bins: Low / Normal / High / Extreme
3. Apply FP-Growth (from `mlxtend`) — faster than Apriori on dense data
4. Filter: minimum confidence ≥ 0.70, lift > 1.0
5. Example output: `{High Tool Wear, High Process Temp} → {OSF}  [support=0.12, confidence=0.85, lift=2.3]`
6. Store rules as JSON — versioned alongside models
7. Encode high-confidence rules as binary features for the classifier

**Packages:** `mlxtend.frequent_patterns.fpgrowth`, `mlxtend.frequent_patterns.association_rules`

---

### 5.8 Recommendation System

**What:** Content-based filtering that maps a predicted failure type to a ranked list of repair actions.

**Why:** Bridges the gap between ML prediction and real-world action. The system doesn't just say "OSF incoming" — it returns: "Replace motor bearing (part #B2241), inspect drive belt tension, check load distribution parameters" with estimated downtime for each.

**How:**
1. Build a maintenance knowledge base (JSON file): maps each failure type + machine age bin → repair actions + part codes + urgency + repair manual ID
2. At inference: construct feature vector from (predicted failure type, machine age bin)
3. Compute cosine similarity against all knowledge base entries
4. Return top 3 matched recommendations ranked by relevance score
5. Include RUL estimate from regression stage to add urgency context

**Output:** Included in the `/predict` API response alongside failure type and RUL value.

---

## 6. Notebook Structure & File Flow

```
Notebooks (run in this order):
──────────────────────────────────────────────────────
1. eda_feature_engineering.ipynb
   Input  → predictive_maintenance.csv  (raw Kaggle data)
   Output → data_engineered.csv         (41 columns)

2. pca_clustering.ipynb
   Input  → data_engineered.csv
   Output → data_clustered.csv          (+ PC1, PC2, Health_regime, Health_regime_enc)

3. time_series.ipynb
   Input  → data_clustered.csv
   Output → visualisations + confirms rolling features are useful

4. association_rules.ipynb
   Input  → data_clustered.csv
   Output → rules.json                  (fault pattern library)

5. classification.ipynb
   Input  → data_clustered.csv + rules.json
   Output → xgboost_classifier.pkl, svm_classifier.pkl

6. regression.ipynb
   Input  → data_clustered.csv + classifier output
   Output → xgboost_regressor.pkl

7. recommendation.ipynb
   Input  → knowledge_base.json + classifier + regressor output
   Output → recommendation engine

8. api/ (FastAPI app)
   Loads all .pkl files + rules.json + knowledge_base.json
   Serves /predict, /health, /feedback, /metrics endpoints
──────────────────────────────────────────────────────
```

---

## 7. Prompts for Each Task

Copy-paste these into any AI assistant to get help with that specific task. Each prompt contains full context so the assistant doesn't need any background.

---

### 🔵 PCA + Clustering

> I have the AI4I 2020 Predictive Maintenance dataset with 10,000 records. I have already done feature engineering and my dataset (`data_engineered.csv`) has 41 columns. For PCA, I need to use only the 5 raw sensor columns: `air_temperature`, `process_temperature`, `rotational_speed`, `torque`, and `tool_wear`. These sensors have strong internal correlations (air_temp ↔ process_temp = 0.876, torque ↔ rotational_speed = -0.875), which justifies PCA. I need to: (1) StandardScale these 5 columns, (2) fit PCA and use a scree plot + cumulative variance plot to choose n_components that explain ≥90% variance, (3) plot a loadings heatmap to interpret each PC, (4) apply K-Means on the PCA output with k=3, using elbow method and silhouette score to validate, (5) map raw cluster IDs to health regime labels (Normal/Degraded/Critical) by computing each cluster's average `machine_failure` rate — highest = Critical. Save the cluster label as `Health_regime` and an ordinal encoded version `Health_regime_enc` (Normal=0, Degraded=1, Critical=2) back into the dataset. Save output as `data_clustered.csv`. Please write clean, well-commented Python code for a Jupyter notebook.

---

### 🟢 Classification

> I have `data_clustered.csv` from a predictive maintenance pipeline (AI4I 2020 dataset, 10,000 records). I need to train a multi-class classifier to predict failure type. First, create a single target column `failure_type` (0=No Failure, 1=TWF, 2=HDF, 3=PWF, 4=OSF, 5=RNF) by combining the individual binary failure columns. The dataset is severely imbalanced — 96.6% no-failure, 3.4% failure. Features to use: `air_temperature`, `process_temperature`, `rotational_speed`, `torque`, `tool_wear`, `type_enc`, `temp_diff`, `power_W`, `wear_rate`, `torque_speed_ratio`, `high_wear_flag`, `thermal_overload`, `Health_regime_enc`, `PC1`, `PC2`. Columns to EXCLUDE (data leakage): `udi`, `type`, `machine_failure`, `twf`, `hdf`, `pwf`, `osf`, `rnf`, `failure_mode_count`. Train two models: (1) XGBoost Classifier with `scale_pos_weight` for imbalance, (2) SVM with RBF kernel on PCA features only (`PC1`, `PC2`) with `class_weight='balanced'`. Use stratified 80/20 train/test split. Evaluate both with F1-macro, ROC-AUC, and confusion matrix. Plot feature importances for XGBoost. Save both models as .pkl files.

---

### 🟠 Regression (RUL)

> I have `data_clustered.csv` for predictive maintenance (AI4I 2020, 10,000 records). I need to engineer a Remaining Useful Life (RUL) target and train a regression model. Since the dataset has no explicit RUL labels, engineer them as follows: sort by `udi`, then for each record compute RUL = number of steps (rows) until the next `machine_failure == 1` event (forward-looking). Records after the last failure can be assigned RUL=0. Features to use: `air_temperature`, `process_temperature`, `rotational_speed`, `torque`, `tool_wear`, `type_enc`, `temp_diff`, `power_W`, `wear_rate`, `torque_speed_ratio`, `tool_wear_roll_mean_5`, `tool_wear_delta`, `torque_roll_std_5`, `Health_regime_enc`, `PC1`, `PC2`. Train an XGBoost Regressor. Evaluate with RMSE, MAE, and R². Also add a note in the notebook that in the deployed pipeline, this regression module only activates when the upstream classifier predicts failure probability > 0.3. Save the trained model as `rul_regressor.pkl`.

---

### 🔴 Time Series Analysis

> I have `data_clustered.csv` with time-series features already engineered (rolling mean window=5, rolling std, first-order delta, and lag-1 for all 5 sensors). The `udi` column is the sequential time index. I need to create a time series analysis section in a Jupyter notebook that: (1) plots `tool_wear_roll_mean_5` and `torque_roll_std_5` over `udi`, comparing failed vs non-failed machines to show divergence, (2) plots `torque_delta` and `tool_wear_delta` in the N=20 steps before each failure event to show the pre-failure spike pattern, (3) trains a baseline binary classifier (`machine_failure` target) using a sliding window of size 10 on `udi` with XGBoost, comparing performance with and without rolling features to quantify their contribution, (4) if time permits, train a simple LSTM on windowed sequences. The key point to make in this notebook: time series analysis here is primarily a feature engineering step — the rolling/lag features feed into the main classifiers. Clearly state this in a markdown cell.

---

### 🟣 Association Rules

> I have `data_clustered.csv` from a predictive maintenance pipeline. I need to mine association rules using FP-Growth from `mlxtend`. Steps: (1) filter data by `Health_regime` and run FP-Growth separately within each cluster (Normal, Degraded, Critical), (2) discretise the 5 sensor columns (`air_temperature`, `process_temperature`, `rotational_speed`, `torque`, `tool_wear`) into 4 bins: Low/Normal/High/Extreme using domain-informed or quantile thresholds, (3) create a transaction-style DataFrame where each row is a machine record and each column is a discretised sensor-bin (one-hot encoded), (4) run `fpgrowth` with min_support=0.05 then `association_rules` with min_confidence=0.70 and lift > 1.0, (5) filter rules where the consequent is one of the failure types (twf, hdf, pwf, osf, rnf), (6) display top rules sorted by confidence, (7) save all rules to `rules.json`. Use `mlxtend.frequent_patterns.fpgrowth` and `mlxtend.frequent_patterns.association_rules`.

---

### 🟡 Recommendation System

> I need to build a content-based recommendation system for a predictive maintenance pipeline. It takes as input: (1) predicted failure type from a classifier (one of: TWF, HDF, PWF, OSF, RNF), (2) machine age bin (Young: 0-1000 UDI, Mid: 1000-5000, Old: 5000+), (3) RUL estimate from a regression model. First, create a maintenance knowledge base as a JSON file with at least 3 entries per failure type, each containing: failure_type, age_bin, recommended_actions (list of 3 strings), part_codes (list), urgency_level (1-3), repair_manual_id, estimated_downtime_hours. Then implement a content-based filtering function: encode failure_type and age_bin as feature vectors, compute cosine similarity between the query vector and all knowledge base entries, return the top 3 matches. Include the RUL value in the output to contextualise urgency. The function signature should be: `get_recommendations(failure_type: str, age_bin: str, rul: float) -> list[dict]`. This will be called from the FastAPI `/predict` endpoint.

---

## 8. MLOps Layer

This is the production wrapper around all the notebooks. Required for the course.

### FastAPI Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/predict` | POST | Accept sensor readings → return failure type + RUL + recommendations |
| `/health` | GET | Model version + system status |
| `/feedback` | POST | Accept corrected labels for retraining buffer |
| `/metrics` | GET | Drift stats + prediction distribution |

### Prefect Workflows
- **Training pipeline:** data ingest → validate → feature engineering → train → evaluate → save
- **Monitoring pipeline:** drift detection → feedback review → conditional retrain → model swap

### Data Drift Detection
Compare incoming sensor distributions vs training distributions using KS-test. Trigger alert + potential retrain when p-value drops below threshold.

### CI/CD (GitHub Actions)
Automates: code lint → unit tests → ML tests (DeepChecks) → Docker build → deploy.

### Docker
One Dockerfile for the FastAPI service. Optional Docker Compose for API + Prefect + database together.

---

## 9. Tech Stack

| Category | Tool |
|----------|------|
| Language | Python 3.10+ |
| ML models | scikit-learn, XGBoost |
| Association rules | mlxtend (FP-Growth) |
| Data | pandas, NumPy |
| Visualisation | matplotlib, seaborn |
| API serving | FastAPI |
| Orchestration | Prefect |
| Dashboard | Streamlit |
| CI/CD | GitHub Actions |
| Containers | Docker |
| Model tracking | MLflow or custom JSON store |
| Testing | DeepChecks or pytest |

---

## 10. Key Things to Remember

**Data leakage — the most common mistake:**
Never include `twf`, `hdf`, `pwf`, `osf`, `rnf`, `failure_mode_count`, or `machine_failure` as input features in any classifier. These are derived from the target. Including them gives artificially perfect results that will fail in deployment.

**Class imbalance:**
96.6% of records have no failure. Never use accuracy as your metric — a model that always predicts "no failure" gets 96.6% accuracy and is completely useless. Always use F1-macro and ROC-AUC.

**Pipeline order matters:**
You cannot do classification before clustering (cluster label is a feature). You cannot do regression before classification (regression only runs when classifier flags a failure). Respect the sequence.

**PCA inputs:**
PCA uses ONLY the 5 raw sensor columns, scaled. Not engineered features. Not encoded type. Just the sensors.

**K-Means inputs:**
K-Means uses the PCA output. Not raw sensors, not the full engineered dataset. PCA output first, then cluster.

**Time series = feature engineering here:**
Don't over-complicate the time series task. The rolling/lag features you already created ARE the time series component. The task is to demonstrate they improve model performance.

**RUL is engineered:**
AI4I doesn't have explicit RUL labels. You compute them from the data using forward-looking failure detection. Document this clearly in your report.

---

*Generated for MachineGuard+ | AI221 ML Engineering | GIKI | April 2026*
