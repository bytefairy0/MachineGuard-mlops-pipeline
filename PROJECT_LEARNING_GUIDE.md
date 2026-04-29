# MachineGuard+ Learning Guide

This guide explains this project as if we built it and now need to teach it to our own team. The goal is not only "how to run it", but "what is happening, why each part exists, and how all ML + FastAPI + Docker + CI/CD pieces connect".

Project name: MachineGuard+

Project idea: Predictive maintenance for machines. We take machine sensor readings, understand the current health condition, predict failure risk/type, estimate tool wear, and recommend what action should be taken.

Think of it like this:

```text
Machine sensors come in
        |
        v
Cleaned / engineered data
        |
        v
ML models understand patterns
        |
        v
FastAPI exposes those models as web endpoints
        |
        v
Docker packages the project so it runs the same everywhere
        |
        v
CI/CD tests the project automatically when code is pushed
```

## 1. What Problem Are We Solving?

Factories have machines that can fail because of heat, power load, overstrain, random failures, or tool wear. Instead of waiting for breakdowns, we want a system that answers:

- Is this machine likely to fail?
- What type of failure is expected?
- Is the machine in a normal, degraded, or critical health regime?
- How much tool wear is expected?
- What maintenance action should we recommend?

This is why the project uses multiple ML techniques together:

- Classification predicts failure class/type.
- Regression predicts a continuous value such as tool wear.
- Clustering groups machines into health regimes.
- Association rules discover common "if this pattern happens, that failure often happens" rules.
- Recommendation suggests actions based on failure type and machine context.
- Time-series logic looks at trends over readings, not only one row.

## 2. Important Folders and Files

```text
data/
  raw_data/
    predictive_maintenance_data.csv
  cleaned_data/
    eda_data.csv
    clustered_data.csv

notebooks/
  eda.ipynb
  clustering.ipynb
  classification.ipynb
  Tool_Wear_Regression.ipynb
  Time_series.ipynb
  Association_rules.ipynb
  MachineGuard_Recommender.ipynb

models/
  pca.pkl
  scaler.pkl
  kmeans.pkl
  xgb_classifier.pkl
  svm_pipeline.pkl
  rules.json
  knowledge_base.json

src/api/main.py
  FastAPI app. This is the web/API layer around the ML system.

main_pipeline.py
  Prefect pipeline. Validates artifacts and computes metrics.

scripts/deepchecks_runner.py
  Runs model/data quality checks.

Dockerfile
  Instructions for building the app image.

docker-compose.yml
  Runs FastAPI and pipeline services together.

.github/workflows/main.yml
  CI pipeline for GitHub Actions.

tests/
  Automated tests for API and pipeline behavior.
```

## 3. Full Project Flow

At a high level, our project has two flows:

1. Training / experimentation flow
2. Serving / prediction flow

### Training / Experimentation Flow

This mainly happens in notebooks and the pipeline.

```text
Raw CSV data
   |
   v
EDA notebook
   - cleans column names
   - studies distributions and correlations
   - creates engineered features
   |
   v
Clustering notebook
   - scales data
   - applies PCA
   - trains K-Means
   - assigns health regimes
   |
   v
Classification notebook
   - trains models like XGBoost / SVM
   - predicts failure or failure type
   |
   v
Regression notebook
   - predicts numeric tool wear
   |
   v
Association rules notebook
   - finds common failure patterns
   |
   v
Recommendation notebook
   - creates knowledge base of actions
   |
   v
Saved outputs in models/ and data/cleaned_data/
```

Saved models are then loaded by the API.

### Serving / Prediction Flow

This happens inside `src/api/main.py`.

When someone sends data to `POST /predict`, the API does this:

```text
User sends sensor JSON
   |
   v
Pydantic validates input shape
   |
   v
engineer_features()
   - creates temp_diff
   - creates power_W
   - creates wear_rate
   - creates torque_speed_ratio
   - creates high_wear_flag
   - creates thermal_overload
   |
   v
apply_pca()
   - scales raw sensor values
   - converts them into PCA components
   |
   v
assign_cluster()
   - K-Means predicts health regime
   - Normal / Degraded / Critical
   |
   v
predict_failure()
   - classifier predicts failure probability and failure type
   |
   v
estimate_tool_wear()
   - regression predicts tool wear if failure risk is high enough
   |
   v
match_rules()
   - association rules explain patterns
   |
   v
get_recommendations()
   - returns maintenance suggestions
   |
   v
API returns JSON response
```

So FastAPI is not doing ML itself. FastAPI is the delivery system. The ML intelligence comes from models and helper functions.

## 4. What Each ML Technique Is Doing Here

### 4.1 Classification

Classification predicts a category.

In this project, classification answers:

```text
What failure type is this machine showing?
```

Possible labels in the API:

- `No Failure`
- `TWF`: Tool Wear Failure
- `HDF`: Heat Dissipation Failure
- `PWF`: Power Failure
- `OSF`: Overstrain Failure
- `RNF`: Random Failure

The main classifier artifact is:

```text
models/xgb_classifier.pkl
```

There is also:

```text
models/svm_pipeline.pkl
```

The API mainly uses the XGBoost classifier for predictions.

Important idea:

The dataset is imbalanced. Most machines are not failing. So accuracy alone can be misleading. A model that says "No Failure" for everything may get high accuracy but be useless. That is why metrics like F1-score are important.

### 4.2 Regression

Regression predicts a number.

In this project, regression answers:

```text
What tool wear value do we expect?
```

Example:

```text
predicted_tool_wear = 172.4
```

The API uses this value to create urgency:

```text
wear >= 200  -> CRITICAL
wear >= 150  -> HIGH
wear >= 100  -> MEDIUM
else         -> LOW
```

In the current API, regression runs only when failure probability is at least `0.3`. That threshold is stored as:

```python
FAILURE_THRESHOLD = 0.3
```

Meaning:

If the machine seems mostly safe, we do not need to estimate emergency tool wear.

### 4.3 Clustering

Clustering is unsupervised learning. It does not need labels.

In this project, clustering answers:

```text
Which health regime does this machine belong to?
```

The health regimes are:

- Normal
- Degraded
- Critical

The saved clustering model is:

```text
models/kmeans.pkl
```

Before clustering, the project applies PCA:

```text
Raw sensor columns -> scaler -> PCA -> K-Means cluster
```

Why?

Because sensors have different scales. For example, RPM can be around 1500 while torque may be around 40. If we do not scale, large-number columns dominate the math. PCA also compresses correlated sensor data into cleaner components.

### 4.4 PCA

PCA means Principal Component Analysis.

It is not a prediction model. It transforms features.

In this project, PCA answers:

```text
Can we compress sensor readings into fewer important directions?
```

Files:

```text
models/scaler.pkl
models/pca.pkl
```

The API applies PCA on the first five sensor columns:

- air_temperature
- process_temperature
- rotational_speed
- torque
- tool_wear

Then K-Means uses the PCA output for cluster assignment.

### 4.5 Association Rules

Association rules find patterns like:

```text
IF high torque AND high tool wear THEN overstrain failure is likely
```

This is similar to market basket analysis:

```text
IF customer buys bread and butter THEN they may buy jam
```

But in this project:

```text
IF machine has certain sensor/failure pattern THEN certain failure type is common
```

Rules are stored in:

```text
models/rules.json
```

The API function `match_rules()` checks which rules match the predicted cluster and failure type. These rules help with explainability. They tell us why a failure may be happening, not only what the model predicted.

### 4.6 Recommendation System

The recommendation system suggests actions.

It answers:

```text
Given this failure type and machine age, what should the maintenance team do?
```

The knowledge base is stored in:

```text
models/knowledge_base.json
```

The API function `get_recommendations()` searches this file. If the failure type matches, it returns the top actions. If age bin also matches, the recommendation gets higher relevance.

Example idea:

```text
failure_type = HDF
age_bin = High

Recommended:
- inspect cooling system
- check temperature control
- schedule maintenance
```

### 4.7 Time Series

Time series means data ordered by time.

In this project, time-series ideas are used in two ways:

1. In notebooks, we create lag/rolling features.
2. In the API, `/predict/timeseries` accepts a CSV and checks the latest row plus tool wear trend.

Endpoint:

```text
POST /predict/timeseries
```

This endpoint expects a CSV file with columns like:

- machine_id
- machine_type
- air_temperature
- process_temperature
- rotational_speed
- torque
- tool_wear
- machine_age_bin

The API uses the last row as the latest machine state and calculates:

```text
tool_wear_trend = last_tool_wear - first_tool_wear
```

This is a simple trend signal. More advanced time-series forecasting could be added later.

## 5. FastAPI: What It Is Doing

FastAPI is a Python web framework.

In our project, FastAPI turns ML code into a usable web service.

Without FastAPI:

```text
Only Python scripts and notebooks can use the model.
```

With FastAPI:

```text
Any app can send HTTP requests and get predictions.
```

Frontend, mobile app, dashboard, another backend, or even Postman can call the API.

### Main API File

```text
src/api/main.py
```

This file does five big jobs:

1. Creates the FastAPI app.
2. Loads model artifacts from `models/`.
3. Defines input/output schemas using Pydantic.
4. Defines prediction helper functions.
5. Defines API endpoints.

### What Is an Endpoint?

An endpoint is a URL path where users can call the API.

Important endpoints in this project:

```text
GET  /
GET  /health
POST /predict
POST /predict/classification
POST /predict/regression
POST /predict/timeseries
POST /feedback
GET  /metrics
```

### `/health`

This checks whether the service is alive.

It returns:

- status
- model version
- loaded artifacts
- prediction count
- feedback buffer size

Use it when you want to ask:

```text
Is my API running correctly?
```

### `/predict`

This is the main endpoint.

Input example:

```json
{
  "machine_id": "M001",
  "machine_type": "L",
  "air_temperature": 298.1,
  "process_temperature": 308.6,
  "rotational_speed": 1551,
  "torque": 42.8,
  "tool_wear": 108,
  "machine_age_bin": "Mid"
}
```

Output includes:

```json
{
  "machine_id": "M001",
  "failure_probability": 0.1234,
  "failure_type": "No Failure",
  "health_regime": "Normal",
  "predicted_tool_wear": null,
  "urgency_level": null,
  "recommendations": [
    {
      "message": "Machine is healthy"
    }
  ],
  "model_version": "1.0.0"
}
```

### Pydantic Models

Pydantic checks input and output data shapes.

In `src/api/main.py`, these classes matter:

- `SensorInput`
- `PredictionResponse`
- `FeedbackInput`
- `MetricsResponse`
- `TaskPredictionResponse`

Example:

```python
class SensorInput(BaseModel):
    machine_id: str
    machine_type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: int
    torque: float
    tool_wear: int
    machine_age_bin: Optional[str] = "Medium"
```

This means FastAPI will reject bad input automatically. For example, if `torque` is missing, the API will return a validation error.

### Uvicorn

FastAPI defines the app, but Uvicorn runs the app as a server.

Command:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Meaning:

- `src.api.main:app`: find the `app` object inside `src/api/main.py`
- `--host 0.0.0.0`: allow access from outside the container/machine
- `--port 8000`: run on port 8000
- `--reload`: restart automatically when code changes

### Swagger Docs

FastAPI automatically gives API docs at:

```text
http://127.0.0.1:8000/docs
```

If using Docker Compose in this project:

```text
http://127.0.0.1:8080/docs
```

Because Docker maps host port `8080` to container port `8000`.

## 6. Docker: What It Is and How It Works Here

Docker packages the project with its environment.

Problem without Docker:

```text
"It runs on my laptop but not on yours."
```

Docker solution:

```text
Package Python version, dependencies, code, and startup command together.
```

### Dockerfile Explained

File:

```text
Dockerfile
```

Current Dockerfile:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
COPY requirements.runtime.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.runtime.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Line by line:

- `FROM python:3.11-slim`: start with a lightweight Python 3.11 Linux image.
- `WORKDIR /app`: inside the container, work inside `/app`.
- `ENV PYTHONDONTWRITEBYTECODE=1`: do not create `.pyc` files.
- `ENV PYTHONUNBUFFERED=1`: print logs immediately.
- `COPY requirements.runtime.txt .`: copy dependency list into container.
- `RUN pip install ...`: install required Python packages.
- `COPY . .`: copy project files into container.
- `EXPOSE 8000`: document that the app uses port 8000.
- `CMD [...]`: default command to start FastAPI with Uvicorn.

### Docker Image vs Container

Image:

```text
Blueprint / packaged template.
```

Container:

```text
Running instance of that image.
```

Example:

```bash
docker build -t machineguard .
docker run -p 8000:8000 machineguard
```

### Docker Compose

Docker Compose runs multiple services together.

File:

```text
docker-compose.yml
```

This project has two services:

```text
fastapi
prefect
```

### `fastapi` Service

```yaml
fastapi:
  build: .
  container_name: machineguard-api
  command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
  ports:
    - "8080:8000"
  volumes:
    - ./:/app
    - ./models:/app/models
    - ./outputs:/app/outputs
```

Meaning:

- Build image from current directory.
- Run the API.
- Map host `8080` to container `8000`.
- Mount local project into `/app`, so code changes are visible.
- Mount `models` and `outputs` so artifacts are shared.

Why `8080:8000`?

```text
Your browser uses localhost:8080
Container app runs on port 8000
Docker forwards traffic between them
```

### `prefect` Service

```yaml
prefect:
  build: .
  container_name: machineguard-prefect
  command: python main_pipeline.py
  depends_on:
    - fastapi
```

This runs the ML pipeline script inside a container.

It validates required files, computes metrics, and writes:

```text
outputs/metrics.json
```

### Useful Docker Commands

Build:

```bash
docker compose build
```

Start:

```bash
docker compose up
```

Start in background:

```bash
docker compose up -d
```

See running containers:

```bash
docker compose ps
```

See logs:

```bash
docker compose logs -f fastapi
```

Stop:

```bash
docker compose down
```

## 7. CI/CD: What It Is and How It Works Here

CI/CD means Continuous Integration / Continuous Deployment.

For this project, we mainly have CI, not full deployment.

CI asks:

```text
When someone pushes code, does the project still work?
```

CD asks:

```text
If tests pass, should we automatically deploy it?
```

This repo has GitHub Actions CI in:

```text
.github/workflows/main.yml
```

### When Does CI Run?

It runs on:

```yaml
on:
  push:
    branches: ["**"]
  pull_request:
    branches: ["**"]
```

Meaning:

- Every push to any branch
- Every pull request to any branch

### What Does the CI Pipeline Do?

```text
1. Checkout repo
2. Setup Python 3.11
3. Install dependencies
4. Run tests
5. Run DeepChecks
6. Run Prefect pipeline
7. Validate outputs/metrics.json
8. Build Docker image
```

### Why Each Step Matters

Checkout:

```text
Downloads your repo code into the GitHub Actions runner.
```

Setup Python:

```text
Uses Python 3.11, same major version as Docker.
```

Install dependencies:

```text
Installs packages from requirements.txt.
```

Run tests:

```text
Checks that API endpoints and pipeline functions still work.
```

Run DeepChecks:

```text
Checks data/model quality and saves outputs/deepchecks_report.json.
```

Run Prefect pipeline:

```text
Runs main_pipeline.py to validate ML artifacts and compute metrics.
```

Validate metrics:

```text
Confirms outputs/metrics.json exists and contains required keys.
```

Build Docker image:

```text
Confirms the project can be packaged into a container.
```

If any step fails, GitHub marks the workflow as failed. This protects the project from broken code being merged.

## 8. Prefect: What It Is Doing Here

Prefect is an orchestration tool.

Simple meaning:

```text
It organizes pipeline steps and runs them as tasks.
```

File:

```text
main_pipeline.py
```

Important pieces:

- `@task`: marks a function as a pipeline task.
- `@flow`: marks the main pipeline flow.

Main flow:

```python
@flow(name="machineguard-multi-ml-pipeline")
def machineguard_pipeline(execute_notebooks: bool = False):
    validate_expected_outputs_result = validate_expected_outputs()
    metrics = evaluate_multi_task_metrics()
    metrics_path = save_metrics(metrics)
```

So when we run:

```bash
python main_pipeline.py
```

It:

1. Checks required data/model artifacts exist.
2. Evaluates ML task metrics.
3. Saves results to `outputs/metrics.json`.

This is not the same as FastAPI. FastAPI serves predictions. Prefect runs the ML pipeline.

## 9. DeepChecks: What It Is Doing Here

DeepChecks is used for model/data validation.

File:

```text
scripts/deepchecks_runner.py
```

It loads:

```text
data/cleaned_data/clustered_data.csv
models/xgb_classifier.pkl
```

Then it runs checks like:

- data integrity
- train/test performance
- model information

Output:

```text
outputs/deepchecks_report.json
```

This is part of ML quality control. Normal software tests check if code works. DeepChecks helps check if ML data/model behavior looks okay.

## 10. Tests: What They Check

Tests are in:

```text
tests/
```

### API Tests

File:

```text
tests/test_api_main.py
```

They check:

- `/health` returns expected fields.
- `/predict` returns correct response shape.
- `/metrics` updates after prediction.
- `/predict/classification` works.
- `/predict/regression` works.
- `/predict/timeseries` accepts CSV upload.

### Pipeline Test

File:

```text
tests/test_main_pipeline.py
```

It checks:

- expected artifacts exist
- metrics can be computed
- `outputs/metrics.json` can be saved

Run tests:

```bash
pytest -q
```

## 11. How to Run the Project

### Local Python Run

Create virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run API:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open docs:

```text
http://127.0.0.1:8000/docs
```

Run tests:

```bash
pytest -q
```

Run pipeline:

```bash
python main_pipeline.py
```

### Docker Run

Build:

```bash
docker compose build
```

Start:

```bash
docker compose up
```

Open docs:

```text
http://127.0.0.1:8080/docs
```

Stop:

```bash
docker compose down
```

## 12. Important "Who Does What?" Summary

```text
notebooks/
  Experiment, train, analyze, create artifacts

models/
  Stores trained ML models and JSON knowledge files

src/api/main.py
  Loads artifacts and exposes predictions through HTTP endpoints

main_pipeline.py
  Runs validation and metrics pipeline using Prefect

scripts/deepchecks_runner.py
  Runs ML quality checks

tests/
  Checks API and pipeline do not break

Dockerfile
  Packages app into a Docker image

docker-compose.yml
  Runs API and pipeline containers

.github/workflows/main.yml
  Runs automated CI checks on GitHub
```

## 13. What You Should Study Next

To understand this project completely, study these topics in this order:

1. HTTP basics
   - Request
   - Response
   - JSON
   - GET vs POST
   - Status codes

2. FastAPI basics
   - `FastAPI()`
   - route decorators like `@app.get()` and `@app.post()`
   - Pydantic models
   - request body validation
   - Swagger docs

3. Model serving
   - `joblib.load()`
   - loading `.pkl` files
   - using saved sklearn/XGBoost models for inference
   - keeping training features and inference features consistent

4. Docker basics
   - image
   - container
   - Dockerfile
   - port mapping
   - volumes
   - Docker Compose

5. CI/CD basics
   - GitHub Actions
   - workflow YAML
   - jobs and steps
   - why tests run before merge/deploy

6. MLOps basics
   - model artifacts
   - metrics artifacts
   - data validation
   - drift monitoring
   - retraining loop

7. Prefect basics
   - flow
   - task
   - orchestration
   - retries
   - pipeline scheduling

## 14. The Most Important Concept

The project is not "just ML notebooks".

It is a small ML system.

The difference:

```text
Notebook project:
  Train model and show metrics.

ML system:
  Train model, save it, serve it through an API, test it, package it,
  validate it, and make sure it can run again reliably.
```

That is why this project has:

- ML notebooks for learning and training.
- `models/` for saved artifacts.
- FastAPI for serving predictions.
- Docker for reproducible running.
- Tests for correctness.
- DeepChecks for ML validation.
- Prefect for pipeline orchestration.
- GitHub Actions for CI.

## 15. Quick Mental Model

If someone asks "what is happening in this project?", answer:

```text
We built a predictive maintenance ML system. The notebooks train and analyze
models for classification, regression, clustering, association rules, time-series
signals, and recommendations. The trained artifacts are saved in models/.
FastAPI loads those artifacts and exposes endpoints so users can send machine
sensor readings and receive failure predictions, health regimes, tool wear
estimates, and maintenance recommendations. Docker packages the API and pipeline
so they can run consistently. GitHub Actions runs tests, DeepChecks, the Prefect
pipeline, and Docker build automatically whenever code is pushed.
```

