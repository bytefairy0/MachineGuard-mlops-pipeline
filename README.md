# MachineGuard+

Adaptive predictive-maintenance ML system for industrial machine health monitoring. The project combines FastAPI inference, a Streamlit operator UI, Prefect orchestration, validation checks, saved ML artefacts, Docker runtime support, and GitHub Actions CI.

## What It Does

- Predicts machine failure probability and failure type from live sensor readings.
- Estimates tool wear and urgency when risk crosses the configured threshold.
- Assigns machines to health regimes with PCA + KMeans clustering.
- Supports CSV sequence upload for simple time-series trend inference.
- Returns maintenance recommendations and triggered association rules when artefacts are available.
- Tracks lightweight runtime metrics, feedback, and drift signals through the API.

## Project Structure

```text
.
├── src/api/main.py                 # FastAPI app, schemas, model loading, inference endpoints
├── streamlit_app.py                # Streamlit dashboard/operator UI
├── main_pipeline.py                # Prefect ML validation/evaluation pipeline
├── scripts/deepchecks_runner.py    # DeepChecks validation runner
├── models/                         # Trained model and knowledge artefacts
├── data/raw_data/                  # Source predictive-maintenance dataset
├── data/cleaned_data/              # Prepared EDA and clustered datasets
├── notebooks/                      # Experiment notebooks for EDA, models, rules, recommender
├── outputs/                        # Generated metrics and validation reports
├── plots/                          # Generated visualizations
├── tests/                          # API and pipeline tests
├── Dockerfile                      # Runtime image for API/UI/pipeline services
├── docker-compose.yml              # Multi-service local Docker setup
├── requirements.txt                # Full dev/CI dependencies
├── requirements.runtime.txt        # Leaner runtime Docker dependencies
└── .github/workflows/main.yml      # GitHub Actions CI pipeline
```

## Models And Artefacts

| File | Purpose |
| --- | --- |
| `models/xgb_classifier.pkl` | Main failure classifier. |
| `models/svm_pipeline.pkl` | Alternative/supporting classifier pipeline. |
| `models/tool_wear_regressor.pkl` | Tool-wear regression model used by the API. |
| `models/xgb_regressor.json` | Native XGBoost regressor artefact fallback. |
| `models/scaler.pkl` | Scales core sensor features before PCA. |
| `models/pca.pkl` | Reduces sensor signals into principal components. |
| `models/kmeans.pkl` | Assigns health regimes/clusters. |
| `models/rules.json` | Association rules used for triggered maintenance patterns. |
| `models/knowledge_base.json` | Recommendation knowledge base. |
| `models/label_encoder.pkl` | Label encoding artefact retained for model compatibility. |

Core API input fields are `machine_id`, `machine_type`, `air_temperature`, `process_temperature`, `rotational_speed`, `torque`, `tool_wear`, and optional `machine_age_bin`.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Useful URLs:

- API root: `http://127.0.0.1:8000/`
- Swagger docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`
- Runtime metrics: `http://127.0.0.1:8000/metrics`

Run the Streamlit UI:

```bash
streamlit run streamlit_app.py
```

## API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Service status/version. |
| `GET` | `/health` | Loaded artefacts and service health. |
| `POST` | `/predict` | Full prediction response with failure, wear, rules, and recommendations. |
| `POST` | `/predict/classification` | Classification-focused prediction. |
| `POST` | `/predict/regression` | Tool-wear regression-focused prediction. |
| `POST` | `/predict/timeseries` | CSV upload for sequence/trend prediction. |
| `POST` | `/feedback` | Store corrected engineer feedback. |
| `GET` | `/metrics` | Prediction counts, drift heuristic, and health-regime distribution. |

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "machine_id": "M001",
    "machine_type": "L",
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551,
    "torque": 42.8,
    "tool_wear": 108,
    "machine_age_bin": "Mid"
  }'
```

## Pipeline And Validation

Run tests:

```bash
pytest -q
```

Run DeepChecks:

```bash
python scripts/deepchecks_runner.py
```

Run the Prefect pipeline:

```bash
python main_pipeline.py
```

Generated outputs:

- `outputs/metrics.json`
- `outputs/deepchecks_report.json`

## Docker

Build the runtime image:

```bash
docker build -t machineguard:local .
```

Run only the FastAPI container:

```bash
docker run --rm -p 8000:8000 machineguard:local
```

Use Docker Compose for the full local stack:

```bash
docker compose build
docker compose up -d
```

Compose services:

| Service | Container | Host URL |
| --- | --- | --- |
| `fastapi` | `machineguard-api` | `http://127.0.0.1:8080/docs` |
| `streamlit` | `machineguard-ui` | `http://127.0.0.1:8501` |
| `prefect-server` | `machineguard-prefect-server` | `http://127.0.0.1:4200` |
| `prefect` | `machineguard-prefect` | Runs `python main_pipeline.py` |

Common Compose commands:

```bash
docker compose ps
docker compose logs -f fastapi
docker compose logs -f streamlit
docker compose logs -f prefect
docker compose down
```

Rebuild cleanly:

```bash
docker compose down --remove-orphans
docker compose build --no-cache
docker compose up -d
```

## Pulling A Published Image

The current GitHub Actions workflow builds a Docker image for validation but does not publish it to a registry yet. If a release image is later pushed to GHCR or Docker Hub, the usage will look like this:

```bash
docker pull ghcr.io/bytefairy0/zedx-ml-lalala:latest
docker run --rm -p 8000:8000 ghcr.io/bytefairy0/zedx-ml-lalala:latest
```

For Compose, replace `build: .` with the published `image:` name for the services that should use the remote image.

## Environment Variables

Email notifications in the Streamlit app can be configured through `.env` or shell environment variables:

```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@example.com
SMTP_PASSWORD=your_app_password
SMTP_SENDER=your_email@example.com
SMTP_USE_TLS=true
SMTP_USE_SSL=false
NOTIFICATION_RECIPIENT=maintenance-team@example.com
```

Keep `.env` local and do not commit real credentials.

## CI/CD

Workflow: `.github/workflows/main.yml`

Triggers:

- Push to any branch.
- Pull request to any branch.

Current CI stages:

1. Checkout repository.
2. Set up Python `3.11`.
3. Install `requirements.txt`.
4. Run `pytest -q`.
5. Run DeepChecks via `python scripts/deepchecks_runner.py`.
6. Run Prefect pipeline via `python main_pipeline.py`.
7. Validate `outputs/metrics.json` structure.
8. Build Docker image as `machineguard:ci`.

Current limitation: CI verifies the image build, but it does not push to GHCR/Docker Hub or deploy to a server. Add registry login and `docker push` steps when publishing is required.

## Daily Developer Flow

```bash
source .venv/bin/activate
pytest -q
python scripts/deepchecks_runner.py
python main_pipeline.py
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
TADA :)
