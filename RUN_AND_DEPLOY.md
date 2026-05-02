# MachineGuard+ Run and Deploy Guide

This guide covers local setup, API run commands, testing, Prefect pipeline execution, DeepChecks, Docker usage, and CI/CD flow.

## 1) Prerequisites

- Python 3.10+ (3.11 recommended)
- `pip`
- Docker + Docker Compose plugin
- Git

From repo root:

```bash
cd /home/crazi_grace/MINE/Codezzz/ZedX-ML-Lalala
```

## 2) Local Environment Setup

Create and activate virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 3) Run FastAPI Locally

Start API:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- Swagger: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

## 3.1) Run Streamlit With Email Notifications

Full setup guide:

- `EMAIL_NOTIFICATIONS_SETUP.md`

Start Streamlit:

```bash
streamlit run streamlit_app.py
```

The sidebar has an **Email notifications** section. Enable it and enter a recipient to send success/failure notifications after live prediction or batch trend runs.

Configure SMTP through environment variables:

```bash
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USERNAME="your_email@example.com"
export SMTP_PASSWORD="your_app_password"
export SMTP_SENDER="your_email@example.com"
export NOTIFICATION_RECIPIENT="maintenance-team@example.com"
```

Or use `.streamlit/secrets.toml` locally:

```toml
[email]
smtp_host = "smtp.gmail.com"
smtp_port = "587"
smtp_username = "your_email@example.com"
smtp_password = "your_app_password"
smtp_sender = "your_email@example.com"
notification_recipient = "maintenance-team@example.com"
```

## 4) Core API Endpoints

- `POST /predict`
- `POST /predict/classification`
- `POST /predict/regression`
- `POST /predict/timeseries` (CSV upload)
- `GET /health`
- `GET /metrics`
- `POST /feedback`

## 5) Run Tests

```bash
pytest -q
```

## 6) Run DeepChecks

```bash
python scripts/deepchecks_runner.py
```

Expected output artifact:

- `outputs/deepchecks_report.json`

## 7) Run Prefect Pipeline

Run pipeline (no notebook execution):

```bash
python main_pipeline.py
```

Optional programmatic run:

```bash
python -c "from main_pipeline import machineguard_pipeline; print(machineguard_pipeline(execute_notebooks=False))"
```

Expected output artifact:

- `outputs/metrics.json`

## 8) Docker (Recommended Runtime Path)

### Build images

```bash
docker compose build
```

### Start services

```bash
docker compose up -d
```

### Check status

```bash
docker compose ps
```

### View logs

```bash
docker compose logs -f fastapi
docker compose logs -f prefect
```

### Stop services

```bash
docker compose down
```

### Rebuild from scratch

```bash
docker compose down --remove-orphans
docker compose build --no-cache
docker compose up -d
```

## 9) CI/CD Pipeline (GitHub Actions)

Workflow file:

- `.github/workflows/main.yml`

Pipeline stages:

1. Install dependencies
2. Run tests
3. Run DeepChecks
4. Run Prefect pipeline
5. Validate `outputs/metrics.json`
6. Build Docker image

### Trigger CI/CD

CI triggers automatically on:

- Push to any branch
- Pull request to any branch

### Manual verification commands (same as CI steps)

```bash
pip install -r requirements.txt
pytest -q
python scripts/deepchecks_runner.py
python main_pipeline.py
test -f outputs/metrics.json
docker build -t machineguard:ci .
```

## 10) Common Troubleshooting

### API fails because model file missing

Ensure these exist in `models/`:

- `pca.pkl`
- `kmeans.pkl`
- `scaler.pkl`
- `xgb_classifier.pkl`
- `rules.json`
- `knowledge_base.json`

Optional if available:

- `xgb_regressor.pkl` or `tool_wear_regressor.pkl`
- `svm_pipeline.pkl` or `svm_classifier.pkl`

### DeepChecks issues with NumPy compatibility

The runner already includes a compatibility shim for `np.Inf`.

### Docker build too slow

- Use regular `docker compose build` first (without `--no-cache`)
- Ensure stable network for large ML wheel downloads
- Keep `.dockerignore` unchanged to avoid large build contexts

## 11) One-Command Daily Flow

```bash
source .venv/bin/activate && pytest -q && python scripts/deepchecks_runner.py && python main_pipeline.py
```

Then run API:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
