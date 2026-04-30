import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from prefect import flow, get_run_logger, task
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, silhouette_score
from sklearn.model_selection import train_test_split

try:
    import papermill as pm
except Exception:  # pragma: no cover - runtime guard
    pm = None


REPO_ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
MODELS_DIR = REPO_ROOT / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"
DATA_DIR = REPO_ROOT / "data" / "cleaned_data"

NOTEBOOK_EXECUTION_PLAN = [
    ("eda.ipynb", DATA_DIR / "eda_data.csv"),
    ("clustering.ipynb", DATA_DIR / "clustered_data.csv"),
    ("Association_rules.ipynb", MODELS_DIR / "rules.json"),
    ("MachineGuard_Recommender.ipynb", MODELS_DIR / "knowledge_base.json"),
]

CLASS_FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "type_enc",
    "temp_diff",
    "power_W",
    "wear_rate",
    "torque_speed_ratio",
    "high_wear_flag",
    "thermal_overload",
    "health_regime_enc",
    "PC1",
    "PC2",
]

REG_FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "type_enc",
    "temp_diff",
    "power_W",
    "torque_speed_ratio",
    "health_regime_enc",
    "PC1",
    "PC2",
]


def _column_variants(col: str) -> List[str]:
    return [col, col.lower(), col.upper(), col.replace("_W", "_w"), col.replace("PC", "pc")]


def _align_to_model_features(input_df: pd.DataFrame, model: Any) -> pd.DataFrame:
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return input_df
    out = pd.DataFrame(index=input_df.index)
    for col in expected:
        source = next((c for c in _column_variants(col) if c in input_df.columns), None)
        out[col] = input_df[source] if source else 0.0
    return out


def _safe_classifier_predict(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Predict with compatibility fallbacks for legacy XGBoost sklearn wrappers."""
    # For legacy pickled XGBoost sklearn wrappers, bypass wrapper internals and
    # use the underlying booster directly.
    if hasattr(model, "get_booster"):
        import xgboost as xgb

        booster = model.get_booster()
        dmat = xgb.DMatrix(features.to_numpy(), feature_names=list(features.columns))
        preds = booster.predict(dmat)
        if preds.ndim == 1:
            return (preds >= 0.5).astype(int)
        return np.argmax(preds, axis=1)

    try:
        return model.predict(features)
    except Exception as exc:
        if "use_label_encoder" in str(exc) and not hasattr(model, "use_label_encoder"):
            setattr(model, "use_label_encoder", False)
            return model.predict(features)
        raise


def _safe_regressor_predict(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Predict regression with compatibility fallbacks for legacy XGBoost wrappers."""
    if hasattr(model, "get_booster"):
        import xgboost as xgb

        booster = model.get_booster()
        dmat = xgb.DMatrix(features.to_numpy(), feature_names=list(features.columns))
        return booster.predict(dmat)

    try:
        return model.predict(features)
    except Exception as exc:
        if "use_label_encoder" in str(exc) and not hasattr(model, "use_label_encoder"):
            setattr(model, "use_label_encoder", False)
            return model.predict(features)
        raise


def _first_existing_model(candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        path = MODELS_DIR / name
        if path.exists():
            return path
    return None


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _failure_type_multiclass(df: pd.DataFrame) -> pd.Series:
    y = np.zeros(len(df), dtype=int)
    y[df["twf"] == 1] = 1
    y[df["hdf"] == 1] = 2
    y[df["pwf"] == 1] = 3
    y[df["osf"] == 1] = 4
    y[df["rnf"] == 1] = 5
    return pd.Series(y, index=df.index)


def _safe_silhouette(df: pd.DataFrame) -> Optional[float]:
    if "cluster_raw" not in df.columns or "PC1" not in df.columns or "PC2" not in df.columns:
        return None
    if df["cluster_raw"].nunique() < 2:
        return None
    sample = df[["PC1", "PC2"]].copy()
    return float(silhouette_score(sample, df["cluster_raw"]))


@task(retries=2, retry_delay_seconds=5)
def run_notebook(notebook_name: str) -> str:
    logger = get_run_logger()
    input_path = NOTEBOOKS_DIR / notebook_name
    output_path = NOTEBOOKS_DIR / f"executed_{notebook_name}"
    if not input_path.exists():
        raise FileNotFoundError(f"Notebook missing: {input_path}")
    if pm is None:
        raise RuntimeError("papermill is required to execute notebooks.")
    logger.info("Executing notebook: %s", notebook_name)
    pm.execute_notebook(str(input_path), str(output_path))
    logger.info("Notebook completed: %s", notebook_name)
    return str(output_path)


@task(retries=2, retry_delay_seconds=2)
def validate_expected_outputs() -> Dict[str, bool]:
    classifier_path = _first_existing_model(
        ["xgb_classifier.pkl", "xgb_isFailure.pkl", "svm_classifier.pkl", "svm_isFailure.pkl"]
    )
    regressor_path = _first_existing_model(["xgb_regressor.pkl", "tool_wear_regressor.pkl"])
    checks = {
        "eda_data_exists": (DATA_DIR / "eda_data.csv").exists(),
        "clustered_data_exists": (DATA_DIR / "clustered_data.csv").exists(),
        "pca_exists": (MODELS_DIR / "pca.pkl").exists(),
        "kmeans_exists": (MODELS_DIR / "kmeans.pkl").exists(),
        "scaler_exists": (MODELS_DIR / "scaler.pkl").exists(),
        "classifier_exists": classifier_path is not None,
        "regressor_exists": regressor_path is not None,
        "rules_exists": (MODELS_DIR / "rules.json").exists(),
        "knowledge_base_exists": (MODELS_DIR / "knowledge_base.json").exists(),
    }
    missing = [k for k, ok in checks.items() if not ok]
    if missing:
        raise RuntimeError(f"Missing expected artifacts: {missing}")
    return checks


@task(retries=1, retry_delay_seconds=2)
def evaluate_multi_task_metrics() -> Dict[str, Any]:
    try:
        logger = get_run_logger()
    except Exception:
        logger = logging.getLogger("machineguard-pipeline")
    clustered_path = DATA_DIR / "clustered_data.csv"
    if not clustered_path.exists():
        raise FileNotFoundError(f"Missing clustered dataset: {clustered_path}")
    df = pd.read_csv(clustered_path)
    metrics: Dict[str, Any] = {"tasks": {}}

    # Classification (accuracy, F1)
    df_class = df.dropna(subset=CLASS_FEATURES + ["machine_failure"]).copy()
    X_class = df_class[CLASS_FEATURES]
    y_class = df_class["machine_failure"].astype(int)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_class,
        y_class,
        test_size=0.2,
        random_state=42,
        stratify=y_class,
    )
    baseline_clf = DummyClassifier(strategy="most_frequent").fit(X_train_c, y_train_c)
    y_pred_base = baseline_clf.predict(X_test_c)
    baseline_class = {
        "accuracy": float(accuracy_score(y_test_c, y_pred_base)),
        "f1": float(f1_score(y_test_c, y_pred_base, average="macro", zero_division=0)),
    }
    improved_class = dict(baseline_class)
    clf_path = _first_existing_model(
        ["xgb_classifier.pkl", "xgb_isFailure.pkl", "svm_classifier.pkl", "svm_isFailure.pkl"]
    )
    if clf_path is not None:
        clf = joblib.load(clf_path)
        X_aligned = _align_to_model_features(X_test_c, clf)
        y_pred = _safe_classifier_predict(clf, X_aligned)
        improved_class = {
            "accuracy": float(accuracy_score(y_test_c, y_pred)),
            "f1": float(f1_score(y_test_c, y_pred, average="macro", zero_division=0)),
        }
    metrics["tasks"]["classification"] = {
        "baseline": baseline_class,
        "improved": improved_class,
    }

    # Regression (RMSE)
    df_reg = df.dropna(subset=REG_FEATURES + ["tool_wear"]).copy()
    X_reg = df_reg[REG_FEATURES]
    y_reg = df_reg["tool_wear"].astype(float)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    baseline_reg = DummyRegressor(strategy="mean").fit(X_train_r, y_train_r)
    y_pred_base_r = baseline_reg.predict(X_test_r)
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test_r, y_pred_base_r)))
    improved_rmse = baseline_rmse
    reg_path = _first_existing_model(["xgb_regressor.pkl", "tool_wear_regressor.pkl"])
    if reg_path is not None:
        reg_model = joblib.load(reg_path)
        X_reg_aligned = _align_to_model_features(X_test_r, reg_model)
        y_pred_r = _safe_regressor_predict(reg_model, X_reg_aligned)
        improved_rmse = float(np.sqrt(mean_squared_error(y_test_r, y_pred_r)))
    metrics["tasks"]["regression"] = {
        "baseline": {"rmse": baseline_rmse},
        "improved": {"rmse": improved_rmse},
    }

    # Clustering (silhouette)
    sil = _safe_silhouette(df)
    metrics["tasks"]["clustering"] = {"silhouette_score": sil}

    # Time series (forecast evaluation proxy on rolling signal)
    if {"tool_wear_roll_mean_5", "tool_wear_lag1"}.issubset(df.columns):
        ts_df = df.dropna(subset=["tool_wear_roll_mean_5", "tool_wear_lag1"]).copy()
        rmse_ts = float(
            np.sqrt(mean_squared_error(ts_df["tool_wear_roll_mean_5"], ts_df["tool_wear_lag1"]))
        )
        metrics["tasks"]["time_series"] = {"forecast_rmse_proxy": rmse_ts}
    else:
        metrics["tasks"]["time_series"] = {"forecast_rmse_proxy": None}

    # Association (rule metrics)
    rules_path = MODELS_DIR / "rules.json"
    if rules_path.exists():
        rules = _load_json(rules_path)
        all_rules: List[Dict[str, Any]] = []
        if isinstance(rules, dict):
            for regime_rules in rules.values():
                if isinstance(regime_rules, list):
                    all_rules.extend(regime_rules)
        elif isinstance(rules, list):
            all_rules = rules
        confs = [float(r.get("confidence", 0.0)) for r in all_rules]
        lifts = [float(r.get("lift", 0.0)) for r in all_rules]
        metrics["tasks"]["association"] = {
            "rule_count": len(all_rules),
            "avg_confidence": float(np.mean(confs)) if confs else 0.0,
            "avg_lift": float(np.mean(lifts)) if lifts else 0.0,
        }
    else:
        metrics["tasks"]["association"] = {"rule_count": 0, "avg_confidence": 0.0, "avg_lift": 0.0}

    # Recommendation coverage
    kb_path = MODELS_DIR / "knowledge_base.json"
    if kb_path.exists():
        kb = _load_json(kb_path)
        if isinstance(kb, list) and kb:
            pairs = {(x.get("failure_type"), x.get("age_bin")) for x in kb}
            metrics["tasks"]["recommendation"] = {
                "entries": len(kb),
                "coverage_pairs": len(pairs),
            }
        else:
            metrics["tasks"]["recommendation"] = {"entries": 0, "coverage_pairs": 0}
    else:
        metrics["tasks"]["recommendation"] = {"entries": 0, "coverage_pairs": 0}

    logger.info("Multi-task metrics computed successfully.")
    return metrics


@task
def save_metrics(metrics: Dict[str, Any]) -> str:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / "metrics.json"
    payload = {
        "run_at_utc": pd.Timestamp.utcnow().isoformat(),
        "metrics": metrics,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return str(output_path)


@flow(name="machineguard-multi-ml-pipeline")
def machineguard_pipeline(execute_notebooks: bool = False) -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info("Starting MachineGuard multi-ML pipeline.")

    if execute_notebooks:
        for notebook_name, expected_output in NOTEBOOK_EXECUTION_PLAN:
            run_notebook.submit(notebook_name)
            logger.info("Queued notebook execution: %s", notebook_name)
            if expected_output:
                logger.info("Expected artifact after execution: %s", expected_output)

    validate_expected_outputs_result = validate_expected_outputs()
    metrics = evaluate_multi_task_metrics()
    metrics_path = save_metrics(metrics)

    return {
        "status": "success",
        "validate_outputs": validate_expected_outputs_result,
        "metrics_path": metrics_path,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = machineguard_pipeline(execute_notebooks=False)
    print(json.dumps(result, indent=2))
