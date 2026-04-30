import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cleaned_data" / "clustered_data.csv"
OUTPUTS_PATH = ROOT / "outputs" / "deepchecks_report.json"


def _resolve_classifier_model_path() -> Path | None:
    candidates = [
        "xgb_classifier.pkl",
        "xgb_isFailure.pkl",
        "svm_classifier.pkl",
        "svm_isFailure.pkl",
    ]
    for name in candidates:
        path = ROOT / "models" / name
        if path.exists():
            return path
    return None


class _DeepchecksCompatibleModel:
    """Compatibility adapter for legacy pickled XGBoost sklearn wrappers."""

    def __init__(self, model: Any):
        self._model = model

    def predict(self, X: Any) -> np.ndarray:
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if hasattr(self._model, "get_booster"):
            import xgboost as xgb

            dmat = xgb.DMatrix(X_df.to_numpy(), feature_names=list(X_df.columns))
            preds = self._model.get_booster().predict(dmat)
            if preds.ndim == 1:
                return (preds >= 0.5).astype(int)
            return np.argmax(preds, axis=1)
        return self._model.predict(X_df)

    def predict_proba(self, X: Any) -> np.ndarray:
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if hasattr(self._model, "get_booster"):
            import xgboost as xgb

            dmat = xgb.DMatrix(X_df.to_numpy(), feature_names=list(X_df.columns))
            preds = self._model.get_booster().predict(dmat)
            if preds.ndim == 1:
                return np.column_stack([1.0 - preds, preds])
            return preds
        return self._model.predict_proba(X_df)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


def _is_passed(result: Any) -> bool:
    if hasattr(result, "passed"):
        passed_attr = getattr(result, "passed")
        if callable(passed_attr):
            return bool(passed_attr())
        return bool(passed_attr)
    if hasattr(result, "have_conditions") and hasattr(result, "passed_conditions"):
        if not result.have_conditions():
            return True
        return bool(result.passed_conditions())
    return True


def run_deepchecks() -> Dict[str, Any]:
    # Compatibility shim for DeepChecks versions still referencing np.Inf
    if not hasattr(np, "Inf"):
        np.Inf = np.inf  # type: ignore[attr-defined]

    from deepchecks.tabular import Dataset
    from deepchecks.tabular.checks import ModelInfo, TrainTestPerformance
    from deepchecks.tabular.suites import data_integrity
    import joblib
    from sklearn.model_selection import train_test_split

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")
    model_path = _resolve_classifier_model_path()
    if model_path is None:
        raise FileNotFoundError("Missing classifier model in models/ directory.")

    df = pd.read_csv(DATA_PATH)
    features = [
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
    y = df["machine_failure"].astype(int)
    X = df[features].copy()

    model = joblib.load(model_path)
    expected_cols = getattr(model, "feature_names_in_", None)
    if expected_cols is not None:
        lower_map = {c.lower(): c for c in X.columns}
        aligned = pd.DataFrame()
        for col in expected_cols:
            source = lower_map.get(col.lower())
            aligned[col] = X[source] if source is not None else 0.0
        X = aligned

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_ds = Dataset(X_train, label=y_train, cat_features=[])
    test_ds = Dataset(X_test, label=y_test, cat_features=[])

    integrity_suite = data_integrity()
    integrity_result = integrity_suite.run(train_dataset=train_ds, test_dataset=test_ds)

    performance_check = TrainTestPerformance()
    compat_model = _DeepchecksCompatibleModel(model)
    perf_result = performance_check.run(
        train_dataset=train_ds,
        test_dataset=test_ds,
        model=compat_model,
    )

    model_info_check = ModelInfo()
    info_result = model_info_check.run(model=model)

    summary = {
        "integrity_passed": _is_passed(integrity_result),
        "performance_passed": _is_passed(perf_result),
        "model_info_passed": _is_passed(info_result),
    }
    return summary


if __name__ == "__main__":
    OUTPUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = run_deepchecks()
    with OUTPUTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
