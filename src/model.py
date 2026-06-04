"""Train at-risk models and persist artifacts (Step 4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import COHORT_PARQUET
from src.features import FORBIDDEN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_ARTIFACTS = OUTPUTS_DIR / "model_artifacts.joblib"


def _evaluate_model(model, X_test: np.ndarray, y_test: pd.Series) -> dict[str, Any]:
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred),
    }


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> dict[str, Any]:
    """Stratified split, scale, train LR + GB; persist joblib artifacts."""
    leaked = set(X.columns) & FORBIDDEN
    assert not leaked, f"Demographics in model input: {leaked}"

    cohort = pd.read_parquet(COHORT_PARQUET)
    if len(cohort) != len(X):
        raise ValueError(
            f"Cohort length {len(cohort)} != X length {len(X)}; row alignment broken"
        )
    id_student = cohort["id_student"].reset_index(drop=True)

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y = y.reset_index(drop=True)

    idx = np.arange(len(X))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        idx,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    id_student_test = id_student.iloc[idx_test].reset_index(drop=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
    lr.fit(X_train_scaled, y_train)

    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        random_state=42,
    )
    gb.fit(X_train_scaled, y_train)

    metrics = {
        "lr": _evaluate_model(lr, X_test_scaled, y_test),
        "gb": _evaluate_model(gb, X_test_scaled, y_test),
    }

    result: dict[str, Any] = {
        "models": {"lr": lr, "gb": gb},
        "scaler": scaler,
        "feature_columns": list(X.columns),
        "splits": {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "idx_test": idx_test,
            "id_student_test": id_student_test,
        },
        "metrics": metrics,
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": result["models"],
            "scaler": scaler,
            "feature_columns": result["feature_columns"],
            "id_student_test": id_student_test,
            "splits": {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            },
        },
        MODEL_ARTIFACTS,
    )

    return result


def load_artifacts() -> dict[str, Any]:
    """Load persisted model artifacts from outputs/."""
    if not MODEL_ARTIFACTS.exists():
        raise FileNotFoundError(f"Model artifacts not found: {MODEL_ARTIFACTS}")
    return joblib.load(MODEL_ARTIFACTS)
