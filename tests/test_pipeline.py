"""End-to-end pipeline test (Step 8)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import COHORT_PARQUET, build_cohort
from src.erasure import predict_before_after
from src.explain import compute_shap
from src.features import FORBIDDEN, load_and_assemble
from src.model import train_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim"
RAW = PROJECT_ROOT / "data" / "raw"


def _pipeline_prerequisites() -> None:
    if not COHORT_PARQUET.exists():
        pytest.skip("Run Step 2 first: data/interim/cohort.parquet missing")
    if not (INTERIM / "X.parquet").exists():
        pytest.skip("Run Step 2–3 first: feature parquets missing")
    if not (RAW / "studentInfo.csv").exists():
        pytest.skip("data/raw symlinks missing — run Step 1")


@pytest.fixture(scope="module")
def pipeline_bundle():
    _pipeline_prerequisites()
    cohort = build_cohort()
    X, y, demographics = load_and_assemble()
    trained = train_models(X, y, random_state=42)
    return {
        "cohort": cohort,
        "X": X,
        "y": y,
        "demographics": demographics,
        "trained": trained,
    }


def test_end_to_end_pipeline(pipeline_bundle):
    cohort = pipeline_bundle["cohort"]
    X = pipeline_bundle["X"]
    demographics = pipeline_bundle["demographics"]
    trained = pipeline_bundle["trained"]

    assert len(cohort) > 0
    assert set(cohort["code_module"].unique()) >= {"BBB", "DDD"}

    leaked = set(X.columns) & FORBIDDEN
    assert not leaked, f"Demographics in X: {leaked}"
    assert len(demographics) == len(X)

    assert trained["metrics"]["lr"]["auc"] > 0.65

    splits = trained["splits"]
    lr = trained["models"]["lr"]
    scaler = trained["scaler"]
    X_test = splits["X_test"]
    shap_values, _ = compute_shap(lr, scaler, X_test)
    assert shap_values.shape[0] == len(X_test)
    assert shap_values.shape[1] == X_test.shape[1]

    id_student_test = splits["id_student_test"]
    pos = int(X_test["clicks_total"].argmax())
    student_id = int(id_student_test.iloc[pos])

    result = predict_before_after(
        lr, scaler, X_test, splits["y_test"], student_id, id_student_test
    )
    assert result["original_prob"] != result["after_prob"]
