"""Cached data load for teacher Streamlit pages (Step 7)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import CLICKSTREAM_PARQUET, load_static_tables  # noqa: E402
from src.explain import compute_shap  # noqa: E402
from src.features import DEMOGRAPHICS_PARQUET, X_PARQUET, Y_PARQUET  # noqa: E402
from src.model import load_artifacts  # noqa: E402


def _build_demographics_test() -> pd.DataFrame:
    """Align demographics to X_test via the same stratified split as train_models."""
    X = pd.read_parquet(X_PARQUET)
    y = pd.read_parquet(Y_PARQUET)["y_at_risk"]
    demographics = pd.read_parquet(DEMOGRAPHICS_PARQUET)
    idx = np.arange(len(X))
    _, _, _, _, _, idx_test = train_test_split(
        X,
        y,
        idx,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    return demographics.iloc[idx_test].reset_index(drop=True)


@st.cache_resource(show_spinner="Loading model and explanations…")
def load_teacher_context() -> dict[str, Any]:
    """Load artifacts, SHAP, aligned demographics, clickstream, and VLE."""
    artifacts = load_artifacts()
    splits = artifacts["splits"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]
    id_student_test = artifacts["id_student_test"]
    lr = artifacts["models"]["lr"]
    scaler = artifacts["scaler"]

    shap_values, explainer = compute_shap(lr, scaler, X_test)

    demographics_test = _build_demographics_test()

    if len(demographics_test) != len(X_test):
        raise ValueError(
            f"demographics_test length {len(demographics_test)} != X_test {len(X_test)}"
        )

    clickstream = pd.read_parquet(CLICKSTREAM_PARQUET)
    vle = load_static_tables()["vle"]

    return {
        "artifacts": artifacts,
        "lr": lr,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "id_student_test": id_student_test,
        "shap_values": shap_values,
        "explainer": explainer,
        "demographics_test": demographics_test,
        "clickstream": clickstream,
        "vle": vle,
    }
