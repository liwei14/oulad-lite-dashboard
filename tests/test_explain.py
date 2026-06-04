"""Step 5 SHAP and label contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from sklearn.model_selection import train_test_split

from src.explain import (
    DISCIPLINE_FEATURES,
    _forum_quiz_relation,
    compute_shap,
    discipline_shap_data_caption,
    discipline_shap_triple_values,
    plot_discipline_shap_triple,
    plot_global_importance,
    plot_individual_waterfall,
    whats_behind_interpretation_caption,
    write_global_shap_html,
)
from src.labels import FEATURE_LABELS
from src.model import MODEL_ARTIFACTS, load_artifacts, train_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim"
OUTPUTS = PROJECT_ROOT / "outputs"
SHAP_HTML = OUTPUTS / "shap_global_lr.html"


@pytest.fixture(scope="module")
def artifacts():
    if MODEL_ARTIFACTS.exists():
        return load_artifacts()
    X = pd.read_parquet(INTERIM / "X.parquet")
    y = pd.read_parquet(INTERIM / "y.parquet")["y_at_risk"]
    train_models(X, y, random_state=42)
    return load_artifacts()


def _figure_label_text(fig: go.Figure) -> str:
    parts = [json.dumps(fig.to_dict())]
    return " ".join(parts)


def test_shap_shape(artifacts):
    lr = artifacts["models"]["lr"]
    scaler = artifacts["scaler"]
    X_test = artifacts["splits"]["X_test"]
    shap_values, _ = compute_shap(lr, scaler, X_test)
    assert shap_values.shape == X_test.shape


def test_feature_label_coverage(artifacts):
    X_test = artifacts["splits"]["X_test"]
    missing = set(X_test.columns) - set(FEATURE_LABELS.keys())
    assert not missing, f"Features without pretty labels: {missing}"


def test_individual_waterfall(artifacts):
    lr = artifacts["models"]["lr"]
    scaler = artifacts["scaler"]
    X_test = artifacts["splits"]["X_test"]
    id_student_test = artifacts["id_student_test"]

    shap_values, explainer = compute_shap(lr, scaler, X_test)
    student_id = int(id_student_test.iloc[0])
    fig = plot_individual_waterfall(
        shap_values,
        X_test,
        student_id=student_id,
        id_student_test=id_student_test,
        explainer=explainer,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1

    label_blob = _figure_label_text(fig)
    human_labels = set(FEATURE_LABELS.values())
    found = sum(1 for lbl in human_labels if lbl in label_blob)
    assert found >= 3, f"Expected >=3 pretty labels in figure, found {found}"


def test_shap_global_html(artifacts):
    lr = artifacts["models"]["lr"]
    scaler = artifacts["scaler"]
    X_test = artifacts["splits"]["X_test"]

    write_global_shap_html(lr, scaler, X_test, SHAP_HTML)
    assert SHAP_HTML.exists()

    html = SHAP_HTML.read_text(encoding="utf-8")
    assert "First TMA submitted by week 4" in html
    assert "tma_submitted_by_28" not in html


def _demographics_test_aligned() -> pd.DataFrame:
    """Same stratified test split as train_models (random_state=42)."""
    X = pd.read_parquet(INTERIM / "X.parquet")
    y = pd.read_parquet(INTERIM / "y.parquet")["y_at_risk"]
    demographics = pd.read_parquet(INTERIM / "demographics.parquet")
    idx = np.arange(len(X))
    _, _, _, _, _, idx_test = train_test_split(
        X, y, idx, test_size=0.2, stratify=y, random_state=42
    )
    return demographics.iloc[idx_test].reset_index(drop=True)


def test_discipline_shap_triple(artifacts):
    lr = artifacts["models"]["lr"]
    scaler = artifacts["scaler"]
    X_test = artifacts["splits"]["X_test"]
    demographics_test = _demographics_test_aligned()

    shap_values, _ = compute_shap(lr, scaler, X_test)
    mask_bbb = demographics_test["code_module"] == "BBB"
    mask_ddd = demographics_test["code_module"] == "DDD"
    X_bbb = X_test.loc[mask_bbb].reset_index(drop=True)
    shap_bbb = shap_values[mask_bbb.values]
    X_ddd = X_test.loc[mask_ddd].reset_index(drop=True)
    shap_ddd = shap_values[mask_ddd.values]

    bbb_values = discipline_shap_triple_values(shap_bbb, X_bbb)
    ddd_values = discipline_shap_triple_values(shap_ddd, X_ddd)
    bbb_caption = whats_behind_interpretation_caption("BBB", bbb_values, ddd_values)
    if _forum_quiz_relation(bbb_values) == "quiz > forum":
        assert "exceed forum activity" in bbb_caption
        assert "humanities = forum first" in bbb_caption

    values = bbb_values
    assert set(values.keys()) == set(DISCIPLINE_FEATURES)

    fig = plot_discipline_shap_triple(shap_bbb, X_bbb, "BBB", highlight=True)
    assert isinstance(fig, go.Figure)
    label_blob = _figure_label_text(fig)
    assert "Forum activity" in label_blob
    assert "Quiz / practice clicks" in label_blob
    assert "First TMA score" in label_blob
    assert "clicks_forum" not in label_blob

    caption = discipline_shap_data_caption(values, values)
    assert "BBB" in caption and "DDD" in caption

    bbb_high_quiz = {
        "clicks_forum": 0.01,
        "clicks_quiz": 0.05,
        "first_tma_score": 0.2,
    }
    ddd_high_quiz = {
        "clicks_forum": 0.01,
        "clicks_quiz": 0.06,
        "first_tma_score": 0.25,
    }
    bbb_text = whats_behind_interpretation_caption("BBB", bbb_high_quiz, ddd_high_quiz)
    assert "quiz / practice clicks" in bbb_text
    assert "exceed forum activity" in bbb_text
    assert "humanities = forum first" in bbb_text
    assert "many teachers treat forum participation as an early" not in bbb_text

    ddd_text = whats_behind_interpretation_caption("DDD", bbb_high_quiz, ddd_high_quiz)
    assert "higher quiz/practice click SHAP aligns" in ddd_text


def test_gb_tree_explainer_smoke(artifacts):
    gb = artifacts["models"]["gb"]
    scaler = artifacts["scaler"]
    X_test = artifacts["splits"]["X_test"]
    shap_values, explainer = compute_shap(gb, scaler, X_test)
    assert shap_values.shape == X_test.shape
    fig = plot_global_importance(shap_values, X_test, top_n=5)
    assert isinstance(fig, go.Figure)
