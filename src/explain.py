"""SHAP explanations and Plotly charts for teacher-facing views (Step 5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.data_loader import id_to_pos
from src.labels import pretty

PRIMARY_COLOR = "#0071e3"
SECONDARY_COLOR = "#86868b"

DISCIPLINE_FEATURES = ("clicks_forum", "clicks_quiz", "first_tma_score")


def _shap_matrix(raw: Any, n_features: int) -> np.ndarray:
    """Normalize SHAP outputs to 2D (n_samples, n_features) for binary at-risk."""
    if isinstance(raw, list):
        if len(raw) == 1:
            arr = np.asarray(raw[0])
        else:
            arr = np.asarray(raw[1] if len(raw) > 1 else raw[-1])
    else:
        arr = np.asarray(raw)

    if arr.ndim == 3:
        arr = arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape {arr.shape}")
    if arr.shape[1] != n_features:
        raise ValueError(f"SHAP features {arr.shape[1]} != X columns {n_features}")
    return arr


def _expected_value_scalar(expected_value: Any) -> float:
    """Coerce explainer.expected_value to a scalar base for waterfall."""
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        ev = np.asarray(expected_value).ravel()
        if ev.size == 0:
            return 0.0
        if ev.size == 1:
            return float(ev[0])
        return float(ev[1] if ev.size > 1 else ev[0])
    return float(expected_value)


def _build_explainer(model: Any, X_scaled: np.ndarray) -> shap.Explainer:
    if isinstance(model, LogisticRegression):
        return shap.LinearExplainer(model, X_scaled)
    if isinstance(model, GradientBoostingClassifier):
        return shap.TreeExplainer(model)
    raise TypeError(f"Unsupported model type for SHAP: {type(model)}")


def compute_shap(
    model: Any,
    scaler: Any,
    X: pd.DataFrame,
) -> tuple[np.ndarray, shap.Explainer]:
    """Standardize X, compute SHAP values; return (shap_values, explainer)."""
    X_scaled = scaler.transform(X)
    explainer = _build_explainer(model, X_scaled)
    raw = explainer.shap_values(X_scaled)
    shap_values = _shap_matrix(raw, n_features=X.shape[1])
    return shap_values, explainer


def plot_global_importance(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    top_n: int = 10,
) -> go.Figure:
    """Mean |SHAP| bar chart with human-readable feature labels."""
    importance = np.abs(shap_values).mean(axis=0)
    order = np.argsort(importance)[::-1][:top_n]
    cols = [X.columns[i] for i in order]
    values = importance[order]

    fig = go.Figure(
        go.Bar(
            x=[pretty(c) for c in cols],
            y=values,
            marker_color=PRIMARY_COLOR,
        )
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Feature",
        yaxis_title="Mean |SHAP|",
        title="Global feature importance (test set)",
    )
    return fig


def discipline_shap_triple_values(
    shap_values: np.ndarray,
    X: pd.DataFrame,
) -> dict[str, float]:
    """Mean |SHAP| for forum, quiz, and first TMA score columns."""
    importance = np.abs(shap_values).mean(axis=0)
    col_index = {name: i for i, name in enumerate(X.columns)}
    return {
        col: float(importance[col_index[col]])
        for col in DISCIPLINE_FEATURES
        if col in col_index
    }


def plot_discipline_shap_triple(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    module_label: str,
    *,
    highlight: bool = False,
) -> go.Figure:
    """Three-bar mean |SHAP| chart for cross-module discipline comparison."""
    values = discipline_shap_triple_values(shap_values, X)
    cols = [c for c in DISCIPLINE_FEATURES if c in values]
    y = [values[c] for c in cols]
    color = PRIMARY_COLOR if highlight else SECONDARY_COLOR
    title = f"Module {module_label}"
    if highlight:
        title += " (this module)"

    fig = go.Figure(
        go.Bar(
            x=[pretty(c) for c in cols],
            y=y,
            marker_color=color,
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Signal",
        yaxis_title="Mean |SHAP|",
        height=320,
    )
    return fig


SHAP_READING_NOTE = (
    "How to read the charts: a single early-warning model is trained jointly on BBB "
    "and DDD (day-28 features). Forum vs quiz bars are mean |SHAP| on each module's "
    "test students only — not tested for statistical significance."
)


def _forum_quiz_relation(values: dict[str, float]) -> str:
    forum = values.get("clicks_forum", 0.0)
    quiz = values.get("clicks_quiz", 0.0)
    if quiz > forum + 1e-12:
        return "quiz > forum"
    if forum > quiz + 1e-12:
        return "forum > quiz"
    return "forum = quiz"


def discipline_shap_data_caption(
    bbb_values: dict[str, float],
    ddd_values: dict[str, float],
) -> str:
    """One-line rank comparison for forum vs quiz on each module subset."""
    return (
        "Cross-module (this test split): BBB — "
        f"{_forum_quiz_relation(bbb_values)}; DDD — "
        f"{_forum_quiz_relation(ddd_values)}. "
        "First TMA score is highest among the three bars on both panels."
    )


def _module_data_facts(module_code: str, values: dict[str, float]) -> str:
    tma = values.get("first_tma_score", 0.0)
    forum = values.get("clicks_forum", 0.0)
    quiz = values.get("clicks_quiz", 0.0)
    relation = _forum_quiz_relation(values)
    if relation == "quiz > forum":
        fq = (
            f"quiz / practice clicks ({quiz:.4f}) exceed forum activity ({forum:.4f})"
        )
    elif relation == "forum > quiz":
        fq = (
            f"forum activity ({forum:.4f}) exceeds quiz / practice clicks ({quiz:.4f})"
        )
    else:
        fq = f"forum and quiz are equal ({forum:.4f})"
    return (
        f"{module_code} (this page): first TMA score has the largest mean |SHAP| "
        f"among the three bars ({tma:.4f}); {fq}."
    )


def _module_pedagogical_note(module_code: str, values: dict[str, float]) -> str:
    relation = _forum_quiz_relation(values)
    if module_code == "BBB":
        if relation == "quiz > forum":
            return (
                "Classroom lens: humanities teachers may still monitor forum "
                "discussions, but this model weights quiz/practice clicks more than "
                "forum clicks for BBB here — follow the bars, not a generic "
                "'humanities = forum first' rule (OULAD has click counts only, no post text)."
            )
        if relation == "forum > quiz":
            return (
                "Classroom lens: higher forum click SHAP matches a common humanities "
                "focus on discussion participation at day 28; still verify in class "
                "(clicks are not discussion quality)."
            )
        return (
            "Classroom lens: forum and quiz carry equal mean |SHAP| for BBB here; "
            "use classroom context alongside TMA and the global feature expander below."
        )
    if relation == "quiz > forum":
        return (
            "Classroom lens: higher quiz/practice click SHAP aligns with many STEM "
            "teachers watching early machine-graded practice; bars are VLE click proxies, "
            "not CMA score trajectories."
        )
    if relation == "forum > quiz":
        return (
            "Classroom lens: forum clicks outweigh quiz in this DDD test split — "
            "unusual for a STEM stereotype; prioritise what the bars show over "
            "subject labels."
        )
    return (
        "Classroom lens: forum and quiz carry equal mean |SHAP| for DDD here; "
        "use context and the global feature expander below."
    )


def whats_behind_interpretation_caption(
    module_code: str,
    bbb_values: dict[str, float],
    ddd_values: dict[str, float],
) -> str:
    """Data-aligned interpretation for the active teacher module page."""
    values = bbb_values if module_code == "BBB" else ddd_values
    return " ".join(
        [
            SHAP_READING_NOTE,
            _module_data_facts(module_code, values),
            discipline_shap_data_caption(bbb_values, ddd_values),
            _module_pedagogical_note(module_code, values),
        ]
    )


def plot_individual_waterfall(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    student_id: int,
    id_student_test: pd.Series,
    explainer: shap.Explainer,
) -> go.Figure:
    """Single-student SHAP waterfall; student_id is OULAD id_student."""
    pos = id_to_pos(student_id, id_student_test)
    row_shap = shap_values[pos]
    base = _expected_value_scalar(explainer.expected_value)

    order = np.argsort(np.abs(row_shap))[::-1]
    features = [X.columns[i] for i in order]
    contributions = row_shap[order]

    fig = go.Figure(
        go.Waterfall(
            name="SHAP",
            orientation="v",
            measure=["relative"] * len(features) + ["total"],
            x=[pretty(f) for f in features] + ["Predicted log-odds"],
            y=list(contributions) + [None],
            base=base,
            increasing={"marker": {"color": PRIMARY_COLOR}},
            decreasing={"marker": {"color": "#86868b"}},
            totals={"marker": {"color": "#1d1d1f"}},
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=f"Why this student? (ID {student_id})",
        yaxis_title="SHAP contribution",
    )
    return fig


def write_global_shap_html(
    model: Any,
    scaler: Any,
    X: pd.DataFrame,
    path: Union[str, Path],
) -> None:
    """Compute global SHAP for LR and write inspection HTML."""
    shap_values, _ = compute_shap(model, scaler, X)
    fig = plot_global_importance(shap_values, X)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out))
