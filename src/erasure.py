"""Pedagogical data erasure simulation (Step 6).

Pedagogical data erasure: a student can request that their fine-grained behavioural
data (clickstream features) be excluded from the predictive model, giving them a
fresh-start opportunity. This implements the Masschelein & Simons (2013) scholè
principle in LA system design.

Erasable: clickstream-derived features.
Non-erasable: TMA scores, registration dates, final results — these may be retained
under institutional and legal obligations, subject to platform policy.

In this demo, the student is the final decision-maker and the teacher is notified
and may negotiate. In any real institutional deployment, additional review
(registrar, ethics committee, data-protection officer) would typically apply —
see README §Honest limitations.

Erasure simulation replaces clickstream features with the cohort median on the
supplied X frame (not literal zeros): fine-grained behaviour is no longer used to
distinguish the student, so model scores tend to regress toward the cohort centre.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.data_loader import id_to_pos
from src.labels import pretty

ERASABLE_FEATURES = [
    "clicks_total",
    "clicks_forum",
    "clicks_content",
    "clicks_quiz",
    "clicks_other",
    "active_days",
    "days_since_first_click",
]
# first_tma_score and registration features are NOT erasable in this demo:
# assessment results and registration timestamps may be retained under
# institutional/legal obligations, subject to platform policy.

NON_ERASABLE_FEATURES = [
    "has_early_tma",
    "tma_submitted_by_28",
    "first_tma_score",
    "first_tma_delay",
    "days_registered_before_start",
    "num_of_prev_attempts",
    "studied_credits",
]


def erasable_labels() -> list[str]:
    """Plain-English labels for data cleared on student erasure."""
    return [pretty(feat) for feat in ERASABLE_FEATURES]


def retained_labels() -> list[str]:
    """Plain-English labels for data kept after student erasure."""
    return [pretty(feat) for feat in NON_ERASABLE_FEATURES]


__all__ = [
    "ERASABLE_FEATURES",
    "NON_ERASABLE_FEATURES",
    "erasable_labels",
    "retained_labels",
    "simulate_erasure",
    "predict_before_after",
    "scatter_position",
    "plot_erasure_scatter_pair",
    "cohort_percentile",
    "erasure_narrative_markdown",
]


def _at_risk_proba(model: Any, scaler: Any, X: pd.DataFrame) -> np.ndarray:
    """P(at-risk) for every row after scaling."""
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1]


def simulate_erasure(
    X: pd.DataFrame,
    student_id: int,
    id_student_test: pd.Series,
) -> pd.DataFrame:
    """Return a copy of X with clickstream features set to cohort medians for one row."""
    pos = id_to_pos(student_id, id_student_test)
    out = X.copy(deep=True)
    missing = set(ERASABLE_FEATURES) - set(out.columns)
    if missing:
        raise KeyError(f"ERASABLE_FEATURES missing from X: {missing}")
    for feat in ERASABLE_FEATURES:
        out.iloc[pos, out.columns.get_loc(feat)] = X[feat].median()
    return out


def cohort_percentile(cohort_probs: np.ndarray, pos: int) -> int:
    """Cohort rank percentile (1–100) for one row in cohort_probs."""
    n = len(cohort_probs)
    rank = int((cohort_probs < cohort_probs[pos]).sum() + 1)
    return int(round(100 * rank / n))


def predict_before_after(
    model: Any,
    scaler: Any,
    X: pd.DataFrame,
    y: pd.Series,
    student_id: int,
    id_student_test: pd.Series,
) -> dict[str, Any]:
    """Compare at-risk probability before and after erasing one student's clickstream."""
    pos = id_to_pos(student_id, id_student_test)
    X_erased = simulate_erasure(X, student_id, id_student_test)

    cohort_original = _at_risk_proba(model, scaler, X)
    cohort_after_this_erased = _at_risk_proba(model, scaler, X_erased)

    original_prob = float(cohort_original[pos])
    after_prob = float(cohort_after_this_erased[pos])
    delta = after_prob - original_prob

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y = y.reset_index(drop=True)

    return {
        "student_id": int(student_id),
        "pos": int(pos),
        "original_prob": original_prob,
        "after_prob": after_prob,
        "delta": float(delta),
        "cohort_original": cohort_original,
        "cohort_after_this_erased": cohort_after_this_erased,
        "true_base_rate": float(y.mean()),
    }


def scatter_position(
    cohort_probs: np.ndarray,
    student_id: int,
    id_student_test: pd.Series,
) -> dict[str, float]:
    """Map a student to scatter coordinates (cohort rank percentile, model confidence).

    X is ascending rank / N (lowest predicted at-risk prob → rank 1).
    Ties: students with strictly lower prob count toward rank; equal probs share rank.
    """
    pos = id_to_pos(student_id, id_student_test)
    n = len(cohort_probs)
    rank = int((cohort_probs < cohort_probs[pos]).sum() + 1)
    return {
        "x": rank / n,
        "y": float(cohort_probs[pos]),
    }


def _cohort_scatter_xy(cohort_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """X/Y coordinates for every test-row given cohort predicted probabilities."""
    n = len(cohort_probs)
    xs = np.empty(n)
    ys = cohort_probs.astype(float)
    for i in range(n):
        rank = int((cohort_probs < cohort_probs[i]).sum() + 1)
        xs[i] = rank / n
    return xs, ys


def _scatter_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Your position relative to peers — least to most engaged-by-model",
        yaxis_title="Model confidence about you (qualitative)",
        showlegend=False,
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(tickformat=".0%")
    return fig


def erasure_narrative_markdown(
    percentile_before: int,
    percentile_after: int,
    *,
    cohort_label: str = "test cohort",
) -> str:
    """Markdown explaining percentile movement after erasure (no raw probabilities)."""
    before_line = (
        f"Before erasure, the system placed this student near the **{percentile_before}th "
        f"percentile** of model confidence in the {cohort_label}."
    )
    if abs(percentile_after - 50) < abs(percentile_before - 50):
        after_line = (
            "After erasure, the system places them closer to the **middle**, because it "
            "has less information to differentiate them from peers."
        )
    else:
        after_line = (
            f"After erasure, the system places them near the **{percentile_after}th "
            "percentile** — erasure changes model inputs, not necessarily toward the "
            "centre for every student."
        )
    return f"{before_line} {after_line}"


def plot_erasure_scatter_pair(
    result: dict[str, Any],
    student_id: int,
    id_student_test: pd.Series,
) -> tuple[go.Figure, go.Figure, int, int]:
    """Before/after erasure scatter plots (no prob labels in UI).

    Returns (fig_before, fig_after, percentile_before, percentile_after).
    """
    cohort_orig = np.asarray(result["cohort_original"])
    cohort_after = np.asarray(result["cohort_after_this_erased"])
    pos = id_to_pos(student_id, id_student_test)

    x_before, y_before = _cohort_scatter_xy(cohort_orig)
    x_after, y_after = _cohort_scatter_xy(cohort_after)

    n = len(cohort_orig)
    percentile_before = cohort_percentile(cohort_orig, pos)
    percentile_after = cohort_percentile(cohort_after, pos)

    focal_x_b, focal_y_b = float(x_before[pos]), float(y_before[pos])
    focal_x_a, focal_y_a = float(x_after[pos]), float(y_after[pos])

    mask = np.ones(n, dtype=bool)
    mask[pos] = False

    fig_before = go.Figure()
    fig_before.add_trace(
        go.Scatter(
            x=x_before[mask],
            y=y_before[mask],
            mode="markers",
            marker=dict(size=5, color="#d1d1d6", opacity=0.7),
            hoverinfo="skip",
        )
    )
    fig_before.add_trace(
        go.Scatter(
            x=[focal_x_b],
            y=[focal_y_b],
            mode="markers",
            marker=dict(size=16, color="#ff3b30"),
            hoverinfo="skip",
        )
    )
    _scatter_layout(fig_before, "Before erasure")

    fig_after = go.Figure()
    fig_after.add_trace(
        go.Scatter(
            x=x_after[mask],
            y=y_after[mask],
            mode="markers",
            marker=dict(size=5, color="#d1d1d6", opacity=0.7),
            hoverinfo="skip",
        )
    )
    fig_after.add_trace(
        go.Scatter(
            x=[focal_x_a],
            y=[focal_y_a],
            mode="markers",
            marker=dict(size=16, color="#ff3b30"),
            hoverinfo="skip",
        )
    )
    fig_after.add_annotation(
        x=focal_x_a,
        y=focal_y_a,
        ax=focal_x_b,
        ay=focal_y_b,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#ff3b30",
        arrowwidth=2,
    )
    _scatter_layout(fig_after, "After erasure")

    return fig_before, fig_after, percentile_before, percentile_after
