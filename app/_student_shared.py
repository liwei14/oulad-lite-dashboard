"""Shared student page UI — per-module descriptive rhythm (Step 8 extension)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from _teacher_shared import module_subset, plot_top_click_buckets  # noqa: E402

from src.data_loader import id_to_pos_module  # noqa: E402
from src.labels import pretty  # noqa: E402

PRIMARY_COLOR = "#0071e3"

MODULE_TITLES = {
    "BBB": "Module BBB — Humanities",
    "DDD": "Module DDD — STEM",
}

ERASURE_MODULE_NOTE = (
    "A data-erasure request applies to your fine-grained click behaviour for the "
    "semester (see below) — not to your assessment scores or registration record."
)


def render_module_rhythm(
    student_id: int,
    module_code: str,
    ctx: dict[str, Any],
) -> None:
    """Descriptive module block: rhythm charts, TMA metrics, class click pattern."""
    demographics_test = ctx["demographics_test"]
    id_student_test = ctx["id_student_test"]
    subset = module_subset(ctx, module_code)
    X_test_m = subset["X_test_m"]
    id_test_m = subset["id_test_m"]

    pos = id_to_pos_module(
        student_id, module_code, demographics_test, id_student_test
    )
    row = ctx["X_test"].iloc[pos]

    col_a, col_b = st.columns(2)
    with col_a:
        student_days = float(row["active_days"])
        cohort_mean_days = float(X_test_m["active_days"].mean())
        fig_days = go.Figure(
            go.Bar(
                x=["You", "Cohort average"],
                y=[student_days, cohort_mean_days],
                marker_color=[PRIMARY_COLOR, "#d1d1d6"],
            )
        )
        fig_days.update_layout(
            template="plotly_white",
            title=pretty("active_days"),
            yaxis_title="Days",
            height=320,
        )
        st.plotly_chart(
            fig_days,
            use_container_width=True,
            key=f"stu_{student_id}_{module_code}_active_days",
        )

    with col_b:
        click_cols = ["clicks_forum", "clicks_content", "clicks_quiz"]
        fig_clicks = go.Figure(
            go.Bar(
                x=[pretty(c) for c in click_cols],
                y=[float(row[c]) for c in click_cols],
                marker_color=PRIMARY_COLOR,
            )
        )
        fig_clicks.update_layout(
            template="plotly_white",
            title="Your click breakdown (weeks 1–4)",
            yaxis_title="Clicks",
            height=320,
        )
        st.plotly_chart(
            fig_clicks,
            use_container_width=True,
            key=f"stu_{student_id}_{module_code}_clicks",
        )

    m1, m2, m3 = st.columns(3)
    with m1:
        submitted = int(row["tma_submitted_by_28"])
        st.metric(
            pretty("tma_submitted_by_28"),
            "Yes" if submitted else "No",
        )
    with m2:
        st.metric(
            pretty("first_tma_score"),
            f"{float(row['first_tma_score']):.0f}",
        )
    with m3:
        st.metric(
            pretty("has_early_tma"),
            "Yes" if int(row["has_early_tma"]) else "No",
        )

    st.plotly_chart(
        plot_top_click_buckets(
            ctx["clickstream"],
            ctx["vle"],
            module_code,
            id_test_m,
        ),
        use_container_width=True,
        key=f"stu_{student_id}_{module_code}_top_buckets",
    )
    st.caption(
        f"Compared with students in **Module {module_code}** in this demo — "
        "not a ranking. Popular resources show what classmates engaged with "
        "in weeks 1–4."
    )


def render_student_courses(student_id: int, ctx: dict[str, Any]) -> None:
    """Expander per enrolled module (BBB / DDD)."""
    from src.data_loader import modules_for_student

    modules = modules_for_student(
        student_id, ctx["demographics_test"], ctx["id_student_test"]
    )
    if not modules:
        st.info("No module enrolments found for this demo profile.")
        return

    st.caption(ERASURE_MODULE_NOTE)

    for module_code in modules:
        title = MODULE_TITLES[module_code]
        with st.expander(title, expanded=False):
            render_module_rhythm(student_id, module_code, ctx)
