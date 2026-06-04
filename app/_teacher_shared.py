"""Shared teacher page UI for BBB and DDD (Step 7)."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from _bootstrap import load_teacher_context  # noqa: E402
from _layout import render_page_footer  # noqa: E402

from src.data_loader import id_to_pos  # noqa: E402
from src.erasure import (  # noqa: E402
    erasure_narrative_markdown,
    plot_erasure_scatter_pair,
    predict_before_after,
)
from src.explain import (  # noqa: E402
    discipline_shap_triple_values,
    plot_discipline_shap_triple,
    plot_global_importance,
    plot_individual_waterfall,
    whats_behind_interpretation_caption,
)
from src.features import _activity_bucket  # noqa: E402
PRIMARY_COLOR = "#0071e3"
MOCK_TEACHERS = ["Dr. Pieters", "Dr. Vermeulen", "Dr. Claes"]
MOCK_DIRECTORS = ["Programme director", "Student support lead"]

WHATS_BEHIND_TITLE = "What's behind — analytical / model-driven"
WHATS_BEHIND_SECTION_HELP = (
    "Prototype attempt to bring subject-domain factors into learning-analytics "
    "dashboard design: BBB (humanities) and DDD (STEM) share one day-28 model but "
    "separate teacher views, risk thresholds, and a BBB|DDD discipline-relevant SHAP "
    "comparison. OULAD's DDD is anonymised STEM — not necessarily mathematics. "
    "Anchored in topic-specific PCK (Grigaliūnienė, Lehtinen, Verschaffel & Depaepe "
    "2025, ZDM-Mathematics Education)."
)

def collapsible_group(title: str):
    """Default-collapsed expander; no cross-session persistence."""
    return st.expander(title, expanded=False)


@contextmanager
def whats_behind_section(module_code: str) -> Iterator[None]:
    """Expander for model-driven group; help icon stays top-right of the title row."""
    # top alignment: when the expander opens, the right column must not vertically
    # center the icon against the full expanded height (which pushes it downward).
    title_col, help_col = st.columns([0.94, 0.06], vertical_alignment="top")
    with help_col:
        st.button(
            "ℹ️",
            key=f"whats_behind_help_{module_code}",
            help=WHATS_BEHIND_SECTION_HELP,
            type="tertiary",
        )
    with title_col:
        with st.expander(WHATS_BEHIND_TITLE, expanded=False):
            yield


def module_subset(ctx: dict[str, Any], module_code: str) -> dict[str, Any]:
    """Mask full test cohort to one OULAD module."""
    demographics_test = ctx["demographics_test"]
    mask = demographics_test["code_module"] == module_code
    X_test_m = ctx["X_test"].loc[mask].reset_index(drop=True)
    y_test_m = ctx["y_test"].loc[mask].reset_index(drop=True)
    demog_m = demographics_test.loc[mask].reset_index(drop=True)
    id_test_m = ctx["id_student_test"].loc[mask].reset_index(drop=True)
    shap_m = ctx["shap_values"][mask.values]

    X_scaled = ctx["scaler"].transform(X_test_m)
    proba_m = ctx["lr"].predict_proba(X_scaled)[:, 1]

    return {
        "X_test_m": X_test_m,
        "y_test_m": y_test_m,
        "demog_m": demog_m,
        "id_test_m": id_test_m,
        "shap_m": shap_m,
        "proba_m": proba_m,
    }


def id_to_pos_m(student_id: int, id_test_m: pd.Series) -> int:
    return id_to_pos(student_id, id_test_m)


def _risk_band_counts(proba_m: np.ndarray) -> dict[str, int]:
    low = int((proba_m < 0.3).sum())
    medium = int(((proba_m >= 0.3) & (proba_m <= 0.6)).sum())
    high = int((proba_m > 0.6).sum())
    return {"Low (<0.3)": low, "Medium (0.3–0.6)": medium, "High (>0.6)": high}


def plot_risk_bands(proba_m: np.ndarray) -> go.Figure:
    counts = _risk_band_counts(proba_m)
    fig = go.Figure(
        go.Bar(
            x=list(counts.keys()),
            y=list(counts.values()),
            marker_color=PRIMARY_COLOR,
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="Students by predicted risk band",
        xaxis_title="Band",
        yaxis_title="Count",
    )
    return fig


def plot_active_days_dist(X_test_m: pd.DataFrame) -> go.Figure:
    days = X_test_m["active_days"].value_counts().sort_index()
    fig = go.Figure(
        go.Bar(
            x=[str(int(d)) for d in days.index],
            y=days.values,
            marker_color=PRIMARY_COLOR,
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="Distribution of active days (weeks 1–4)",
        xaxis_title="Active days",
        yaxis_title="Students",
    )
    return fig


def plot_top_click_buckets(
    clickstream: pd.DataFrame,
    vle: pd.DataFrame,
    module_code: str,
    id_test_m: pd.Series,
) -> go.Figure:
    """Top 3 activity buckets by click volume for module test students (day ≤ 28)."""
    student_ids = set(int(s) for s in id_test_m)
    clk = clickstream[
        (clickstream["code_module"] == module_code)
        & (clickstream["id_student"].isin(student_ids))
        & (clickstream["date"] <= 28)
    ].copy()
    merged = clk.merge(
        vle[["id_site", "code_module", "code_presentation", "activity_type"]],
        on=["id_site", "code_module", "code_presentation"],
        how="left",
    )
    merged["bucket"] = merged["activity_type"].map(
        lambda t: _activity_bucket(t) if pd.notna(t) else "other"
    )
    totals = merged.groupby("bucket")["sum_click"].sum().sort_values(ascending=False)
    top = totals.head(3)
    labels = {
        "forum": "Forum",
        "content": "Course content",
        "quiz": "Quiz / CMA",
        "other": "Other",
    }
    fig = go.Figure(
        go.Bar(
            x=[labels.get(b, b.title()) for b in top.index],
            y=top.values,
            marker_color=PRIMARY_COLOR,
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="Top resource types clicked (weeks 1–4, cumulative to day 28)",
        xaxis_title="Resource type",
        yaxis_title="Total clicks",
    )
    return fig


def _render_group1(
    subset: dict[str, Any],
    ctx: dict[str, Any],
    module_code: str,
    hide_model: bool,
) -> None:
    X_test_m = subset["X_test_m"]
    y_test_m = subset["y_test_m"]
    id_test_m = subset["id_test_m"]
    proba_m = subset["proba_m"]

    n = len(X_test_m)
    pct_at_risk = 100.0 * float(y_test_m.mean()) if n else 0.0
    st.metric("Students in this module (test cohort)", n)
    st.metric("Observed at-risk rate (Fail / Withdrawn)", f"{pct_at_risk:.1f}%")

    if hide_model:
        st.plotly_chart(plot_active_days_dist(X_test_m), use_container_width=True)
    else:
        st.plotly_chart(plot_risk_bands(proba_m), use_container_width=True)

    st.plotly_chart(
        plot_top_click_buckets(ctx["clickstream"], ctx["vle"], module_code, id_test_m),
        use_container_width=True,
    )


def _render_group2(
    subset: dict[str, Any],
    ctx: dict[str, Any],
    module_code: str,
    hide_model: bool,
    risk_threshold: float,
) -> None:
    if hide_model:
        st.info("Model predictions hidden by teacher preference.")
        return

    X_test_m = subset["X_test_m"]
    id_test_m = subset["id_test_m"]
    shap_m = subset["shap_m"]
    proba_m = subset["proba_m"]

    subset_bbb = module_subset(ctx, "BBB")
    subset_ddd = module_subset(ctx, "DDD")
    bbb_values = discipline_shap_triple_values(
        subset_bbb["shap_m"], subset_bbb["X_test_m"]
    )
    ddd_values = discipline_shap_triple_values(
        subset_ddd["shap_m"], subset_ddd["X_test_m"]
    )

    st.markdown("**Discipline-relevant signals (BBB vs DDD, test cohort)**")
    col_bbb, col_ddd = st.columns(2)
    with col_bbb:
        st.plotly_chart(
            plot_discipline_shap_triple(
                subset_bbb["shap_m"],
                subset_bbb["X_test_m"],
                "BBB",
                highlight=module_code == "BBB",
            ),
            use_container_width=True,
        )
    with col_ddd:
        st.plotly_chart(
            plot_discipline_shap_triple(
                subset_ddd["shap_m"],
                subset_ddd["X_test_m"],
                "DDD",
                highlight=module_code == "DDD",
            ),
            use_container_width=True,
        )
    st.caption(
        "Mean |SHAP| on each module's test students (single model trained on BBB + DDD, "
        "day-28 features)."
    )
    st.caption(whats_behind_interpretation_caption(module_code, bbb_values, ddd_values))

    with st.expander("Global feature importance (all features)", expanded=False):
        st.plotly_chart(
            plot_global_importance(shap_m, X_test_m),
            use_container_width=True,
        )

    options = [int(s) for s in id_test_m]
    labels = {sid: f"Student #{sid}" for sid in options}
    selected = st.selectbox(
        "Select a student",
        options=options,
        format_func=lambda sid: labels[sid],
    )
    pos = id_to_pos_m(selected, id_test_m)
    prob = float(proba_m[pos])
    flagged = prob >= risk_threshold

    st.plotly_chart(
        plot_individual_waterfall(
            shap_m,
            X_test_m,
            student_id=selected,
            id_student_test=id_test_m,
            explainer=ctx["explainer"],
        ),
        use_container_width=True,
    )
    st.write(f"Predicted P(at-risk): **{prob:.2f}**")
    st.write(
        f"Flagged at module threshold ({risk_threshold:.2f}): "
        f"**{'Yes' if flagged else 'No'}**"
    )


def _render_group3(
    subset: dict[str, Any],
    hide_model: bool,
    risk_threshold: float,
    discussion_prompts: list[str],
) -> None:
    st.markdown(
        "#### Discussion prompts for teacher judgment — *must be verified against context*"
    )

    X_test_m = subset["X_test_m"]
    id_test_m = subset["id_test_m"]
    proba_m = subset["proba_m"]

    if hide_model:
        low_engagement = X_test_m["active_days"] <= X_test_m["active_days"].quantile(0.2)
        candidates = id_test_m[low_engagement].head(5)
        for sid in candidates:
            pos = id_to_pos_m(int(sid), id_test_m)
            st.markdown(f"**Student #{int(sid)}** — low active days ({int(X_test_m.iloc[pos]['active_days'])})")
            st.caption(
                "Consider a check-in about study rhythm and access barriers — "
                "descriptive signal only, not a model prediction."
            )
            st.caption("_verify with context_")
    else:
        order = np.argsort(proba_m)[::-1][:5]
        for pos in order:
            sid = int(id_test_m.iloc[pos])
            prob = float(proba_m[pos])
            st.markdown(f"**Student #{sid}** — predicted P(at-risk) {prob:.2f}")
            for prompt in discussion_prompts:
                st.markdown(f"- {prompt}")
                st.caption("_verify with context_")

    st.markdown("_Use the case conference form below to convene a meeting._")


def _render_erasure_educator(
    ctx: dict[str, Any], subset: dict[str, Any], module_code: str
) -> None:
    """Model consequence preview for educators; not shown on the student page."""
    st.write(
        "If a student requests erasure of fine-grained clickstream data, this preview "
        "shows how the model's view of that student changes. Students only see which "
        "data categories were cleared — not these charts."
    )
    id_test_m = subset["id_test_m"]
    options = [int(s) for s in id_test_m]
    selected = st.selectbox(
        "Student (module test cohort)",
        options=options,
        format_func=lambda sid: f"Student #{sid}",
        key=f"erasure_educator_student_{module_code}",
    )
    sim_key = f"erasure_educator_result_{module_code}_{selected}"
    if st.button(
        "Simulate erasure for selected student",
        key=f"erasure_sim_{module_code}_{selected}",
    ):
        st.session_state[sim_key] = True

    if st.session_state.get(sim_key):
        X_test_m = subset["X_test_m"]
        y_test_m = subset["y_test_m"]
        id_test_m = subset["id_test_m"]
        lr = ctx["lr"]
        scaler = ctx["scaler"]
        n_cohort = len(id_test_m)

        result = predict_before_after(
            lr, scaler, X_test_m, y_test_m, selected, id_test_m
        )
        fig_before, fig_after, percentile_before, percentile_after = (
            plot_erasure_scatter_pair(result, selected, id_test_m)
        )
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            st.plotly_chart(fig_before, use_container_width=True)
        with viz_col2:
            st.plotly_chart(fig_after, use_container_width=True)

        cohort_label = f"{module_code} module test cohort"
        st.caption(f"Cohort: {cohort_label} (n={n_cohort}).")
        st.markdown(
            erasure_narrative_markdown(
                percentile_before,
                percentile_after,
                cohort_label=cohort_label,
            )
        )
        st.warning(
            "Erasure changes what the model knows about the student — not their actual "
            "learning trajectory, and not the support you will continue to offer. "
            "Returning to baseline in this view is a data effect, not a learning outcome.",
            icon="⚠️",
        )
        toward_middle = abs(percentile_after - 50) < abs(percentile_before - 50)
        caption_tail = (
            "back toward the cohort centre"
            if toward_middle
            else "in ways that do not always move every student toward the centre"
        )
        st.caption(
            "The model has less information about the student after erasure, so it falls "
            f"{caption_tail}. This is the technical face of the pedagogical "
            "principle, not a change in their actual situation."
        )


def _render_case_conference(form_key: str) -> None:
    st.subheader("Case conference")
    st.caption(
        "See Special design B on the Intro page.",
    )

    if "case_conferences" not in st.session_state:
        st.session_state["case_conferences"] = []

    with st.form(key=form_key):
        teachers = st.multiselect("Invite teachers", MOCK_TEACHERS)
        directors = st.multiselect("Invite director(s)", MOCK_DIRECTORS)
        topic = st.text_area("Meeting topic")
        student_informed = st.checkbox(
            "Student will be informed of this meeting and offered to attend.",
        )
        meeting_date = st.date_input(
            "Proposed offline meeting date",
            value=date.today(),
        )
        submitted = st.form_submit_button("Log case conference")

    if submitted:
        if not student_informed:
            st.error(
                "You must confirm the student will be informed before logging "
                "a case conference."
            )
        else:
            st.session_state["case_conferences"].append(
                {
                    "teachers": teachers,
                    "directors": directors,
                    "topic": topic,
                    "meeting_date": str(meeting_date),
                }
            )
            st.success("Case conference logged. Student will be notified.")


def _render_timeline(posts: list[str], share_key: str) -> None:
    st.subheader("Colleague sharing timeline")
    st.caption(
        "Teacher well-being and professional community — anchored in "
        "Wenger 1998 / Hargreaves & Fullan 2012 professional capital.",
    )
    for post in posts:
        st.markdown(f"- {post}")
    st.text_input("Share a moment with colleagues…", key=share_key)


def render_teacher_page(
    *,
    module_code: str,
    page_title: str,
    hide_toggle_key: str,
    risk_threshold: float,
    discussion_prompts: list[str],
    timeline_posts: list[str],
    case_form_key: str,
) -> None:
    """Render a module-specific teacher dashboard."""
    ctx = load_teacher_context()
    subset = module_subset(ctx, module_code)

    st.header(page_title)
    hide_model = st.toggle(
        "Hide model predictions",
        value=False,
        key=hide_toggle_key,
        help="Deliberate non-use: see only descriptive LA, decide without "
        "algorithmic suggestion. (Depaepe, Bewust Digitaal.)",
    )

    with collapsible_group("What is — class overview (descriptive)"):
        _render_group1(subset, ctx, module_code, hide_model)

    with whats_behind_section(module_code):
        _render_group2(
            subset,
            ctx,
            module_code,
            hide_model,
            risk_threshold,
        )

    with collapsible_group("What next — discussion prompts for teacher judgment"):
        _render_group3(subset, hide_model, risk_threshold, discussion_prompts)

    with collapsible_group("Data erasure (educator view)"):
        _render_erasure_educator(ctx, subset, module_code)

    _render_case_conference(case_form_key)
    _render_timeline(timeline_posts, share_key=f"share_moment_{module_code}")

    render_page_footer()
