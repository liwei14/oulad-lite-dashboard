"""Student view — reflection, SRL, contact, data erasure (Step 8)."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from _bootstrap import load_teacher_context  # noqa: E402
from _layout import render_page_footer  # noqa: E402
from _student_shared import render_student_courses  # noqa: E402

from src.data_loader import (  # noqa: E402
    default_demo_student_id,
    rhythm_row_pos_for_student,
    student_picker_label,
)
from src.erasure import erasable_labels, retained_labels  # noqa: E402
from src.labels import pretty  # noqa: E402

PRIMARY_COLOR = "#0071e3"
CHECK_IN_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.title("Your learning view")
st.write("Your space to reflect, plan, and connect — no rankings, no risk scores.")

st.info(
    "Your teachers can see a model-assisted view of your engagement, including "
    "suggested discussion prompts. You see the inputs (your own behavioural data) "
    "and you can request erasure at any time. This asymmetry is deliberate: "
    "research (Hattie & Timperley 2007) shows that telling students they are "
    "'predicted to fail' harms learning.",
    icon="🛡️",
)

ctx = load_teacher_context()
X_test = ctx["X_test"]
id_student_test = ctx["id_student_test"]
demographics_test = ctx["demographics_test"]

unique_ids = sorted({int(x) for x in id_student_test})
default_id = default_demo_student_id(
    demographics_test, id_student_test, X_test, prefer_dual_module=True
)
student_id = st.selectbox(
    "Demo: choose a student profile",
    options=unique_ids,
    index=unique_ids.index(default_id),
    format_func=lambda s: student_picker_label(
        s, demographics_test, id_student_test
    ),
    help=(
        "A real platform would authenticate you; this picker is for demonstration "
        "only. Most profiles enrol in one module; the default is the test-set "
        "profile enrolled in both BBB and DDD."
    ),
)
pos = rhythm_row_pos_for_student(
    student_id, demographics_test, id_student_test, X_test
)
row = X_test.iloc[pos]

st.divider()
st.subheader("Your rhythm this term")

col_a, col_b = st.columns(2)
with col_a:
    student_days = float(row["active_days"])
    cohort_mean_days = float(X_test["active_days"].mean())
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
    st.plotly_chart(fig_days, use_container_width=True)

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
    st.plotly_chart(fig_clicks, use_container_width=True)

st.caption(
    "Compared with all students in this demo test cohort (BBB and DDD combined) — "
    "not a ranking."
)

st.divider()
st.subheader("Your courses — weeks 1–4")
render_student_courses(student_id, ctx)

st.divider()
st.subheader("Set a goal for next week (SRL entry)")

with st.form("srl_goal_form"):
    goal_text = st.text_area("What's one specific thing you'll try this week?")
    check_in = st.selectbox("When will you check in on this goal?", CHECK_IN_DAYS)
    submitted = st.form_submit_button("Save goal")
    if submitted:
        if goal_text.strip():
            st.session_state.setdefault("srl_goals", []).append(
                {"goal": goal_text.strip(), "check_in": check_in}
            )
            st.toast("Goal saved — planning is the first step.")
        else:
            st.warning("Please write a short goal before saving.")

if st.session_state.get("srl_goals"):
    st.markdown("**Your saved goals (this session)**")
    for item in st.session_state["srl_goals"][-3:]:
        st.write(f"- {item['goal']} _(check-in: {item['check_in']})_")

st.caption(
    "Self-regulated learning (Zimmerman 2002) — the act of planning matters more "
    "than the plan itself."
)

st.divider()
st.subheader("Talk to a teacher")

btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    if st.button("💬 Send a message to my teacher", use_container_width=True):
        st.session_state["show_message_form"] = True
with btn_col2:
    if st.button("☕ Request an in-person meeting", use_container_width=True):
        st.session_state["show_meeting_form"] = True

if st.session_state.get("show_message_form"):
    with st.form("mock_message"):
        st.text_input("Subject", placeholder="Question about this week's reading")
        st.text_area("Message")
        if st.form_submit_button("Send (demo only)"):
            st.success("Message queued for your teacher (mock — not sent).")
            st.session_state["show_message_form"] = False

if st.session_state.get("show_meeting_form"):
    with st.form("mock_meeting"):
        st.selectbox("Preferred day", CHECK_IN_DAYS)
        st.text_area("What would you like to discuss?")
        if st.form_submit_button("Request (demo only)"):
            st.success("Meeting request logged (mock — not sent).")
            st.session_state["show_meeting_form"] = False

st.write(
    "Online conversation is a great start. For anything important — a struggle, "
    "a confusing topic, a personal worry — we encourage meeting your teacher in person."
)

st.divider()
st.subheader(
    "Start fresh this semester",
    help=(
        "Masschelein & Simons (2013) scholè — school releases you from being defined by "
        "predefined past records. Simons & Masschelein (2021): you should always be able "
        "to begin anew, free from an algorithmic persona. Aligned with GDPR-style "
        "data-subject rights."
    ),
)
st.write(
    "Each semester, you may request that we erase the fine-grained clicking data we "
    "collected about you. This gives you a fresh start: the system will no longer use "
    "this data to estimate where you are. Your assessment results and registration "
    "record are kept (we are required to). Your teacher will be informed but the "
    "decision is yours."
)

erasure_key = f"erasure_result_{student_id}"


@st.dialog("Confirm data erasure")
def _erasure_dialog() -> None:
    st.write(
        "Your teacher (T. Janssens) will be notified. They may want to talk with you "
        "about this. You can proceed without their agreement."
    )
    note = st.text_area("Optional: a note for your teacher about why.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with c2:
        if st.button("Confirm erasure", type="primary", use_container_width=True):
            st.session_state[erasure_key] = {
                "confirmed": True,
                "teacher_note": note.strip() if note else "",
            }
            st.rerun()


if st.button("Request data erasure for this semester"):
    _erasure_dialog()

if erasure_key in st.session_state:
    record = st.session_state[erasure_key]
    st.success(
        "Your erasure request is recorded. Your teacher (T. Janssens) will be notified."
    )
    if record.get("teacher_note"):
        st.caption(f'Your note to your teacher: "{record["teacher_note"]}"')

    st.subheader("What we cleared from your learning profile")
    cleared_md = "\n".join(f"- {label}" for label in erasable_labels())
    st.markdown(cleared_md)

    st.subheader("What we still keep")
    kept_md = "\n".join(f"- {label}" for label in retained_labels())
    st.markdown(
        kept_md
        + "\n\nAssessment results and your registration record are kept because the "
        "institution is required to retain them."
    )

    st.write(
        "You can **start this semester with confidence**: the system will no longer use "
        "the cleared click data to build a profile of your early engagement. Your teacher "
        "knows about your request and will continue to support you."
    )
    st.warning(
        "Erasure changes what the system remembers about your clicks — not your actual "
        "learning, and not the support your teacher will continue to offer.",
        icon="⚠️",
    )

render_page_footer()
