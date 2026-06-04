"""Teacher view — Module BBB (Humanities), Step 7."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from _teacher_shared import render_teacher_page  # noqa: E402

BBB_PROMPTS = [
    "Consider inviting to the next forum discussion.",
    "Consider a 1-1 check-in.",
    "Consider suggesting a peer reading group.",
]

BBB_TIMELINE = [
    "Hectic week prepping for module BBB-2014J assessment. ☕ Anyone for coffee Thursday?",
    "Shared a rubric tweak for the first essay — feedback in the humanities channel.",
    "Great moment today — student finally engaged in the forum thread on Week 2 readings.",
    "Office hours packed; reminded myself to leave space for quiet processors.",
    "Colleague tip: pairing forum prompts with short audio summaries helped participation.",
    "End-of-week note: three students moved from lurker to poster — small wins matter.",
]

render_teacher_page(
    module_code="BBB",
    page_title="Teacher view — Module BBB (Humanities)",
    hide_toggle_key="hide_model_bbb",
    risk_threshold=0.55,
    discussion_prompts=BBB_PROMPTS,
    timeline_posts=BBB_TIMELINE,
    case_form_key="case_conference_bbb",
)
