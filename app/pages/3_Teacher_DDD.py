"""Teacher view — Module DDD (STEM), Step 7."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from _teacher_shared import render_teacher_page  # noqa: E402

DDD_PROMPTS = [
    "Consider reviewing error patterns in the last CMA together.",
    "Consider a prerequisite-mastery check-in.",
    "Consider offering targeted problem-set practice.",
]

DDD_TIMELINE = [
    "CMA deadline crunch for DDD-2014J — problem-set clinic Friday 14:00?",
    "Noticed a cluster of low scores on the latest machine-graded quiz — will open revision hour.",
    "Student finally asked a question in the STEM forum after weeks of silence — worth celebrating.",
    "Swapped CMA rubric hints with a colleague; fewer 'mystery zero' emails this term.",
    "Lab support short-staffed; flagged to director — students may need async alternatives.",
    "Quick win: targeted practice sheet for prerequisite gaps reduced repeat failures.",
]

render_teacher_page(
    module_code="DDD",
    page_title="Teacher view — Module DDD (STEM)",
    hide_toggle_key="hide_model_ddd",
    risk_threshold=0.45,
    discussion_prompts=DDD_PROMPTS,
    timeline_posts=DDD_TIMELINE,
    case_form_key="case_conference_ddd",
)
