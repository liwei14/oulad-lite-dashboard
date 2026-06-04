"""Human-readable UI labels for model feature columns (Step 5)."""

from __future__ import annotations

FEATURE_LABELS = {
    "clicks_total": "Total VLE clicks (weeks 1–4)",
    "clicks_forum": "Forum activity (weeks 1–4)",
    "clicks_content": "Course content views (weeks 1–4)",
    "clicks_quiz": "Quiz / practice clicks (weeks 1–4)",
    "clicks_other": "Other resource clicks (weeks 1–4)",
    "active_days": "Days active in first 4 weeks",
    "days_since_first_click": "Pre-course engagement (days)",
    "has_early_tma": "First TMA exists in first 4 weeks (0/1)",
    "tma_submitted_by_28": "First TMA submitted by week 4 (0/1)",
    "first_tma_score": "First TMA score (0–100, 0 if no/late submission)",
    "first_tma_delay": "First TMA submission delay (days; 0 if not submitted)",
    "days_registered_before_start": "Days registered before course start",
    "num_of_prev_attempts": "Previous attempts at this module",
    "studied_credits": "Credit load this term",
}


def pretty(col: str) -> str:
    """Map raw feature column name to plain-English chart label."""
    return FEATURE_LABELS.get(col, col)
