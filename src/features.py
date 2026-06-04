"""Day-28 feature engineering for the at-risk model."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.data_loader import (
    CLICKSTREAM_PARQUET,
    COHORT_PARQUET,
    INTERIM_DIR,
    load_static_tables,
)

FORBIDDEN = {
    "gender",
    "age_band",
    "disability",
    "region",
    "imd_band",
    "highest_education",
    "code_module",
    "code_presentation",
    "id_student",
    "final_result",
}

JOIN_KEYS = ["code_module", "code_presentation", "id_student"]

DEMO_COLS = [
    "gender",
    "age_band",
    "disability",
    "region",
    "imd_band",
    "highest_education",
    "code_module",
]

FEATURE_COLS = [
    "clicks_total",
    "clicks_forum",
    "clicks_content",
    "clicks_quiz",
    "clicks_other",
    "active_days",
    "days_since_first_click",
    "has_early_tma",
    "tma_submitted_by_28",
    "first_tma_score",
    "first_tma_delay",
    "days_registered_before_start",
    "num_of_prev_attempts",
    "studied_credits",
]

FORUM_TYPES = {"forumng", "ouelluminate", "oucollaborate"}
CONTENT_TYPES = {
    "oucontent",
    "resource",
    "subpage",
    "page",
    "homepage",
    "htmlactivity",
}
QUIZ_TYPES = {"quiz"}

X_PARQUET = INTERIM_DIR / "X.parquet"
Y_PARQUET = INTERIM_DIR / "y.parquet"
DEMOGRAPHICS_PARQUET = INTERIM_DIR / "demographics.parquet"


def _activity_bucket(activity_type: str) -> str:
    if activity_type in FORUM_TYPES:
        return "forum"
    if activity_type in CONTENT_TYPES:
        return "content"
    if activity_type in QUIZ_TYPES:
        return "quiz"
    return "other"


def build_clickstream_features(
    clickstream: pd.DataFrame, vle: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate VLE clicks into four activity buckets by day 28."""
    assert clickstream["date"].max() <= 28, (
        f"Clickstream leakage: max date {clickstream['date'].max()}"
    )

    merged = clickstream.merge(
        vle[["id_site", "code_module", "code_presentation", "activity_type"]],
        on=["id_site", "code_module", "code_presentation"],
        how="left",
    )
    merged["bucket"] = merged["activity_type"].map(_activity_bucket)

    grouped = merged.groupby(JOIN_KEYS, as_index=False)
    agg = grouped.agg(
        clicks_total=("sum_click", "sum"),
        active_days=("date", "nunique"),
        first_date=("date", "min"),
    )

    for bucket in ("forum", "content", "quiz", "other"):
        bucket_clicks = (
            merged[merged["bucket"] == bucket]
            .groupby(JOIN_KEYS)["sum_click"]
            .sum()
            .rename(f"clicks_{bucket}")
        )
        agg = agg.merge(
            bucket_clicks.reset_index(),
            on=JOIN_KEYS,
            how="left",
        )

    for col in ("clicks_forum", "clicks_content", "clicks_quiz", "clicks_other"):
        if col not in agg.columns:
            agg[col] = 0
        agg[col] = agg[col].fillna(0).astype(np.int64)

    agg["days_since_first_click"] = (28 - agg["first_date"]).clip(0, 28).astype(
        np.int64
    )
    agg.drop(columns=["first_date"], inplace=True)
    agg["clicks_total"] = agg["clicks_total"].astype(np.int64)
    agg["active_days"] = agg["active_days"].astype(np.int64)

    return agg


def _first_early_tma_by_presentation(assessments: pd.DataFrame) -> pd.DataFrame:
    """Per presentation: earliest TMA with date <= 28, if any."""
    assess = assessments.copy()
    assess["date"] = pd.to_numeric(assess["date"], errors="coerce")
    early_tmas = assess[
        (assess["assessment_type"] == "TMA")
        & assess["date"].notna()
        & (assess["date"] <= 28)
    ].sort_values("date")

    if len(early_tmas) == 0:
        return pd.DataFrame(
            columns=[
                "code_module",
                "code_presentation",
                "id_assessment",
                "first_tma_date",
            ]
        )

    return (
        early_tmas.groupby(["code_module", "code_presentation"], as_index=False)
        .first()[
            ["code_module", "code_presentation", "id_assessment", "date"]
        ]
        .rename(columns={"date": "first_tma_date"})
    )


def build_assessment_features(
    assessments: pd.DataFrame, student_assessment: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Submission-level TMA features + presentation-level has_early_tma flag."""
    sa = student_assessment.copy()
    sa["date_submitted"] = pd.to_numeric(sa["date_submitted"], errors="coerce")
    sa = sa[sa["date_submitted"].notna() & (sa["date_submitted"] <= 28)]
    if len(sa) > 0:
        assert sa["date_submitted"].max() <= 28

    first_tma = _first_early_tma_by_presentation(assessments)
    presentation_flag = pd.DataFrame(
        columns=["code_module", "code_presentation", "has_early_tma"]
    )
    if len(first_tma) > 0:
        presentation_flag = first_tma[
            ["code_module", "code_presentation"]
        ].copy()
        presentation_flag["has_early_tma"] = 1

    if len(first_tma) == 0:
        submission_feats = pd.DataFrame(
            columns=JOIN_KEYS
            + [
                "tma_submitted_by_28",
                "first_tma_score",
                "first_tma_delay",
            ]
        )
        return submission_feats, presentation_flag

    submissions = sa.merge(first_tma, on="id_assessment", how="inner")
    submissions = submissions.rename(columns={"score": "first_tma_score"})
    submissions["first_tma_delay"] = (
        submissions["date_submitted"] - submissions["first_tma_date"]
    )
    submissions["tma_submitted_by_28"] = 1

    submission_feats = submissions[
        JOIN_KEYS
        + ["tma_submitted_by_28", "first_tma_score", "first_tma_delay"]
    ].drop_duplicates(JOIN_KEYS)

    return submission_feats, presentation_flag


def build_registration_features(
    student_registration: pd.DataFrame,
) -> pd.DataFrame:
    """Days registered before course start (negative registration dates)."""
    reg = student_registration[JOIN_KEYS].copy()
    date_reg = pd.to_numeric(
        student_registration["date_registration"].replace("", np.nan),
        errors="coerce",
    )
    reg["days_registered_before_start"] = np.where(
        date_reg < 0, -date_reg, 0
    ).astype(np.float64)
    return reg


def build_prior_load_features(student_info: pd.DataFrame) -> pd.DataFrame:
    """Prior attempts and study load from studentInfo."""
    return student_info[
        JOIN_KEYS + ["num_of_prev_attempts", "studied_credits"]
    ].copy()


def assemble_feature_matrix(
    cohort: pd.DataFrame,
    clickstream_d28: pd.DataFrame,
    vle: pd.DataFrame,
    assessments: pd.DataFrame,
    student_assessment: pd.DataFrame,
    student_registration: pd.DataFrame,
    student_info: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build X, y, demographics; save interim parquets."""
    click_feats = build_clickstream_features(clickstream_d28, vle)
    submission_feats, presentation_flag = build_assessment_features(
        assessments, student_assessment
    )
    reg_feats = build_registration_features(student_registration)
    prior_feats = build_prior_load_features(student_info)

    base = cohort[JOIN_KEYS + ["y_at_risk"]].copy()

    merged = base.merge(click_feats, on=JOIN_KEYS, how="left")
    if len(presentation_flag) > 0:
        merged = merged.merge(
            presentation_flag,
            on=["code_module", "code_presentation"],
            how="left",
        )
    else:
        merged["has_early_tma"] = 0
    merged = merged.merge(submission_feats, on=JOIN_KEYS, how="left")
    merged = merged.merge(reg_feats, on=JOIN_KEYS, how="left")
    merged = merged.merge(prior_feats, on=JOIN_KEYS, how="left")

    click_cols = [
        "clicks_total",
        "clicks_forum",
        "clicks_content",
        "clicks_quiz",
        "clicks_other",
        "active_days",
        "days_since_first_click",
    ]
    for col in click_cols:
        merged[col] = merged[col].fillna(0)

    merged["has_early_tma"] = merged["has_early_tma"].fillna(0).astype(np.int64)
    merged["tma_submitted_by_28"] = (
        merged["tma_submitted_by_28"].fillna(0).astype(np.int64)
    )
    merged["first_tma_score"] = merged["first_tma_score"].fillna(0.0)
    merged["first_tma_delay"] = merged["first_tma_delay"].fillna(0.0)

    merged.loc[merged["has_early_tma"] == 0, "tma_submitted_by_28"] = 0
    merged.loc[merged["tma_submitted_by_28"] == 0, "first_tma_score"] = 0.0
    merged.loc[merged["tma_submitted_by_28"] == 0, "first_tma_delay"] = 0.0

    merged["days_registered_before_start"] = merged[
        "days_registered_before_start"
    ].fillna(0)

    y = merged["y_at_risk"].copy()
    demographics = cohort[DEMO_COLS].copy()
    X = merged[FEATURE_COLS].copy()

    leaked = set(X.columns) & FORBIDDEN
    assert not leaked, f"Demographics leaked into X: {leaked}"

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    X.to_parquet(X_PARQUET, index=False)
    y.to_frame(name="y_at_risk").to_parquet(Y_PARQUET, index=False)
    demographics.to_parquet(DEMOGRAPHICS_PARQUET, index=False)

    return X, y, demographics


def load_and_assemble() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load interim cohort/clickstream and static tables; build feature matrix."""
    cohort = pd.read_parquet(COHORT_PARQUET)
    clickstream = pd.read_parquet(CLICKSTREAM_PARQUET)
    tables = load_static_tables()

    return assemble_feature_matrix(
        cohort=cohort,
        clickstream_d28=clickstream,
        vle=tables["vle"],
        assessments=tables["assessments"],
        student_assessment=tables["studentAssessment"],
        student_registration=tables["studentRegistration"],
        student_info=tables["studentInfo"],
    )
