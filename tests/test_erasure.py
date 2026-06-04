"""Step 6 erasure simulation contract tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import id_to_pos, pos_to_id
from src.erasure import (
    ERASABLE_FEATURES,
    cohort_percentile,
    predict_before_after,
    scatter_position,
    simulate_erasure,
)
from pathlib import Path

from src.model import MODEL_ARTIFACTS, load_artifacts, train_models

INTERIM = Path(__file__).resolve().parents[1] / "data" / "interim"


@pytest.fixture(scope="module")
def artifacts():
    if MODEL_ARTIFACTS.exists():
        return load_artifacts()
    X = pd.read_parquet(INTERIM / "X.parquet")
    y = pd.read_parquet(INTERIM / "y.parquet")["y_at_risk"]
    train_models(X, y, random_state=42)
    return load_artifacts()


@pytest.fixture(scope="module")
def high_clicks_student(artifacts):
    X_test = artifacts["splits"]["X_test"]
    id_student_test = artifacts["id_student_test"]
    pos = int(X_test["clicks_total"].argmax())
    student_id = int(id_student_test.iloc[pos])
    return {"pos": pos, "student_id": student_id, "X_test": X_test, "id_student_test": id_student_test}


def test_predict_before_after_effect(artifacts, high_clicks_student):
    lr = artifacts["models"]["lr"]
    scaler = artifacts["scaler"]
    X_test = artifacts["splits"]["X_test"]
    y_test = artifacts["splits"]["y_test"]
    id_student_test = artifacts["id_student_test"]
    student_id = high_clicks_student["student_id"]

    result = predict_before_after(
        lr, scaler, X_test, y_test, student_id, id_student_test
    )

    assert result["original_prob"] != result["after_prob"]
    assert result["student_id"] == student_id
    assert result["pos"] == high_clicks_student["pos"]


def test_erasure_delta_visible_on_high_clicks(artifacts, high_clicks_student):
    """用户验收: measurable effect for demo-default student."""
    lr = artifacts["models"]["lr"]
    scaler = artifacts["scaler"]
    X_test = artifacts["splits"]["X_test"]
    y_test = artifacts["splits"]["y_test"]
    id_student_test = artifacts["id_student_test"]

    result = predict_before_after(
        lr,
        scaler,
        X_test,
        y_test,
        high_clicks_student["student_id"],
        id_student_test,
    )
    # Median imputation yields a smaller prob shift than literal zeros but stays visible.
    assert abs(result["delta"]) > 0.04, (
        f"Erasure delta {result['delta']:.4f} too small for max clicks_total student"
    )
    pos = high_clicks_student["pos"]
    cohort_orig = result["cohort_original"]
    cohort_after = result["cohort_after_this_erased"]
    pct_before = cohort_percentile(cohort_orig, pos)
    pct_after = cohort_percentile(cohort_after, pos)
    assert abs(pct_after - 50) < abs(pct_before - 50), (
        f"Demo student should move toward 50th percentile: {pct_before} -> {pct_after}"
    )


def test_simulate_erasure_no_input_mutation(artifacts, high_clicks_student):
    X_test = artifacts["splits"]["X_test"]
    id_student_test = artifacts["id_student_test"]
    student_id = high_clicks_student["student_id"]

    before = X_test.copy(deep=True)
    _ = simulate_erasure(X_test, student_id, id_student_test)
    pd.testing.assert_frame_equal(X_test, before)


def test_non_erasable_unchanged(artifacts, high_clicks_student):
    X_test = artifacts["splits"]["X_test"]
    id_student_test = artifacts["id_student_test"]
    student_id = high_clicks_student["student_id"]
    pos = high_clicks_student["pos"]

    erased = simulate_erasure(X_test, student_id, id_student_test)
    for col in ("first_tma_score", "num_of_prev_attempts"):
        assert erased.iloc[pos][col] == X_test.iloc[pos][col]


def test_erasure_targets_correct_row(artifacts, high_clicks_student):
    X_test = artifacts["splits"]["X_test"]
    id_student_test = artifacts["id_student_test"]
    student_id = high_clicks_student["student_id"]
    pos = high_clicks_student["pos"]

    erased = simulate_erasure(X_test, student_id, id_student_test)
    assert int(id_student_test.iloc[pos]) == student_id
    assert erased.iloc[pos]["clicks_total"] == X_test["clicks_total"].median()

    for neighbor in (pos - 1, pos + 1):
        if 0 <= neighbor < len(X_test):
            for feat in ERASABLE_FEATURES:
                assert erased.iloc[neighbor][feat] == X_test.iloc[neighbor][feat]


def test_erasure_row_matches_id_contract(artifacts, high_clicks_student):
    """Row mutated must be the student_id row, not a neighbor."""
    X_test = artifacts["splits"]["X_test"]
    id_student_test = artifacts["id_student_test"]
    student_id = high_clicks_student["student_id"]
    pos = id_to_pos(student_id, id_student_test)

    erased = simulate_erasure(X_test, student_id, id_student_test)
    assert pos_to_id(pos, id_student_test) == student_id
    assert erased.iloc[pos]["clicks_total"] == X_test["clicks_total"].median()
    assert X_test.iloc[pos]["clicks_total"] > X_test["clicks_total"].median()


def test_scatter_position(artifacts, high_clicks_student):
    lr = artifacts["models"]["lr"]
    scaler = artifacts["scaler"]
    X_test = artifacts["splits"]["X_test"]
    y_test = artifacts["splits"]["y_test"]
    id_student_test = artifacts["id_student_test"]
    student_id = high_clicks_student["student_id"]
    pos = high_clicks_student["pos"]

    result = predict_before_after(
        lr, scaler, X_test, y_test, student_id, id_student_test
    )
    cohort = result["cohort_original"]
    pos_dict = scatter_position(cohort, student_id, id_student_test)

    assert pos_dict["y"] == pytest.approx(float(cohort[pos]))
    assert 0 < pos_dict["x"] <= 1.0
