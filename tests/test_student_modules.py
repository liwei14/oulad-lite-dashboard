"""Student tab: per-module row lookup (id_student + code_module)."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.data_loader import (
    default_demo_student_id,
    id_to_pos_module,
    modules_for_student,
)
from src.features import X_PARQUET, Y_PARQUET, DEMOGRAPHICS_PARQUET
from src.model import load_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim"


@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts()


@pytest.fixture(scope="module")
def demographics_test():
    X = pd.read_parquet(X_PARQUET)
    y = pd.read_parquet(Y_PARQUET)["y_at_risk"]
    demographics = pd.read_parquet(DEMOGRAPHICS_PARQUET)
    idx = np.arange(len(X))
    _, _, _, _, _, idx_test = train_test_split(
        X, y, idx, test_size=0.2, stratify=y, random_state=42
    )
    return demographics.iloc[idx_test].reset_index(drop=True)


def _bbb_ddd_student(
    id_student_test: pd.Series, demographics_test: pd.DataFrame
) -> Optional[int]:
    """Student enrolled in both BBB and DDD within the test split."""
    mods: dict[int, set[str]] = {}
    for pos, sid in enumerate(id_student_test):
        key = int(sid)
        mods.setdefault(key, set()).add(
            str(demographics_test.iloc[pos]["code_module"])
        )
    for sid, enrolled in mods.items():
        if enrolled == {"BBB", "DDD"}:
            return sid
    return None


def test_default_demo_student_is_dual_enrollment(artifacts, demographics_test):
    id_student_test = artifacts["id_student_test"]
    X_test = artifacts["splits"]["X_test"]
    sid = default_demo_student_id(
        demographics_test, id_student_test, X_test, prefer_dual_module=True
    )
    assert modules_for_student(sid, demographics_test, id_student_test) == [
        "BBB",
        "DDD",
    ]


def test_modules_for_student_dual_enrollment(artifacts, demographics_test):
    id_student_test = artifacts["id_student_test"]
    sid = _bbb_ddd_student(id_student_test, demographics_test)
    if sid is None:
        pytest.skip("No BBB+DDD student in test set")
    assert modules_for_student(sid, demographics_test, id_student_test) == [
        "BBB",
        "DDD",
    ]


def test_id_to_pos_module_dual_rows_align(artifacts, demographics_test):
    id_student_test = artifacts["id_student_test"]
    sid = _bbb_ddd_student(id_student_test, demographics_test)
    if sid is None:
        pytest.skip("No dual-enrollment student in test set")

    X_test = artifacts["splits"]["X_test"]

    pos_bbb = id_to_pos_module(sid, "BBB", demographics_test, id_student_test)
    pos_ddd = id_to_pos_module(sid, "DDD", demographics_test, id_student_test)
    assert pos_bbb != pos_ddd

    row_bbb = X_test.iloc[pos_bbb]
    row_ddd = X_test.iloc[pos_ddd]
    assert row_bbb["clicks_total"] != row_ddd["clicks_total"] or (
        row_bbb["active_days"] != row_ddd["active_days"]
    )
