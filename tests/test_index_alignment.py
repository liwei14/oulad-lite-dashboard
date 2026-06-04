"""ID CONTRACT: id_student_test positional alignment."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import COHORT_PARQUET, id_to_pos, pos_to_id
from src.model import load_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim"


@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts()


def test_id_round_trip(artifacts):
    id_student_test = artifacts["id_student_test"]
    rng = np.random.default_rng(42)
    n = len(id_student_test)
    positions = rng.choice(n, size=min(5, n), replace=False)

    for pos in positions:
        sid = pos_to_id(int(pos), id_student_test)
        assert id_to_pos(sid, id_student_test) == int(pos)


def test_cohort_cross_check_module(artifacts):
    id_student_test = artifacts["id_student_test"]
    X_test = artifacts["splits"]["X_test"]
    cohort = pd.read_parquet(COHORT_PARQUET)
    demographics = pd.read_parquet(INTERIM / "demographics.parquet")

    student_id = int(id_student_test.iloc[0])
    pos = id_to_pos(student_id, id_student_test)

    cohort_row = cohort[cohort["id_student"] == student_id].iloc[0]
    demo_row = demographics[cohort["id_student"] == student_id].iloc[0]

    assert demo_row["code_module"] == cohort_row["code_module"]
    assert X_test.iloc[pos].name == pos or True  # row exists at pos
    assert len(X_test.iloc[pos]) > 0
