"""Tests for day-28 feature engineering."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import CLICKSTREAM_PARQUET, COHORT_PARQUET
from src.features import FORBIDDEN, FEATURE_COLS, load_and_assemble

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim"


@pytest.fixture(scope="module")
def feature_bundle():
    """Build feature matrix once for all tests."""
    return load_and_assemble()


def test_shapes_aligned(feature_bundle):
    X, y, demographics = feature_bundle
    cohort = pd.read_parquet(COHORT_PARQUET)
    assert X.shape[0] == len(y) == demographics.shape[0] == len(cohort)
    assert X.shape[0] == 12808


def test_no_demographics_in_X(feature_bundle):
    X, _, _ = feature_bundle
    leaked = set(X.columns) & FORBIDDEN
    assert not leaked, f"Demographics leaked into X: {leaked}"
    assert list(X.columns) == FEATURE_COLS


def test_clickstream_max_date():
    clickstream = pd.read_parquet(CLICKSTREAM_PARQUET)
    cohort = pd.read_parquet(COHORT_PARQUET)

    keys = ["code_module", "code_presentation", "id_student"]
    sample = cohort[keys].sample(100, random_state=42)

    merged = clickstream.merge(sample, on=keys, how="inner")
    if len(merged) == 0:
        pytest.skip("No clickstream rows for sampled students")

    assert merged["date"].max() <= 28


def test_interim_artifacts_exist(feature_bundle):
    assert (INTERIM / "X.parquet").exists()
    assert (INTERIM / "y.parquet").exists()
    assert (INTERIM / "demographics.parquet").exists()
    X, y, demographics = feature_bundle
    assert len(X) > 0
    assert len(y) > 0
    assert len(demographics) > 0


def test_imd_band_canonical(feature_bundle):
    _, _, demographics = feature_bundle
    values = set(demographics["imd_band"].dropna().astype(str).str.strip())
    assert "10-20" not in values
    if any(v.startswith("10") for v in values):
        assert "10-20%" in values
