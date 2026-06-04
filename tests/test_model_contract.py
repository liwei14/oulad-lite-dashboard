"""Step 4 model training contract tests."""

from pathlib import Path

import pandas as pd
import pytest

from src.fairness import audit
from src.features import FORBIDDEN
from src.model import MODEL_ARTIFACTS, load_artifacts, train_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim"


@pytest.fixture(scope="module")
def trained():
    X = pd.read_parquet(INTERIM / "X.parquet")
    y = pd.read_parquet(INTERIM / "y.parquet")["y_at_risk"]
    return train_models(X, y, random_state=42)


def test_lr_auc_floor(trained):
    assert trained["metrics"]["lr"]["auc"] > 0.65


def test_forbidden_not_in_splits(trained):
    splits = trained["splits"]
    for name in ("X_train", "X_test"):
        leaked = set(splits[name].columns) & FORBIDDEN
        assert not leaked, f"Demographics in {name}: {leaked}"


def test_model_artifacts_persisted():
    assert MODEL_ARTIFACTS.exists()


def test_id_student_test_in_artifacts(trained):
    artifacts = load_artifacts()
    assert "id_student_test" in artifacts
    id_test = artifacts["id_student_test"]
    X_test = artifacts["splits"]["X_test"]
    assert len(id_test) == len(X_test)


def test_fairness_imd_band_smoke(trained):
    splits = trained["splits"]
    demographics = pd.read_parquet(INTERIM / "demographics.parquet")
    demographics_test = demographics.iloc[splits["idx_test"]].reset_index(drop=True)

    scaler = trained["scaler"]
    lr = trained["models"]["lr"]
    X_test_scaled = scaler.transform(splits["X_test"])
    proba = pd.Series(lr.predict_proba(X_test_scaled)[:, 1])

    tables = audit(proba, splits["y_test"], demographics_test)
    imd = tables["imd_band"]
    assert len(imd) >= 2
    assert "flag_rate_ci_lower" in imd.columns
    assert "warn_small_n" in imd.columns
    bands = set(imd["imd_band"].astype(str))
    assert "10-20" not in bands
    if any(b.startswith("10") for b in bands):
        assert "10-20%" in bands
