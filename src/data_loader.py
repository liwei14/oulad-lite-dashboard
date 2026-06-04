"""Load OULAD CSVs, filter BBB+DDD cohort, and write interim parquet files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

COHORT_PARQUET = INTERIM_DIR / "cohort_bbb_ddd.parquet"
CLICKSTREAM_PARQUET = INTERIM_DIR / "clickstream_bbb_ddd_d28.parquet"

DtypeSpec = Union[str, np.dtype, type]

EXPECTED_SCHEMA: dict[str, dict[str, Any]] = {
    "courses": {
        "required_cols": [
            "code_module",
            "code_presentation",
            "module_presentation_length",
        ],
        "dtypes": {
            "code_module": "object",
            "code_presentation": "object",
            "module_presentation_length": np.int64,
        },
    },
    "studentInfo": {
        "required_cols": [
            "code_module",
            "code_presentation",
            "id_student",
            "gender",
            "region",
            "highest_education",
            "imd_band",
            "age_band",
            "num_of_prev_attempts",
            "studied_credits",
            "disability",
            "final_result",
        ],
        "dtypes": {
            "code_module": "object",
            "code_presentation": "object",
            "id_student": np.int64,
            "gender": "object",
            "region": "object",
            "highest_education": "object",
            "imd_band": "object",
            "age_band": "object",
            "num_of_prev_attempts": np.int64,
            "studied_credits": np.int64,
            "disability": "object",
            "final_result": "object",
        },
    },
    "studentRegistration": {
        "required_cols": [
            "code_module",
            "code_presentation",
            "id_student",
            "date_registration",
            "date_unregistration",
        ],
        "dtypes": {
            "code_module": "object",
            "code_presentation": "object",
            "id_student": np.int64,
            "date_registration": np.float64,
            "date_unregistration": np.float64,
        },
    },
    "assessments": {
        "required_cols": [
            "code_module",
            "code_presentation",
            "id_assessment",
            "assessment_type",
            "date",
            "weight",
        ],
        "dtypes": {
            "code_module": "object",
            "code_presentation": "object",
            "id_assessment": np.int64,
            "assessment_type": "object",
            "date": np.float64,
            "weight": np.float64,
        },
    },
    "studentAssessment": {
        "required_cols": [
            "id_assessment",
            "id_student",
            "date_submitted",
            "is_banked",
            "score",
        ],
        "dtypes": {
            "id_assessment": np.int64,
            "id_student": np.int64,
            "date_submitted": np.float64,
            "is_banked": np.int64,
            "score": np.float64,
        },
    },
    "vle": {
        "required_cols": [
            "id_site",
            "code_module",
            "code_presentation",
            "activity_type",
            "week_from",
            "week_to",
        ],
        "dtypes": {
            "id_site": np.int64,
            "code_module": "object",
            "code_presentation": "object",
            "activity_type": "object",
            "week_from": np.float64,
            "week_to": np.float64,
        },
    },
    "studentVle": {
        "required_cols": [
            "code_module",
            "code_presentation",
            "id_student",
            "id_site",
            "date",
            "sum_click",
        ],
        "dtypes": {
            "code_module": "object",
            "code_presentation": "object",
            "id_student": np.int64,
            "id_site": np.int64,
            "date": np.int64,
            "sum_click": np.int64,
        },
    },
}

_NUMERIC_DTYPES = (np.int64, np.float64, np.int32, np.float32)
_INTEGER_DTYPES = (np.int64, np.int32)

IMD_BAND_ALIASES = {"10-20": "10-20%"}


def normalize_imd_band(series: pd.Series) -> pd.Series:
    """Map known OULAD imd_band typos to canonical decile labels."""
    out = series.astype("object").copy()
    stripped = out.astype(str).str.strip()
    return stripped.replace(IMD_BAND_ALIASES)


def _dtype_kind(dtype: DtypeSpec) -> str:
    if dtype == "object":
        return "object"
    return np.dtype(dtype).kind


def _coerce_column(series: pd.Series, target_dtype: DtypeSpec) -> pd.Series:
    kind = _dtype_kind(target_dtype)

    if kind == "object":
        return series.astype("object")

    if kind in ("i", "u"):
        numeric = pd.to_numeric(series.replace("", np.nan), errors="coerce")
        if numeric.isna().any():
            raise ValueError(
                f"Column has missing values but target dtype is integer: {numeric.isna().sum()} NaNs"
            )
        return numeric.astype(np.int64)

    if kind == "f":
        return pd.to_numeric(series.replace("", np.nan), errors="coerce").astype(np.float64)

    raise TypeError(f"Unsupported target dtype: {target_dtype}")


def _series_matches_dtype(series: pd.Series, target_dtype: DtypeSpec) -> bool:
    kind = _dtype_kind(target_dtype)
    if kind == "object":
        return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
    if kind in ("i", "u"):
        return pd.api.types.is_integer_dtype(series)
    if kind == "f":
        return pd.api.types.is_float_dtype(series)
    return False


def validate_schema(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Assert required columns exist and dtypes match after coercion."""
    if table_name not in EXPECTED_SCHEMA:
        raise KeyError(f"Unknown table_name: {table_name}")

    spec = EXPECTED_SCHEMA[table_name]
    required = spec["required_cols"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            f"{table_name}: missing required columns: {missing}"
        )

    out = df.copy()
    for col, target_dtype in spec["dtypes"].items():
        out[col] = _coerce_column(out[col], target_dtype)
        if not _series_matches_dtype(out[col], target_dtype):
            raise AssertionError(
                f"{table_name}.{col}: expected {target_dtype}, got {out[col].dtype}"
            )

    if table_name == "studentInfo":
        out["imd_band"] = normalize_imd_band(out["imd_band"])

    return out


def load_static_tables() -> dict[str, pd.DataFrame]:
    """Load six small OULAD tables with pandas (not studentVle)."""
    tables = {
        "courses": "courses.csv",
        "studentInfo": "studentInfo.csv",
        "studentRegistration": "studentRegistration.csv",
        "assessments": "assessments.csv",
        "studentAssessment": "studentAssessment.csv",
        "vle": "vle.csv",
    }
    result: dict[str, pd.DataFrame] = {}
    for name, filename in tables.items():
        path = RAW_DIR / filename
        df = pd.read_csv(path, low_memory=False)
        result[name] = validate_schema(df, name)
    return result


def load_clickstream_filtered(
    modules: Sequence[str],
    max_day: Optional[int] = None,
) -> pd.DataFrame:
    """Query studentVle via DuckDB; never load the full CSV into pandas."""
    path = (RAW_DIR / "studentVle.csv").resolve()
    if not path.exists():
        raise FileNotFoundError(f"studentVle not found: {path}")

    module_list = ", ".join(f"'{m}'" for m in modules)
    filters = [f"code_module IN ({module_list})"]
    if max_day is not None:
        filters.append(f"date <= {int(max_day)}")
    where_clause = " AND ".join(filters)

    sql = f"""
        SELECT
            code_module,
            code_presentation,
            id_student,
            id_site,
            date,
            sum_click
        FROM read_csv_auto(
            ?,
            columns={{
                'code_module': 'VARCHAR',
                'code_presentation': 'VARCHAR',
                'id_student': 'INTEGER',
                'id_site': 'INTEGER',
                'date': 'INTEGER',
                'sum_click': 'INTEGER'
            }}
        )
        WHERE {where_clause}
    """

    con = duckdb.connect()
    try:
        df = con.execute(sql, [str(path)]).df()
    finally:
        con.close()

    return validate_schema(df, "studentVle")


def build_cohort(modules: Sequence[str] = ("BBB", "DDD")) -> pd.DataFrame:
    """Inner-join studentInfo + registration; filter modules; label y_at_risk."""
    tables = load_static_tables()
    info = tables["studentInfo"]
    reg = tables["studentRegistration"]

    cohort = info.merge(
        reg,
        on=["code_module", "code_presentation", "id_student"],
        how="inner",
    )
    cohort = cohort[cohort["code_module"].isin(modules)].copy()

    date_unreg = pd.to_numeric(
        cohort["date_unregistration"].replace("", np.nan),
        errors="coerce",
    )
    pre_course = date_unreg.notna() & (date_unreg < 0)
    cohort = cohort[~pre_course].copy()

    cohort["y_at_risk"] = cohort["final_result"].isin(
        ["Fail", "Withdrawn"]
    ).astype(int)

    return cohort


def save_interim(cohort: pd.DataFrame, clickstream: pd.DataFrame) -> None:
    """Write filtered cohort and day-28 clickstream to data/interim/."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(COHORT_PARQUET, index=False)
    clickstream.to_parquet(CLICKSTREAM_PARQUET, index=False)


MODULE_ORDER: tuple[str, ...] = ("BBB", "DDD")


def modules_for_student(
    student_id: int,
    demographics_test: pd.DataFrame,
    id_student_test: pd.Series,
) -> list[str]:
    """Return enrolled demo modules for a test student, in fixed BBB then DDD order.

    demographics_test aligns positionally with id_student_test (no id_student column).
    """
    sid = int(student_id)
    enrolled: list[str] = []
    for pos, test_sid in enumerate(id_student_test):
        if int(test_sid) != sid:
            continue
        mod = str(demographics_test.iloc[pos]["code_module"])
        if mod not in enrolled:
            enrolled.append(mod)
    return [m for m in MODULE_ORDER if m in enrolled]


def default_demo_student_id(
    demographics_test: pd.DataFrame,
    id_student_test: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    *,
    prefer_dual_module: bool = True,
) -> int:
    """Pick default student for Tab 4 demo (prefer BBB+DDD enrolment)."""
    if prefer_dual_module:
        seen: set[int] = set()
        for test_sid in id_student_test:
            sid = int(test_sid)
            if sid in seen:
                continue
            seen.add(sid)
            if modules_for_student(sid, demographics_test, id_student_test) == [
                "BBB",
                "DDD",
            ]:
                return sid
    if X_test is not None:
        return int(id_student_test.iloc[int(X_test["clicks_total"].argmax())])
    return int(id_student_test.iloc[0])


def rhythm_row_pos_for_student(
    student_id: int,
    demographics_test: pd.DataFrame,
    id_student_test: pd.Series,
    X_test: pd.DataFrame,
) -> int:
    """Row in X_test for full-cohort rhythm chart (max clicks among enrollments)."""
    sid = int(student_id)
    modules = modules_for_student(sid, demographics_test, id_student_test)
    if len(modules) <= 1:
        return id_to_pos(sid, id_student_test)
    best_pos = id_to_pos(sid, id_student_test)
    best_clicks = float(X_test.iloc[best_pos]["clicks_total"])
    for module in modules:
        pos = id_to_pos_module(sid, module, demographics_test, id_student_test)
        clicks = float(X_test.iloc[pos]["clicks_total"])
        if clicks > best_clicks:
            best_clicks = clicks
            best_pos = pos
    return int(best_pos)


def student_picker_label(
    student_id: int,
    demographics_test: pd.DataFrame,
    id_student_test: pd.Series,
) -> str:
    """Selectbox label with enrolled modules."""
    mods = modules_for_student(
        int(student_id), demographics_test, id_student_test
    )
    suffix = ", ".join(mods) if mods else "—"
    return f"Student #{int(student_id)} · {suffix}"


def id_to_pos_module(
    student_id: int,
    code_module: str,
    demographics_test: pd.DataFrame,
    id_student_test: pd.Series,
) -> int:
    """Map (id_student, code_module) to positional row in X_test / SHAP arrays."""
    sid = int(student_id)
    module = str(code_module)
    for pos, test_sid in enumerate(id_student_test):
        if int(test_sid) != sid:
            continue
        if str(demographics_test.iloc[pos]["code_module"]) == module:
            return int(pos)
    raise KeyError(
        f"student_id {sid} not enrolled in module {module} in test set"
    )


def id_to_pos(student_id: int, id_student_test: pd.Series) -> int:
    """Map OULAD id_student to positional row in X_test / SHAP arrays.

    If the student appears in multiple modules, returns the last matching row
    (dict overwrite). Prefer id_to_pos_module when module is known.
    """
    lookup = {int(sid): i for i, sid in enumerate(id_student_test)}
    key = int(student_id)
    if key not in lookup:
        raise KeyError(f"student_id {key} not in test set")
    return lookup[key]


def pos_to_id(pos: int, id_student_test: pd.Series) -> int:
    """Map positional test row to OULAD id_student."""
    if pos < 0 or pos >= len(id_student_test):
        raise IndexError(f"pos {pos} out of range for test set (n={len(id_student_test)})")
    return int(id_student_test.iloc[pos])
