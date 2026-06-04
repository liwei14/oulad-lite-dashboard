"""Per-group fairness audit with Wilson confidence intervals."""

from __future__ import annotations

from math import sqrt
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.data_loader import normalize_imd_band

IMD_BAND_ORDER = [
    "0-10%",
    "10-20%",
    "20-30%",
    "30-40%",
    "40-50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90-100%",
    "Missing",
]


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (pure Python)."""
    if n == 0:
        return (0.0, 1.0)
    if alpha != 0.05:
        raise NotImplementedError("This demo only uses alpha=0.05")
    z = 1.959963984540054
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _normalize_group_labels(series: pd.Series, attribute: str | None = None) -> pd.Series:
    out = series.astype("object").copy()
    if attribute == "imd_band":
        out = normalize_imd_band(out)
    mask = out.isna() | (out.astype(str).str.strip() == "")
    out.loc[mask] = "Missing"
    return out.astype(str)


def _sort_groups(groups: pd.Series, attribute: str) -> list[str]:
    unique = groups.unique().tolist()
    if attribute == "imd_band":
        order = {label: i for i, label in enumerate(IMD_BAND_ORDER)}
        return sorted(unique, key=lambda g: order.get(g, len(IMD_BAND_ORDER)))
    return sorted(unique)


def flag_rate_table(
    predictions: pd.Series,
    demographics: pd.DataFrame,
    attribute: str,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Per-group flag rate, base rate, TPR/FPR with Wilson CIs."""
    if len(predictions) != len(demographics) or len(predictions) != len(y_test):
        raise ValueError("predictions, demographics, and y_test must align in length")

    y_test = y_test.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)
    demographics = demographics.reset_index(drop=True)

    y_pred = (predictions >= threshold).astype(int)
    groups = _normalize_group_labels(demographics[attribute], attribute=attribute)

    rows = []
    for group in _sort_groups(groups, attribute):
        mask = groups == group
        n = int(mask.sum())
        y_g = y_test[mask]
        pred_g = y_pred[mask]

        base_rate = float(y_g.mean()) if n > 0 else 0.0

        k_flag = int(pred_g.sum())
        flag_rate = k_flag / n if n > 0 else 0.0
        flag_lo, flag_hi = wilson_ci(k_flag, n)

        positives = y_g == 1
        negatives = y_g == 0
        n_pos = int(positives.sum())
        n_neg = int(negatives.sum())

        if n_pos > 0:
            k_tp = int((pred_g[positives] == 1).sum())
            tpr = k_tp / n_pos
            tpr_lo, tpr_hi = wilson_ci(k_tp, n_pos)
        else:
            tpr = 0.0
            tpr_lo, tpr_hi = (0.0, 1.0)

        if n_neg > 0:
            k_fp = int((pred_g[negatives] == 1).sum())
            fpr = k_fp / n_neg
            fpr_lo, fpr_hi = wilson_ci(k_fp, n_neg)
        else:
            fpr = 0.0
            fpr_lo, fpr_hi = (0.0, 1.0)

        rows.append(
            {
                attribute: group,
                "n": n,
                "base_rate": base_rate,
                "flag_rate": flag_rate,
                "flag_rate_ci_lower": flag_lo,
                "flag_rate_ci_upper": flag_hi,
                "true_positive_rate": tpr,
                "tpr_ci_lower": tpr_lo,
                "tpr_ci_upper": tpr_hi,
                "false_positive_rate": fpr,
                "fpr_ci_lower": fpr_lo,
                "fpr_ci_upper": fpr_hi,
                "warn_small_n": n < 30,
            }
        )

    return pd.DataFrame(rows)


def audit(
    predictions: pd.Series,
    y_test: pd.Series,
    demographics_test: pd.DataFrame,
    attributes: Iterable[str] = ("imd_band", "gender", "disability"),
) -> dict[str, pd.DataFrame]:
    """Run flag_rate_table for each protected attribute."""
    return {
        attr: flag_rate_table(predictions, demographics_test, attr, y_test)
        for attr in attributes
    }


def print_audit(audit_dict: dict[str, pd.DataFrame], attribute: Optional[str] = None) -> None:
    """Print fairness tables with CI columns and small-n notice."""
    print("Groups with n<30 are flagged — point estimates are noisy.")
    keys = [attribute] if attribute else list(audit_dict.keys())
    for key in keys:
        if key not in audit_dict:
            raise KeyError(f"Attribute not in audit: {key}")
        print(f"\n=== Fairness audit: {key} ===")
        print(audit_dict[key].to_string(index=False))
