from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _safe_corr(fn, x: Iterable[float], y: Iterable[float]) -> float:
    x_array = np.asarray(list(x), dtype=np.float64)
    y_array = np.asarray(list(y), dtype=np.float64)
    if x_array.size < 2 or y_array.size < 2:
        return float("nan")
    if np.allclose(x_array, x_array[0]) or np.allclose(y_array, y_array[0]):
        return float("nan")
    return float(fn(x_array, y_array)[0])


def srcc(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return _safe_corr(spearmanr, y_true, y_pred)


def plcc(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return _safe_corr(pearsonr, y_true, y_pred)


def macro_user_correlations(frame: pd.DataFrame) -> dict[str, float]:
    rows = []
    for _, group in frame.groupby("user_id"):
        rows.append({"srcc": srcc(group["score"], group["prediction"]), "plcc": plcc(group["score"], group["prediction"])})
    metrics = pd.DataFrame(rows)
    return {
        "macro_srcc": float(metrics["srcc"].dropna().mean()) if not metrics.empty else float("nan"),
        "macro_plcc": float(metrics["plcc"].dropna().mean()) if not metrics.empty else float("nan"),
    }


def same_image_rank_correlation(frame: pd.DataFrame) -> float:
    correlations: list[float] = []
    for _, group in frame.groupby("image_id"):
        value = srcc(group["score"], group["prediction"])
        if not np.isnan(value):
            correlations.append(value)
    return float(np.mean(correlations)) if correlations else float("nan")


def threshold_accuracy(frame: pd.DataFrame, threshold: float) -> float:
    truth = (frame["score"] >= threshold).astype(np.int64)
    pred = (frame["prediction"] >= threshold).astype(np.int64)
    return float((truth == pred).mean())
