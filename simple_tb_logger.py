
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Dict, Mapping, Union

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


def _utc_timestamp_string() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


DEFAULT_METRICS: Dict[str, Callable[[pd.Series, pd.Series], float]] = {
    "rmse": rmse,
    "mae": mae,
}


def _split_series(series: pd.Series, split: Union[pd.Timestamp, float]) -> tuple[pd.Series, pd.Series]:
    if isinstance(split, float):
        if not 0.0 < split < 1.0:
            raise ValueError("Fraction split must be in (0, 1)")
        n_train = int(len(series) * split)
        return series.iloc[: n_train], series.iloc[n_train:]
    if isinstance(split, pd.Timestamp):
        return series.loc[: split], series.loc[split:]
    raise TypeError("split must be a float or pd.Timestamp")


def log_basic_metrics(
    *,
    y_true: pd.Series,
    y_pred: pd.Series,
    split: Union[pd.Timestamp, float],
    model_name: str,
    run_dir: str | Path = "runs_simple",
    metrics: Dict[str, Callable[[pd.Series, pd.Series], float]] | None = None,
) -> str:
    if metrics is None:
        metrics = DEFAULT_METRICS

    y_true = y_true.sort_index()
    y_pred = y_pred.reindex_like(y_true)

    y_true_train, y_true_test = _split_series(y_true, split)
    y_pred_train, y_pred_test = _split_series(y_pred, split)

    run_path = Path(run_dir) / model_name
    run_path.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_path))
    all_metrics: Dict[str, Dict[str, float]] = {}

    for split_name, yt, yp in [
        ("train", y_true_train, y_pred_train),
        ("test",  y_true_test,  y_pred_test),
    ]:
        all_metrics[split_name] = {}
        for m_name, fn in metrics.items():
            val = fn(yt, yp)
            writer.add_scalar(f"{split_name}/{m_name}", val, global_step=0)
            all_metrics[split_name][m_name] = val

    writer.close()

    cfg = {"model_name": model_name, "split": str(split)}
    with open(run_path / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    with open(run_path / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    return str(run_path)
