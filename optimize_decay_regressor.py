import pandas as pd
import numpy as np
import optuna
from typing import List, Optional

from shared.predictive_model.decay_regress import DecayRegressor
from shared.predictive_model.window import ExpandingRefittingWindow
from shared.predictive_model import PredictiveModel
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
    plot_parallel_coordinate,
    plot_contour,
    plot_edf,
)
from delorean.stats import regress_robust


def mean_helper_fixed(scalar: float = 0.0, x_names: List[str] = None) -> pd.Series:
    mean = np.ones(len(x_names)) * scalar
    return pd.Series(mean, index=x_names)


def covar_helper(raw_vars: np.ndarray, x_names) -> pd.DataFrame:
    """Build a diagonal covariance matrix from a 1D array of variances."""
    cov = np.diag(raw_vars)
    return pd.DataFrame(cov, index=x_names, columns=x_names)


def optimize_decay_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 100,
    plot: bool = False,
    *,
    default_covar: Optional[pd.DataFrame] = None,
    tune_cov_names: Optional[List[str]] = None,
):
    """
    Tune a subset of diagonal covariance variances and the decay_scale.

    Args:
        X, y: training frame/series
        n_trials: Optuna trials
        plot: show Optuna plots
        default_covar: square DataFrame, index=columns=X.columns (base prior covariance)
        tune_cov_names: subset of X.columns to tune; others stay at default values

    Returns: dict with fields { 'study', 'best_params', 'prior_covar', 'decay_scale' }
    """
    x_names = list(X.columns)

    # ---- validate tune list
    tune_cov_names = tune_cov_names or []
    missing = [n for n in tune_cov_names if n not in x_names]
    if missing:
        raise ValueError(f"tune_cov_names has names not in X.columns: {missing}")

    # ---- build base variances (1D) from default_covar (or identity)
    if default_covar is None:
        base_vars = pd.Series(1.0, index=x_names, dtype=float)
    else:
        default_covar = default_covar.reindex(index=x_names, columns=x_names)
        if default_covar.isnull().any().any():
            raise ValueError("default_covar must cover all X.columns on both axes.")
        base_vars = pd.Series(np.diag(default_covar.values), index=x_names, dtype=float)

    # ---- static parts of the model config
    BASE_MODEL_CONFIG = {
        "x_names": X.columns,
        "y_name": "da_imb_spread",
        "model_name": "optimizer",
        "model_uri": None,
        "model_class": DecayRegressor,
        "model_params": {
            "shift_betas": {"periods": 2, "freq": "D"},
            # 'decay_scale', 'prior_mean', 'prior_covar' injected per-trial
        },
        "window_class": ExpandingRefittingWindow,
        "loadx": lambda start, end: 10,
        "loady": lambda start, end: 10,
        "window_params": {
            "start": pd.Timestamp(ts_input="2023-01-01", tz="UTC"),
            "buffer": pd.Timedelta(days=1),
            "freq": "30min",
        },
    }

    # ---- objective
    def objective(trial: optuna.Trial) -> float:
        # start from defaults
        trial_vars = base_vars.copy()

        # suggest only for selected names; others remain at default
        # fixed, sensible bounds relative to the default value
        for name in tune_cov_names:
            b = float(trial_vars.loc[name])
            low = max(b * 1e-2, 1e-10)
            high = max(b * 1e2, 1e-8)
            var = trial.suggest_float(f"cov_var__{name}", low, high, log=True)
            trial_vars.loc[name] = float(var)

        # decay_scale (keep fixed search range inside function)
        days = trial.suggest_int("decay_scale_days", 5, 500)

        # build model config for this trial
        model_cfg = dict(BASE_MODEL_CONFIG)
        model_params = dict(BASE_MODEL_CONFIG["model_params"])
        model_params.update(
            {
                "decay_scale": pd.Timedelta(days=int(days)),
                "prior_mean": mean_helper_fixed(0.0, X.columns),
                "prior_covar": covar_helper(trial_vars.values, X.columns),
            }
        )
        model_cfg["model_params"] = model_params

        # fit & predict
        model = PredictiveModel(**model_cfg)
        model.fit(X, y)
        preds = model.model.walkforward_predict(X, shift_betas={"periods": 2, "freq": "D"})

        # metric
        target, pred = y.dropna().align(preds.dropna(), join="inner")
        return regress_robust([pred.values], target.values).stats.r2

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    # ---- reconstruct the best prior covariance & decay scale
    best = study.best_trial.params
    best_vars = base_vars.copy()
    for name in tune_cov_names:
        key = f"cov_var__{name}"
        if key in best:
            best_vars.loc[name] = float(best[key])

    best_decay_days = int(best.get("decay_scale_days", 0))
    best_prior_covar = covar_helper(best_vars.values, X.columns)

    if plot:
        plot_optimization_history(study).show()
        plot_param_importances(study).show()
        plot_slice(study).show()
        plot_parallel_coordinate(study).show()
        plot_contour(study).show()
        plot_edf(study).show()

    print(f"Best value: {study.best_value}")
    print("Best hyperparameters:")
    for k, v in best.items():
        print(f"  {k}: {v}")

    return {
        "study": study,
        "best_params": best,
        "prior_covar": best_prior_covar,
        "decay_scale": pd.Timedelta(days=best_decay_days),
    }