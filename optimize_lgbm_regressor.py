
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from typing import Dict, Tuple

# ------------------------------------------------------------------
# helper – yields (train_idx, val_idx) pairs for an expanding window
# ------------------------------------------------------------------
def expanding_splits(n_samples: int,
                     init_train_window: int,
                     test_window: int):
    """
    Example with init=2, test=2, n=10
        train [0:2) -> test [2:4)
        train [0:4) -> test [4:6)
        train [0:6) -> test [6:8)
        train [0:8) -> test [8:10)
    """
    train_end = init_train_window
    while train_end + test_window <= n_samples:
        yield (np.arange(0, train_end),
               np.arange(train_end, train_end + test_window))
        train_end += test_window

# ------------------------------------------------------------------
# main optimiser
# ------------------------------------------------------------------
def optimize_lgbm_regressor(
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int       = 100,
        random_state: int   = 42,
        init_train_window: int = 14,
        test_window: int       = 14,
) -> Tuple[Dict, float, lgb.LGBMRegressor]:

    def objective(trial):
        params = {
            "objective"       : "regression",
            "metric"          : "rmse",
            "verbosity"       : -1,
            "boosting_type"   : "gbdt",
            "num_leaves"      : trial.suggest_int("num_leaves",       20, 150),
            "max_depth"       : trial.suggest_int("max_depth",        3,  15),
            "n_estimators"    : trial.suggest_int("n_estimators",     50, 500),
            "subsample"       : trial.suggest_float("subsample",      0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree",0.5, 1.0),
            "min_child_samples":trial.suggest_int("min_child_samples", 5, 100),
        }

        fold_rmses = []
        for tr_idx, val_idx in expanding_splits(len(X),
                                                init_train_window,
                                                test_window):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            model = lgb.LGBMRegressor(**params, random_state=random_state)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                early_stopping_rounds=50,
                verbose=False,
            )
            preds = model.predict(X_val, num_iteration=model.best_iteration_)
            fold_rmses.append(
                mean_squared_error(y_val, preds, squared=False)
            )

        return float(np.mean(fold_rmses))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_score  = study.best_value

    final_model = lgb.LGBMRegressor(
        **best_params,
        objective="regression",
        metric="rmse",
        random_state=random_state
    )
    final_model.fit(X, y)

    print(f"Best CV RMSE: {best_score:.4f}")
    print("Best hyper-parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    return best_params, best_score, final_model

# Example usage:
# best_params, best_rmse, model = optimize_lgbm_regressor(
#     X,
#     y,
#     n_trials=150,
#     init_train_window=14,
#     test_window=14
# )
