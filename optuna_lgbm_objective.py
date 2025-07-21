
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# Define the expanding splits function
def expanding_splits(n_samples: int, init_train_window: int, test_window: int):
    train_end = init_train_window
    while train_end + test_window <= n_samples:
        yield (np.arange(0, train_end),
               np.arange(train_end, train_end + test_window))
        train_end += test_window

# Define the objective function for Optuna
def objective(trial):
    params = {
        "objective":        "regression",
        "boosting_type":    "gbdt",
        "verbosity":        -1,

        # --- search space ---
        "num_leaves":       trial.suggest_int("num_leaves", 20, 150),
        "max_depth":        trial.suggest_int("max_depth", 3, 15),
        "n_estimators":     trial.suggest_int("n_estimators", 50, 500),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "subsample_freq":   1,  # turn bagging on
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "min_child_samples":trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha":        trial.suggest_float("reg_alpha",  1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    fold_rmses = []
    for tr_idx, val_idx in expanding_splits(len(X), init_train_window, test_window):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params, random_state=random_state)

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)]   # silence training log
        )

        preds = model.predict(X_val, num_iteration=model.best_iteration_)
        fold_rmses.append(mean_squared_error(y_val, preds, squared=False))

    return float(np.mean(fold_rmses))
