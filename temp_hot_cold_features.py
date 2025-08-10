import pandas as pd
import numpy as np

EPS = 1e-9  # avoid divide-by-zero in std

def _check(df: pd.DataFrame, temp_col: str) -> pd.DataFrame:
    if temp_col not in df.columns:
        raise ValueError(f"'{temp_col}' not in DataFrame.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex.")
    return df.sort_index()

def _hod(idx: pd.DatetimeIndex) -> pd.Series:
    # half-hour-of-day bucket: 0..47
    return (idx.hour * 2 + (idx.minute // 30)).astype(int)

# 1) Rolling 10-day same-time z-score + hot/cold flags (leak-safe)
def add_rolling_z_hot_cold(
    df: pd.DataFrame,
    temp_col: str = "temp",
    *,
    window_days: int = 10,
    min_days: int = 3,
    z_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Adds 3 columns:
      - z_roll_10d_samehod
      - is_hot_roll  (1 if z >=  z_threshold)
      - is_cold_roll (1 if z <= -z_threshold)

    Rolling stats use ONLY the past `window_days` days at the SAME half-hour (shift(1) ⇒ no leakage).
    """
    df = _check(df.copy(), temp_col)
    df["_HOD"] = _hod(df.index)

    g = df.groupby("_HOD")[temp_col]
    roll_mean = g.shift(1).rolling(window=window_days, min_periods=min_days).mean()
    roll_std  = g.shift(1).rolling(window=window_days, min_periods=min_days).std().fillna(0.0)

    z = (df[temp_col] - roll_mean) / roll_std.replace(0.0, EPS)
    df["z_roll_10d_samehod"] = z
    df["is_hot_roll"]  = (z >=  z_threshold).astype(int)
    df["is_cold_roll"] = (z <= -z_threshold).astype(int)

    df.drop(columns=["_HOD"], inplace=True)
    return df

# 2) “Yearly” z-score: train uses frozen μ/σ by HOD; test uses fresh rolling μ/σ by HOD
def add_yearly_z_hot_cold(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    temp_col: str = "temp",
    *,
    window_days_test: int = 10,
    min_days_test: int = 3,
    z_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Adds 3 columns:
      - z_year       : TRAIN → (T - μ_hod_train)/σ_hod_train;
                       TEST  → (T - rolling_mean_test)/rolling_std_test   (same-HOD, leak-safe)
      - is_hot_year  : 1 if z_year >=  z_threshold
      - is_cold_year : 1 if z_year <= -z_threshold
    """
    df = _check(df.copy(), temp_col)
    df["_HOD"] = _hod(df.index)

    # Split
    train_mask = df.index <= pd.Timestamp(train_end)
    test_mask  = ~train_mask

    # ---- TRAIN: frozen μ/σ by HOD (computed on train only)
    mu_by_hod = df.loc[train_mask].groupby("_HOD")[temp_col].mean()
    sd_by_hod = df.loc[train_mask].groupby("_HOD")[temp_col].std().fillna(0.0)

    mu_train = df["_HOD"].map(mu_by_hod)
    sd_train = df["_HOD"].map(sd_by_hod).replace(0.0, EPS)
    z_train = (df[temp_col] - mu_train) / sd_train

    # ---- TEST: rolling μ/σ within TEST window only (same-HOD, leak-safe)
    if test_mask.any():
        test_df = df.loc[test_mask, [temp_col, "_HOD"]].copy()
        g = test_df.groupby("_HOD")[temp_col]
        roll_mean = g.shift(1).rolling(window=window_days_test, min_periods=min_days_test).mean()
        roll_std  = g.shift(1).rolling(window=window_days_test, min_periods=min_days_test).std().fillna(0.0)
        z_test = (test_df[temp_col] - roll_mean) / roll_std.replace(0.0, EPS)
        # place back
        z_year = z_train.copy()
        z_year.loc[test_mask] = z_test
    else:
        z_year = z_train

    df["z_year"] = z_year
    df["is_hot_year"]  = (df["z_year"] >=  z_threshold).astype(int)
    df["is_cold_year"] = (df["z_year"] <= -z_threshold).astype(int)

    df.drop(columns=["_HOD"], inplace=True)
    return df
