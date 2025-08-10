import pandas as pd
import numpy as np

EPS = 1e-9

def _check(df: pd.DataFrame, temp_col: str) -> pd.DataFrame:
    if temp_col not in df.columns:
        raise ValueError(f"'{temp_col}' not in DataFrame.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex.")
    return df.sort_index()

def _hod(idx: pd.DatetimeIndex) -> pd.Series:
    # half-hour-of-day: 0..47
    return (idx.hour * 2 + (idx.minute // 30)).astype(int)

# 1) Rolling 10-day same-time z-score + hot/cold flags (NO shift)
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

    Rolling stats are computed within each half-hour-of-day over the last `window_days`
    occurrences (one per day per HOD). No shift is applied.
    """
    df = _check(df.copy(), temp_col)
    df["_HOD"] = _hod(df.index)

    # rolling mean/std per HOD (includes current timestamp)
    roll_mean = df.groupby("_HOD")[temp_col] \                  .transform(lambda x: x.rolling(window_days, min_periods=min_days).mean())
    roll_std  = df.groupby("_HOD")[temp_col] \                  .transform(lambda x: x.rolling(window_days, min_periods=min_days).std()).fillna(0.0)

    z = (df[temp_col] - roll_mean) / roll_std.replace(0.0, EPS)
    df["z_roll_10d_samehod"] = z
    df["is_hot_roll"]  = (z >=  z_threshold).astype(int)
    df["is_cold_roll"] = (z <= -z_threshold).astype(int)

    df.drop(columns=["_HOD"], inplace=True)
    return df

# 2) “Yearly” feature:
#    - TRAIN: static μ/σ by HOD fitted on train only.
#    - TEST : 1-year rolling μ/σ by HOD computed over the *entire* df (no shift).
#    Combine -> return z_year, hot/cold flags.
def add_yearly_z_hot_cold(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    temp_col: str = "temp",
    *,
    window_days_year: int = 365,   # occurrences per HOD (≈ days)
    min_days_year: int = 30,
    z_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Adds 3 columns:
      - z_year
      - is_hot_year  (1 if z_year >=  z_threshold)
      - is_cold_year (1 if z_year <= -z_threshold)
    """
    df = _check(df.copy(), temp_col)
    df["_HOD"] = _hod(df.index)

    train_end = pd.Timestamp(train_end)
    train_mask = df.index <= train_end
    test_mask  = ~train_mask

    # TRAIN: μ/σ by HOD on train only
    mu_by_hod = df.loc[train_mask].groupby("_HOD")[temp_col].mean()
    sd_by_hod = df.loc[train_mask].groupby("_HOD")[temp_col].std().fillna(0.0)
    mu_train = df["_HOD"].map(mu_by_hod)
    sd_train = df["_HOD"].map(sd_by_hod).replace(0.0, EPS)
    z_train  = (df[temp_col] - mu_train) / sd_train

    # FULL DF: 1-year rolling μ/σ by HOD (no shift)
    roll_mean_year = df.groupby("_HOD")[temp_col] \                       .transform(lambda x: x.rolling(window_days_year, min_periods=min_days_year).mean())
    roll_std_year  = df.groupby("_HOD")[temp_col] \                       .transform(lambda x: x.rolling(window_days_year, min_periods=min_days_year).std()).fillna(0.0)
    z_roll_year = (df[temp_col] - roll_mean_year) / roll_std_year.replace(0.0, EPS)

    # COMBINE: train uses static; test uses rolling-year
    z_year = pd.Series(np.nan, index=df.index, dtype=float)
    z_year.loc[train_mask] = z_train.loc[train_mask]
    z_year.loc[test_mask]  = z_roll_year.loc[test_mask]

    df["z_year"] = z_year
    df["is_hot_year"]  = (df["z_year"] >=  z_threshold).astype(int)
    df["is_cold_year"] = (df["z_year"] <= -z_threshold).astype(int)

    df.drop(columns=["_HOD"], inplace=True)
    return df
