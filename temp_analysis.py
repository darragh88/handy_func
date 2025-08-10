import numpy as np
import pandas as pd

EPS = 1e-6  # avoid divide-by-zero

# ---------------------------
# Helpers
# ---------------------------

def _ensure_datetime_index(s: pd.Series) -> pd.Series:
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex.")
    return s.sort_index()

def _hod_index(idx: pd.DatetimeIndex) -> pd.Series:
    return (idx.hour * 2 + (idx.minute // 30)).astype(int)

# ---------------------------
# 0) Fit yearly climatology on TRAIN ONLY (frozen)
# ---------------------------

def fit_yearly_climatology_by_hod(temp_c: pd.Series, train_end: pd.Timestamp):
    """
    Returns dict with mu_by_hod and sd_by_hod (Series indexed by HOD 0..47),
    fitted on [.., train_end].
    """
    s = _ensure_datetime_index(temp_c)
    train = s.loc[:train_end]
    hod = _hod_index(train.index)
    mu_by_hod = train.groupby(hod).mean()
    sd_by_hod = train.groupby(hod).std().fillna(0.0)
    return {"mu_by_hod": mu_by_hod, "sd_by_hod": sd_by_hod}

# ---------------------------
# 1) Rolling same-HOD z-score (10-day): full-series (leak-safe)
# ---------------------------

def compute_rolling_z_same_hod(temp_c: pd.Series, window_days: int = 10, min_days: int = 3) -> pd.Series:
    s = _ensure_datetime_index(temp_c)
    df = pd.DataFrame({"T_C": s})
    df["HOD"] = _hod_index(df.index)
    g = df.groupby("HOD")["T_C"]
    roll_mean = g.shift(1).rolling(window=window_days, min_periods=min_days).mean()
    roll_std  = g.shift(1).rolling(window=window_days, min_periods=min_days).std().fillna(0.0)
    z = (df["T_C"] - roll_mean) / (roll_std.replace(0.0, EPS))
    z.name = "z_10d_same_hod"
    return z

def compute_rolling_mean_same_hod(temp_c: pd.Series, window_days: int = 10, min_days: int = 3) -> pd.Series:
    s = _ensure_datetime_index(temp_c)
    df = pd.DataFrame({"T_C": s})
    df["HOD"] = _hod_index(df.index)
    roll_mean = (
        df.groupby("HOD")["T_C"]
        .shift(1)
        .rolling(window=window_days, min_periods=min_days)
        .mean()
    )
    roll_mean.name = "mean_10d_same_hod"
    return roll_mean

# ---------------------------
# 2) Rolling same-HOD z / mean computed *only within TEST window*
#    (fresh state starting at test_start)
# ---------------------------

def compute_rolling_z_same_hod_test(
    temp_c: pd.Series,
    test_start: pd.Timestamp,
    window_days: int = 10,
    min_days: int = 3,
) -> pd.Series:
    """
    Same as compute_rolling_z_same_hod, but computes rolling stats using
    ONLY data from [test_start, ...]. No train history is used.
    """
    s = _ensure_datetime_index(temp_c)
    test = s.loc[test_start:]
    df = pd.DataFrame({"T_C": test})
    df["HOD"] = _hod_index(df.index)
    g = df.groupby("HOD")["T_C"]
    roll_mean = g.shift(1).rolling(window=window_days, min_periods=min_days).mean()
    roll_std  = g.shift(1).rolling(window=window_days, min_periods=min_days).std().fillna(0.0)
    z = (df["T_C"] - roll_mean) / (roll_std.replace(0.0, EPS))
    z.name = "z_10d_same_hod_test"
    return z

def compute_rolling_mean_same_hod_test(
    temp_c: pd.Series,
    test_start: pd.Timestamp,
    window_days: int = 10,
    min_days: int = 3,
) -> pd.Series:
    """
    Rolling 10-day same-HOD MEAN using ONLY data from [test_start, ...].
    """
    s = _ensure_datetime_index(temp_c)
    test = s.loc[test_start:]
    df = pd.DataFrame({"T_C": test})
    df["HOD"] = _hod_index(df.index)
    roll_mean = (
        df.groupby("HOD")["T_C"]
        .shift(1)
        .rolling(window=window_days, min_periods=min_days)
        .mean()
    )
    roll_mean.name = "mean_10d_same_hod_test"
    return roll_mean

# ---------------------------
# 3) Background regime (hot/medium/cold) vs frozen yearly baseline
#    (works for train or test; pass the rolling mean you want)
# ---------------------------

def classify_background_regime(
    rolling_mean_same_hod: pd.Series,
    index: pd.DatetimeIndex,
    mu_by_hod: pd.Series,
    sd_by_hod: pd.Series,
    z_threshold_hot: float = 0.5,
    z_threshold_cold: float = -0.5,
) -> pd.Series:
    """
    Regime z = (rolling_mean - mu_hod) / sd_hod, with mu/sd frozen from train.
      - 'hot'   if z >= z_threshold_hot
      - 'cold'  if z <= z_threshold_cold
      - 'medium' otherwise
    """
    hod = _hod_index(index)
    mu = pd.Series(hod, index=index).map(mu_by_hod)
    sd = pd.Series(hod, index=index).map(sd_by_hod).replace(0.0, EPS)

    z_bg = (rolling_mean_same_hod.reindex(index) - mu) / sd
    regime = pd.Series("medium", index=index, dtype="object")
    regime[z_bg >= z_threshold_hot] = "hot"
    regime[z_bg <= z_threshold_cold] = "cold"
    regime.name = "regime_bg"
    return regime

# ---------------------------
# 4) Spike indicators gated by regime
# ---------------------------

def hot_spike_indicator(z_rolling: pd.Series, regime_bg: pd.Series, z_spike_threshold: float = 1.0) -> pd.Series:
    z = z_rolling.reindex(regime_bg.index)
    out = (regime_bg.eq("hot") & (z >= z_spike_threshold)).astype(int)
    out.name = "hot_spike_indicator"
    return out

def cold_spike_indicator(z_rolling: pd.Series, regime_bg: pd.Series, z_spike_threshold: float = 1.0) -> pd.Series:
    z = z_rolling.reindex(regime_bg.index)
    out = (regime_bg.eq("cold") & (z <= -z_spike_threshold)).astype(int)
    out.name = "cold_spike_indicator"
    return out


# Example usage
if __name__ == "__main__":
    # temp: pd.Series of °C at 30-min freq (DatetimeIndex)
    # Replace with your actual temperature series
    temp = pd.Series(dtype=float)  

    train_end  = pd.Timestamp("2023-10-01")
    test_start = train_end + pd.Timedelta(minutes=30)

    # 1) Freeze yearly baseline from TRAIN
    clim = fit_yearly_climatology_by_hod(temp, train_end)
    mu_by_hod, sd_by_hod = clim["mu_by_hod"], clim["sd_by_hod"]

    # 2) TRAIN rolling (optional, if you want features on train too)
    z10_train = compute_rolling_z_same_hod(temp.loc[:train_end])
    m10_train = compute_rolling_mean_same_hod(temp.loc[:train_end])

    # 3) TEST rolling computed *within* test window only
    z10_test = compute_rolling_z_same_hod_test(temp, test_start, window_days=10, min_days=3)
    m10_test = compute_rolling_mean_same_hod_test(temp, test_start, window_days=10, min_days=3)

    # 4) Regimes (use the rolling mean you prefer: train or test)
    regime_test = classify_background_regime(
        rolling_mean_same_hod=m10_test,
        index=m10_test.index,
        mu_by_hod=mu_by_hod,
        sd_by_hod=sd_by_hod,
        z_threshold_hot=0.5,
        z_threshold_cold=-0.5,
    )

    # 5) Spike indicators gated by regime
    hot_ind_test  = hot_spike_indicator(z10_test, regime_test, z_spike_threshold=1.0)
    cold_ind_test = cold_spike_indicator(z10_test, regime_test, z_spike_threshold=1.0)

    # Final test features
    test_features = pd.concat([z10_test, m10_test, regime_test, hot_ind_test, cold_ind_test], axis=1)
