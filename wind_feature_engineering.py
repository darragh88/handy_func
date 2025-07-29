import pandas as pd
from typing import List

###############################################################################
# 1. Forecast‑revision features
###############################################################################

def add_forecast_diff(
    df: pd.DataFrame,
    f1: str = "ec12",
    f2: str = "ec00",
    out: str = "diff",
) -> pd.DataFrame:
    """
    Add a simple difference column: forecast_run_1 – forecast_run_2
    (no lagging needed because rows are already aligned).
    """
    df[out] = df[f1] - df[f2]
    return df


def add_rolling_abs_deviation(
    df: pd.DataFrame,
    col: str = "diff",
    windows: List[int] = [12, 24, 48, 336],  # 6 h, 12 h, 24 h, seven days
    prefix: str = "absdev",
) -> pd.DataFrame:
    """
    For every window in `windows` (number of half‑hour intervals),
    create a column containing the rolling SUM of |value|.
    """
    for w in windows:
        df[f"{prefix}_{w}"] = df[col].abs().rolling(window=w, min_periods=1).sum()
    return df


###############################################################################
# 2. Ramp & acceleration features
###############################################################################

def add_ramp_and_accel(
    df: pd.DataFrame,
    col: str = "ec12",
    ramp_col: str = "ramp",
    accel_col: str = "accel",
) -> pd.DataFrame:
    """
    • ramp  = first forward difference (Δ over 30 min)  
    • accel = second difference (Δ of ramp)
    """
    df[ramp_col]  = df[col].diff()
    df[accel_col] = df[ramp_col].diff()
    return df


###############################################################################
# 3. Period‑of‑day weekly ramp benchmark
###############################################################################

def add_weekly_period_ramp_stats(
    df: pd.DataFrame,
    ramp_col: str = "ramp",
    avg_col: str = "ramp_wk_avg",
    dev_col: str = "ramp_vs_wk_avg",
) -> pd.DataFrame:
    """
    For each half‑hour slot (0‑47) compute the mean ramp
    *over the previous 7 occurrences* (≈ one week) **of that same slot**,
    then store both the average and today's deviation from it.
    """
    # Identify the slot 0‑47 for every row
    slot = (df.index.hour * 2) + (df.index.minute // 30)
    df["_slot"] = slot

    # Rolling mean within each slot series (length‑7 window)
    df[avg_col] = (
        df.groupby("_slot")[ramp_col]
          .apply(lambda s: s.rolling(7, min_periods=1).mean())
          .reset_index(level=0, drop=True)
    )

    # Deviation of today’s ramp from its slot‑wise 1‑week average
    df[dev_col] = df[ramp_col] - df[avg_col]

    # Clean up helper column
    df.drop(columns="_slot", inplace=True)
    return df


###############################################################################
# 4. Convenience wrapper
###############################################################################

def engineer_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the whole feature‑engineering pipeline in one call.
    """
    df = (
        df.pipe(add_forecast_diff)                  # diff = ec12 – ec00
          .pipe(add_rolling_abs_deviation)          # absdev_12, 24, 48, 336
          .pipe(add_ramp_and_accel)                 # ramp, accel
          .pipe(add_weekly_period_ramp_stats)       # ramp_wk_avg, ramp_vs_wk_avg
    )
    return df

###############################################################################
# 2 b.  Multi‑period ramp (e.g. 1 h, 1.5 h, 3 h)
###############################################################################

def add_multi_period_ramps(
    df: pd.DataFrame,
    col: str = "ec12",
    hours: list = [1, 1.5],          # lengths you’re interested in
    prefix: str = "ramp",
) -> pd.DataFrame:
    """
    For each value in `hours`, compute the forecast change over
    that many hours.  Data are half‑hourly, so steps = hours * 2.

    Example:
        hours = [1, 1.5] ➜
        ramp_1h  = F_t – F_(t‑2)
        ramp_1p5h = F_t – F_(t‑3)
    """
    for h in hours:
        steps = int(round(h * 2))            # convert hours ➜ half‑hour steps
        # Nice column name: "ramp_1h", "ramp_1p5h", etc.
        label = str(h).replace(".", "p")
        df[f"{prefix}_{label}h"] = df[col].diff(periods=steps)
    return df
