import pandas as pd
import numpy as np

def lagged_moving_average(
        s: pd.Series,
        *,
        lag="2D",              # how “old” the newest usable value must be
        window=None,           # e.g. "10D"; None = unlimited history
        mask=None,             # Boolean Series of same index, optional
        min_periods: int = 1   # at least this many obs to emit a number
) -> pd.Series:
    """
    Mean of values in (t - window .. t - lag]  aligned to the original index t.
    - `lag` / `window` accept strings like "2D", "36H", or Timedelta objects.
    - Duplicated time-stamps are fine; every row is treated independently.
    """
    if mask is not None:
        s = s.where(mask)       # rows that fail the mask become NaN

    lag = pd.Timedelta(lag)
    if window is None:
        # make the window large enough to span the whole series
        window = s.index[-1] - s.index[0] + pd.Timedelta("1ns")
    window = pd.Timedelta(window)

    # --- 1. push the data *forward* by `lag` days --------------------------
    s_lag = s.shift(freq=lag)

    # --- 2. make sure rolling sees both the shifted data *and* the
    #       original time-stamps we want results for ------------------------
    all_idx = s.index.union(s_lag.index)          # sorted, deduplicated
    s_lag   = s_lag.reindex(all_idx)

    # --- 3. rolling mean on that shifted series ---------------------------
    # Window ends at “now” (t), and because we shifted ahead by `lag`,
    # the newest real observation it can ever include is t - lag.
    ma = (
        s_lag
        .rolling(window=window, closed="both", min_periods=min_periods)
        .mean()
    )

    # --- 4. Pull the numbers back to the rows we care about ---------------
    return ma.reindex(s.index)

if __name__ == "__main__":
    # Example usage
    idx = pd.to_datetime([
        "2023-01-01",
        "2023-01-02",
        "2023-01-03",
        "2023-01-07 09:00",
        "2023-01-07 18:00"
    ])
    s = pd.Series([10, 20, 30, 40, 50], index=idx)
    ma = lagged_moving_average(
        s,
        lag="2D",
        window="5D"
    )
    print(ma)
