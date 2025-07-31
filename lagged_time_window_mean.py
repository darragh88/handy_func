import pandas as pd

def lagged_time_window_mean(
        s: pd.Series,
        *,
        min_lag="2D",          # how far back data must be (≥ this)
        window="10D",          # how much history you average (≤ this)
        mask=None,             # optional Boolean filter
        min_periods: int = 1   # require at least this many obs in the window
    ) -> pd.Series:
    """
    Rolling mean that uses values in (t - window … t - min_lag] and
    aligns the result on t.

    * min_lag / window can be strings like '2D' or Timedelta objects
    * duplicated time-stamps are fine – every row is treated independently
    """
    if mask is not None:
        s = s.where(mask)

    # step-1: push the series forward by the *minimum* lag you must respect
    s_shifted = s.shift(freq=min_lag)

    # step-2: take a time-based rolling mean on that shifted series
    ma_shifted = (
        s_shifted
        .rolling(window=window, closed="both", min_periods=min_periods)
        .mean()
    )

    # step-3: pull the result back to the original time-line
    return ma_shifted.shift(freq=pd.Timedelta(min_lag) * -1)

if __name__ == "__main__":
    idx = pd.to_datetime([
        "2023-01-01",
        "2023-01-02",
        "2023-01-03",
        "2023-01-07 09:00",   # two intraday points
        "2023-01-07 18:00"
    ])
    s = pd.Series([10, 20, 30, 40, 50], index=idx)

    ma = lagged_time_window_mean(
        s,
        min_lag="2D",
        window="5D"   # whatever look-back span you want
    )
    print(ma)
