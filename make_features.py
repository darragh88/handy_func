import pandas as pd
import numpy as np

def make_features(df, price_col="close"):
    px = df[price_col]

    # 1) Returns
    df["ret_1"] = px.pct_change()
    df["ret_log_1"] = np.log(px).diff()

    # 2) Trend / EMAs
    df["ema_fast"] = px.ewm(span=12, adjust=False).mean()
    df["ema_slow"] = px.ewm(span=26, adjust=False).mean()
    df["ema_diff"] = df["ema_fast"] - df["ema_slow"]
    df["ema_diff_pct"] = df["ema_diff"] / df["ema_slow"]

    # Crossover signal: 1 when fast crosses above slow, -1 when below, 0 otherwise
    cross = np.sign(df["ema_diff"])
    df["ema_cross_signal"] = cross.diff().fillna(0)

    # 3) Momentum: 14-period RSI
    delta = px.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/14, adjust=False).mean()
    roll_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / roll_down
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # 4) Volatility: 20d rolling std of returns
    df["vol_20"] = df["ret_log_1"].rolling(20).std()

    # 5) Bollinger Band width (%)
    sma20 = px.rolling(20).mean()
    std20 = px.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df["bb_width_pct"] = (upper - lower) / sma20

    # 6) Time-since-last-high/low (lookback 60)
    lookback = 60
    rolling_max_idx = px.rolling(lookback).apply(lambda s: np.argmax(s), raw=False)
    rolling_min_idx = px.rolling(lookback).apply(lambda s: np.argmin(s), raw=False)
    # distance from end of window -> "time since"
    df["ts_since_high_60"] = lookback - 1 - rolling_max_idx
    df["ts_since_low_60"]  = lookback - 1 - rolling_min_idx

    return df
