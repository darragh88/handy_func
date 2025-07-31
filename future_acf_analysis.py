import numpy as np
import matplotlib.pyplot as plt

def future_acf(event_windows):
    """
    event_windows : list of lists or 2D-array
        Each inner list/row is [x₀, x₁, x₂, …, x₇], where
        x₀ = value at the spike event,
        xₖ = value k steps ahead (k=1..7).
    Returns
    -------
    lags : array of ints (1..7)
    acfs : array of float, the correlation at each lag
    """
    arr = np.asarray(event_windows)
    if arr.ndim != 2 or arr.shape[1] < 8:
        raise ValueError("Each event window must have ≥8 elements")

    x0 = arr[:, 0]  # spike values
    lags = np.arange(1, arr.shape[1])
    acfs = []
    for k in lags:
        xk = arr[:, k]
        # Pearson corr; nan’s are dropped automatically by np.corrcoef
        r = np.corrcoef(x0, xk)[0, 1]
        acfs.append(r)
    return lags, np.array(acfs)

# Example usage:
# Suppose you have 100 spike events, each with the next 7 observations recorded
# in a list-of-lists called `windows` of shape (100, 8):
#   windows[i] = [spread_at_spike_i,
#                 spread_1step_after_i,
#                 …,
#                 spread_7steps_after_i]
#
# lags, acfs = future_acf(windows)
# plt.bar(lags, acfs)
# plt.xlabel("Lag (steps ahead)")
# plt.ylabel("Corr(spike, future value)")
# plt.title("Future-ACF of Spread Post-Spike")
# plt.ylim(-1, 1)
# plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) YOUR INPUT SERIES
# --------------------
# spread: pd.Series, DatetimeIndex at 30 min freq, length ≫ 7 days
# e.g. spread.index.freq == '30T'
spread = …

# 2) PIVOT INTO A DAY×48 TABLE
# ----------------------------
#    Rows = calendar days; columns = time‐of‐day slots (datetime.time)
df = spread.to_frame(name="value")
df["date"] = df.index.floor("D")
df["tod"]  = df.index.time

daily = df.pivot(index="date", columns="tod", values="value").sort_index()
# daily.shape == (#days, 48)

# 3) PICK YOUR SLOT
# -----------------
#    Could be any of daily.columns, e.g. 00:00:00, 00:30:00, …, 23:30:00
slot = pd.to_datetime("10:00").time()   # ← change to your target half-hour
series_slot = daily[slot].dropna()

# 4) COMPUTE FUTURE ACF AT LAGS 1…7
# ---------------------------------
lags = np.arange(1, 8)
acfs = [series_slot.autocorr(lag=lag) for lag in lags]

# 5) PLOT
# -------
plt.bar(lags, acfs)
plt.xlabel("Lag (days ahead)")
plt.ylabel("Autocorrelation")
plt.title(f"Future ACF at slot {slot} (lags 1–7 days)")
plt.ylim(-1, 1)
plt.show()
