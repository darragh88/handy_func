import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- INPUTS --------------------------------------------------
price = ...  # <— your price or signal Series (indexed by DateTime)
model = price.ewm(span=20).mean()  # example EWMA model; replace with yours
returns = price.pct_change()

z_thresh = 3  # how many σ away from mean you call a spike
look_ahead = 10  # days (or bars) you consider “post-spike”
# -------------------------------------------------------------

# Z-score of the return (robust IQR version keeps outliers from inflating σ)
mad = returns.rolling(252).apply(
    lambda x: np.median(np.abs(x - np.median(x))), raw=True)
zscore = (returns - returns.rolling(252).median()) / (1.4826 * mad)

spike_dates = zscore[zscore.abs() >= z_thresh].index

# Compute model error
error = price - model  # raw error
abs_err = error.abs()  # MAE building block
sq_err = error ** 2  # MSE / RMSE building block

# Tag each timestamp with condition
cond = pd.Series("baseline", index=price.index, dtype="object")

for spike in spike_dates:
    end = spike + pd.Timedelta(look_ahead, unit=price.index.freqstr or "D")
    cond.loc[spike:end] = "post-spike"

# Stack into a DataFrame for easy groupby
df = pd.DataFrame({
    "price": price,
    "model": model,
    "error": error,
    "abs_err": abs_err,
    "sq_err": sq_err,
    "cond": cond
})

# Rolling MAE
roll_mae = df["abs_err"].rolling(look_ahead).mean()

# Plot Rolling MAE
fig, ax = plt.subplots()
ax.plot(roll_mae, label=f"Rolling {look_ahead}-period MAE")
ax.set_title("Error over time (spike windows shaded)")
ax.legend()

# Shade post-spike regions
for spike in spike_dates:
    ax.axvspan(spike, spike + pd.Timedelta(look_ahead, unit="D"),
               alpha=0.2, linewidth=0)

plt.show()

# Boxplot absolute error comparison
fig, ax = plt.subplots()
sns.boxplot(x="cond", y="abs_err", data=df, ax=ax)
ax.set_title("Absolute error: post-spike vs baseline")
plt.show()

# T-test
group_a = df.loc[df["cond"] == "post-spike", "abs_err"]
group_b = df.loc[df["cond"] == "baseline", "abs_err"]
t, p = stats.ttest_ind(group_a.dropna(), group_b.dropna(), equal_var=False)
print(f"T-stat {t:.2f}, p-value {p:.4f}")

# Spike size vs subsequent MAE
windows = []
for spike in spike_dates:
    win = df.loc[spike:spike + pd.Timedelta(look_ahead, unit="D")]
    windows.append({
        "spike_size": abs(returns.loc[spike]),
        "post_mae": win["abs_err"].mean()
    })
windf = pd.DataFrame(windows)

plt.figure()
plt.scatter(windf["spike_size"], windf["post_mae"])
plt.xlabel("Spike magnitude (|return|)")
plt.ylabel(f"Mean MAE in next {look_ahead} steps")
plt.title("Bigger spikes → larger subsequent error?")
plt.show()
