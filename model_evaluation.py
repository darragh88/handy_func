import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Add / swap any metrics you need
METRICS = {
    "RMSE":  lambda y, ŷ: np.sqrt(mean_squared_error(y, ŷ)),     # lower = better
    "MAE":   lambda y, ŷ: mean_absolute_error(y, ŷ),             # lower = better
    "MAPE":  lambda y, ŷ: np.mean(np.abs((y - ŷ) / y))*100,      # lower = better
    "R²":    lambda y, ŷ: r2_score(y, ŷ),                        # higher = better
    "Corr":  lambda y, ŷ: np.corrcoef(y, ŷ)[0, 1],                # higher = better
    "Bias":  lambda y, ŷ: np.mean(ŷ - y)                         # closer to 0 = better
}

SIGN = {m: -1 if m in {"RMSE", "MAE", "MAPE"} else 1 for m in METRICS}

import pandas as pd

# dfs = [df1, df2, ..., df10]   # your DataFrames
results = []
for i, df in enumerate(dfs, start=1):
    y, yhat = df["actuals"].values, df["predictions"].values
    row = {"model": f"Model {i}"}
    for name, fn in METRICS.items():
        row[name] = SIGN[name] * fn(y, yhat)       # sign-adjusted
    results.append(row)

score_df = pd.DataFrame(results).set_index("model")

scaled = score_df.apply(lambda s: (s - s.min()) / (s.max() - s.min()), axis=0)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
for model in scaled.index:
    ax.plot(scaled.columns, scaled.loc[model], marker="o", label=model)

ax.set_ylabel("Scaled (0‒1, higher = better)")
ax.set_xlabel("Metric")
ax.set_title("Overall comparison of 10 models")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

for metric in score_df.columns:
    plt.figure(figsize=(6, 3))
    plt.bar(score_df.index, score_df[metric])
    plt.title(metric)
    plt.ylabel(f"{metric} (signed)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
