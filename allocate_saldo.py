
import pandas as pd
import numpy as np

# region_dfs: Dict[str, pd.DataFrame]
#   keys are region names (e.g. "tokyo", "chubu", …)
#   each df has
#     • 'saldo_kwh'                      – group or solo saldo
#     • 'is_same_wide_area_category_X'   – one column per region X (0/1 flags)

# 1. Pre‑compute, for each region, which columns encode its group‑membership flags
membership_cols = {
    region: [c for c in df.columns if c.startswith('is_same_wide_area_category_')]
    for region, df in region_dfs.items()
}

# 2. Build per‑region “hour‑of‑day → median solo‑saldo” baselines
baselines: dict[str, pd.Series] = {}
for region, df in region_dfs.items():
    # solo = “only this region in its own group”
    solo = df[membership_cols[region]].sum(axis=1) == 1

    # grab absolute solo saldo values
    solo_vals = df.loc[solo, 'saldo_kwh'].abs()

    # median by hour, reindex to 0–23, then fill gaps
    med_by_hour = (
        solo_vals
        .groupby(solo_vals.index.hour)
        .median()
        .reindex(range(24))
    )
    med_by_hour.interpolate(method='linear', inplace=True)

    baselines[region] = med_by_hour

# 3. Allocate every timestamp’s group‑saldo back to individual regions
regions = list(region_dfs.keys())
# all dfs share the same DatetimeIndex
idx = next(iter(region_dfs.values())).index

allocated = pd.DataFrame(0.0, index=idx, columns=regions)

for ts in idx:
    # find which regions are “active” (membership flag == 1 for their own col)
    active = [
        r for r in regions
        if region_dfs[r].at[ts, f'is_same_wide_area_category_{r}'] == 1
    ]
    # the observed group saldo (same across all active dfs)
    Sg = region_dfs[active[0]].at[ts, 'saldo_kwh']

    if len(active) == 1:
        # solo ⇒ just take abs
        allocated.at[ts, active[0]] = abs(Sg)
    else:
        hr = ts.hour
        # look up each region’s baseline weight at this hour
        weights = np.array([baselines[r].loc[hr] for r in active], dtype=float)
        # fallback to equal split if all weights zero/NaN
        if np.nansum(weights) == 0:
            weights = np.ones_like(weights)
        weights /= np.nansum(weights)

        # distribute
        for r, w in zip(active, weights):
            allocated.at[ts, r] = Sg * w

# `allocated` now has one column per region with your estimated per-region saldo_kwh
