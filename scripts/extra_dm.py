"""Post-hoc reading of the frozen test forecasts (no refit, no new forecasts): DM tests of selected pairs other
than "vs HAR", written to results/dm_extra.csv. Same pooled DM as the main table (daily cross-sectional mean of
the QLIKE differential, Newey-West lag 7)."""
import numpy as np
import pandas as pd
from kvh import config
from kvh.metrics import diebold_mariano, qlike

fc = pd.read_csv(config.RESULTS / "forecasts_test.csv")
w = fc.pivot_table(index=["date", "symbol"], columns="model", values="forecast")
w = w[[c for c in w.columns if "base" not in c]].dropna()  # full panel (Kronos-base only covers a subset)
rv = fc.drop_duplicates(["date", "symbol"]).set_index(["date", "symbol"])["rv"].reindex(w.index)
q = pd.DataFrame({m: qlike(rv.to_numpy(), w[m].to_numpy()) for m in w.columns}, index=w.index)
pairs = [("Kronos-small", "log-HAR"), ("Kronos-mini", "log-HAR"), ("Combo: log-HAR x Kronos-small", "log-HAR"),
         ("Student", "Kronos-small"), ("Student", "log-HAR"), ("Kronos-mini", "Kronos-small"), ("LightGBM", "log-HAR")]
rows = []
for a, b in pairs:
    d = (q[a] - q[b]).groupby(level="date").mean().to_numpy()
    st, p, md = diebold_mariano(d, np.zeros_like(d), 7)
    rows.append(dict(model=a, versus=b, mean_qlike_diff=round(md, 4), dm=round(st, 2), p=round(p, 4)))
out = pd.DataFrame(rows)
out.insert(0, "n", len(w))
out.to_csv(config.RESULTS / "dm_extra.csv", index=False)
print(out.to_string())

# ---- robustness: drop the single 2025-10-10 crash day (pooled RV ~20x its median), post-hoc reading --------------
CRASH = "2025-10-10"
dq = q.groupby(level="date").mean()
keep = dq.index != CRASH
rows = []
for m in ["HAR", "log-HAR", "GARCH(1,1)", "LightGBM", "Kronos-mini", "Kronos-small", "Student", "Combo: log-HAR x Kronos-small"]:
    r = dict(model=m, qlike_all=round(dq[m].mean(), 4), qlike_excl_crash=round(dq.loc[keep, m].mean(), 4))
    for ref in ("HAR", "log-HAR"):
        if m == ref:
            r[f"dm_vs_{ref}"], r[f"p_vs_{ref}"] = np.nan, np.nan
            continue
        d = (dq.loc[keep, m] - dq.loc[keep, ref]).to_numpy()
        st, p, _ = diebold_mariano(d, np.zeros_like(d), 7)
        r[f"dm_vs_{ref}"], r[f"p_vs_{ref}"] = round(st, 2), round(p, 4)
    rows.append(r)
rob = pd.DataFrame(rows)
rob.insert(1, "days_excl", int(keep.sum()))
rob.to_csv(config.RESULTS / "robustness_excl_crash.csv", index=False)
print(rob.to_string())
