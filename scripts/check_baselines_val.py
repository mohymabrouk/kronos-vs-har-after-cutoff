"""Sanity check of the baselines on the validation period (no test data)."""
import time
import numpy as np, pandas as pd
from kvh import config
from kvh.data import load_daily
from kvh.baselines import all_baselines
from kvh.pipeline import days_between, attach_rv
from kvh.metrics import qlike, mz_r2

dailies = {s: load_daily(s) for s in config.SYMBOLS}
days = days_between(config.VAL_START, config.VAL_END)
t = time.time()
b = all_baselines(dailies, days, {"HAR": 730, "log-HAR": 730})
print("seconds", round(time.time() - t, 1))
b = attach_rv(b, dailies)
for m, g in b.groupby("model"):
    print(f"{m:12s} n={len(g)} QLIKE={qlike(g.rv, g.forecast).mean():.4f} MSE={((g.rv-g.forecast)**2).mean()*1e8:.1f} MZ={mz_r2(g.rv, g.forecast):.3f} mean_f/mean_rv={g.forecast.mean()/g.rv.mean():.3f}")
