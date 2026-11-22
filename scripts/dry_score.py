"""Dry run of the scoring code on validation-period baseline forecasts (code check only, no test data)."""
import numpy as np, pandas as pd
from kvh import config
from kvh.data import load_daily
from kvh.baselines import all_baselines
from kvh.pipeline import days_between, attach_rv, score
from kvh.metrics import model_confidence_set
dailies = {s: load_daily(s) for s in config.SYMBOLS}
b = attach_rv(all_baselines(dailies, days_between(config.VAL_START, config.VAL_END), {"HAR": 730, "log-HAR": 0}), dailies)
models = ["HAR", "log-HAR", "EWMA (RV)", "EWMA (r2)", "GARCH(1,1)", "LightGBM"]
perf, ql = score(b, models)
print(perf.round(4).to_string())
print(all_baselines.cpu_s_per_forecast)
dq = ql.groupby(level="date").mean()
print(model_confidence_set(dq.to_numpy(), models, reps=500))
