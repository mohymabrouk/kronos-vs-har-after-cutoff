import numpy as np, pandas as pd
from kvh import config
from kvh.data import load_daily
from kvh.baselines import har_forecast
from kvh.pipeline import days_between
from kvh.metrics import qlike
days = days_between(config.VAL_START, config.VAL_END)
for w in (365, 730, 0):
    tot = []
    for s in config.SYMBOLS:
        d = load_daily(s)
        f = har_forecast(d, days, w)
        rv = d.rv.reindex(f.index)
        q = qlike(rv, f)
        tot.append(q)
        if w == 730:
            print(s, round(q.mean(), 3), "max", round(q.max(), 1), "ratio rv/f max", round((rv / f).max(), 1), "min f %2", round(f.min() * 1e4, 3), "floor hits", int((f <= d.rv.quantile(0.001) * 1.0000001).sum()))
    print("window", w, np.concatenate(tot).mean().round(4), "median", np.median(np.concatenate(tot)).round(4))
