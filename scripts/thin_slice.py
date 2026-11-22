"""Thin slice: BTC, July 2024 (validation month), HAR vs Kronos-mini, end to end."""
from datetime import date, timedelta
import numpy as np, pandas as pd
from kvh import config
from kvh.data import load_daily
from kvh.baselines import har_forecast
from kvh.kronos_run import load_forecasts
from kvh.pipeline import kronos_raw
from kvh.metrics import qlike, mz_r2

d = load_daily("BTCUSDT")
days = [date(2024, 7, 1) + timedelta(days=k) for k in range(31)]
har = har_forecast(d, days, 730)
k = load_forecasts("mini", "T09", ["BTCUSDT"], date(2024, 7, 1), date(2024, 7, 31)).set_index("date")
rv = d["rv"].reindex(days)
out = pd.DataFrame({"rv": rv, "har": har})
for v in ("cc_act", "cc_rec", "park"):
    out[v] = kronos_raw(k, v, 16)
print((out * 1e4).describe().T[["mean", "50%", "min", "max"]].round(2))
for c in ["har", "cc_act", "cc_rec", "park"]:
    print(c, "QLIKE raw %.3f" % qlike(out.rv, out[c]).mean(), "QLIKE scaled %.3f" % qlike(out.rv, out[c] * (out.rv / out[c]).mean()).mean(), "MZ %.2f" % mz_r2(out.rv, out[c]))
