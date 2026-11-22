"""Baseline forecasts of the next-day realised variance: HAR-RV, log-HAR, EWMA, GARCH(1,1), LightGBM.

Convention: a forecast row for day d uses only daily data of days <= d-1 (everything closed by 00:00 UTC of d).
All functions take the per-asset daily frame from data.load_daily (index = date; columns rv, ret, pk, ...).
"""

import logging
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd

from kvh import config

log = logging.getLogger(__name__)


# ---- HAR --------------------------------------------------------------------------------------------------
def har_features(rv: pd.Series, lags=config.HAR_LAGS) -> pd.DataFrame:
    """Row d holds the regressors known at 00:00 of day d: RV_{d-1}, mean RV over d-w..d-1, mean over d-m..d-1."""
    d, w, m = lags
    out = pd.DataFrame(index=rv.index)
    out["rv_d"] = rv.shift(1).rolling(d).mean()
    out["rv_w"] = rv.shift(1).rolling(w).mean()
    out["rv_m"] = rv.shift(1).rolling(m).mean()
    return out


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return b


def har_forecast(daily: pd.DataFrame, days, window: int, log_form: bool = False, start=None) -> pd.Series:
    """Refit HAR by OLS for every forecast day on target days in [d - window, d - 1] (window 0 = expanding from
    TRAIN_START) and predict day d. log_form: regress log RV on log regressors, forecast exp(fit + s^2/2)."""
    rv = daily["rv"]
    F = har_features(rv)
    y = rv.copy()
    if log_form:
        F, y = np.log(F), np.log(y)
    ok = F.notna().all(axis=1) & y.notna()
    F, y = F[ok], y[ok]
    idx = np.array(F.index)
    Xa, ya = F.to_numpy(), y.to_numpy()
    rv_a = rv.reindex(F.index).to_numpy()
    out = {}
    for d in days:
        if d not in F.index:
            continue
        first = start or config.TRAIN_START
        lo = first if window == 0 else max(first, d - timedelta(days=window))
        sel = (idx >= lo) & (idx < d)
        if sel.sum() < 100:
            continue
        b = ols(Xa[sel], ya[sel])
        x = Xa[idx == d][0]
        pred = b[0] + x @ b[1:]
        if log_form:
            resid = ya[sel] - (b[0] + Xa[sel] @ b[1:])
            pred = np.exp(pred + 0.5 * resid.var())
        # insanity filter (Bollerslev, Patton & Quaedvlieg 2016): a forecast outside the range of the RVs in the
        # fitting window is replaced by the window mean (only past data)
        lo_rv, hi_rv = rv_a[sel].min(), rv_a[sel].max()
        if not (lo_rv <= pred <= hi_rv):
            pred = rv_a[sel].mean()
        out[d] = pred
    return pd.Series(out, name="har_log" if log_form else "har")


# ---- EWMA -------------------------------------------------------------------------------------------------
def ewma_forecast(x: pd.Series, lam: float = config.EWMA_LAMBDA) -> pd.Series:
    """sigma2_d = lam * sigma2_{d-1} + (1 - lam) * x_{d-1}, started at the mean of the first 30 observations."""
    v = x.to_numpy()
    s = np.empty_like(v)
    s[0] = np.nanmean(v[:30])
    for t in range(1, len(v)):
        s[t] = lam * s[t - 1] + (1 - lam) * v[t - 1]
    return pd.Series(s, index=x.index)


# ---- GARCH(1,1) ---------------------------------------------------------------------------------------------
def garch_forecast(daily: pd.DataFrame, days) -> pd.Series:
    """GARCH(1,1), normal QMLE with constant mean, on daily log returns (in %). Refit at each month start on the
    last GARCH_WINDOW days before it; inside the month the variance recursion is filtered with the actual returns
    of days <= d-1. Forecast = conditional variance of ret_d (same [00:00, 24:00) window as RV_d)."""
    from arch import arch_model

    r = daily["ret"] * 100.0
    out = {}
    days = sorted(days)
    months = sorted({(d.year, d.month) for d in days})
    for (y, m) in months:
        m0 = date(y, m, 1)
        hist = r[(r.index < m0)].iloc[-config.GARCH_WINDOW:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(hist, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(disp="off")
        mu, om, al, be = (res.params[k] for k in ("mu", "omega", "alpha[1]", "beta[1]"))
        s2 = float(res.conditional_volatility.iloc[-1]) ** 2
        e_prev = float(hist.iloc[-1] - mu)
        cur = hist.index[-1]
        mdays = [d for d in days if (d.year, d.month) == (y, m)]
        seq = r[(r.index > cur) & (r.index <= mdays[-1])]
        # walk forward one day at a time: forecast for day t uses e_{t-1}, s2_{t-1}
        for t, rt in seq.items():
            s2 = om + al * e_prev ** 2 + be * s2
            if t in mdays:
                out[t] = s2 / 1e4
            e_prev = rt - mu
    return pd.Series(out, name="garch")


# ---- LightGBM on HAR features ----------------------------------------------------------------------------------
def lgbm_features(daily: pd.DataFrame) -> pd.DataFrame:
    F = np.log(har_features(daily["rv"]))
    F["ret_1"] = daily["ret"].shift(1)
    F["absret_1"] = daily["ret"].shift(1).abs()
    F["log_pk_1"] = np.log(daily["pk"].shift(1).clip(lower=1e-10))
    F["log_rv_2"] = np.log(daily["rv"].shift(2))
    return F


def lgbm_forecast(dailies: dict, days) -> dict:
    """Pooled LightGBM on log RV, refit at each month start on all target days before it (all assets)."""
    import lightgbm as lgb

    frames = []
    for s, d in dailies.items():
        F = lgbm_features(d)
        F["y"] = np.log(d["rv"])
        F["symbol"] = s
        frames.append(F.dropna())
    panel = pd.concat(frames)
    panel["date"] = panel.index
    feats = [c for c in panel.columns if c not in ("y", "symbol", "date")]
    out = {s: {} for s in dailies}
    for (y, m) in sorted({(d.year, d.month) for d in days}):
        m0 = date(y, m, 1)
        tr = panel[(panel["date"] < m0) & (panel["date"] >= config.TRAIN_START)]
        mdl = lgb.train(config.LGBM_PARAMS, lgb.Dataset(tr[feats].to_numpy(), tr["y"].to_numpy()),
                        num_boost_round=config.LGBM_ROUNDS)
        bias = float(np.mean(np.exp(tr["y"].to_numpy() - mdl.predict(tr[feats].to_numpy()))))  # smearing estimate for exp(log forecast)
        nm = date(y + (m == 12), m % 12 + 1, 1)
        te = panel[(panel["date"] >= m0) & (panel["date"] < nm)]
        te = te[te["date"].isin(set(days))]
        if len(te):
            pred = np.exp(mdl.predict(te[feats].to_numpy())) * bias
            for (s, d), p in zip(zip(te["symbol"], te["date"]), pred):
                out[s][d] = p
    return {s: pd.Series(v, name="lgbm") for s, v in out.items()}


def all_baselines(dailies: dict, days, har_windows: dict) -> pd.DataFrame:
    """Long frame: date, symbol, model, forecast (decimal variance)."""
    import time

    rows = []
    cpu = {}

    def timed(name, fn):
        t0 = time.process_time()
        out = fn()
        cpu[name] = cpu.get(name, 0.0) + time.process_time() - t0
        return out

    lg = timed("LightGBM", lambda: lgbm_forecast(dailies, days))
    for s, d in dailies.items():
        sd = [x for x in days if x in d.index]
        fc = {
            "HAR": timed("HAR", lambda: har_forecast(d, sd, har_windows["HAR"])),
            "log-HAR": timed("log-HAR", lambda: har_forecast(d, sd, har_windows["log-HAR"], log_form=True)),
            "EWMA (RV)": timed("EWMA (RV)", lambda: ewma_forecast(d["rv"]).reindex(sd)),
            "EWMA (r2)": timed("EWMA (r2)", lambda: ewma_forecast(d["ret"] ** 2).reindex(sd)),
            "GARCH(1,1)": timed("GARCH(1,1)", lambda: garch_forecast(d, sd)),
            "LightGBM": lg[s].reindex(sd),
        }
        for name, ser in fc.items():
            ser = ser.dropna()
            rows.append(pd.DataFrame({"date": list(ser.index), "symbol": s, "model": name, "forecast": ser.to_numpy()}))
        log.info("baselines done for %s (%d days)", s, len(sd))
    out = pd.concat(rows, ignore_index=True)
    n = out.groupby("model").size()
    all_baselines.cpu_s_per_forecast = {k: v / max(int(n.get(k, 1)), 1) for k, v in cpu.items()}
    return out
