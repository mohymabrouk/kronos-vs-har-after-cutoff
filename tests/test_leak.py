"""Test 4: changing any data at or after the forecast time (00:00 UTC of day d) leaves every input and forecast
for day d unchanged: the Kronos context window and its sampled forecast, HAR / log-HAR, EWMA, GARCH, LightGBM,
and the student's features and forecast. Changing data before the forecast time does change them (sanity)."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import torch

from kvh import baselines, student
from kvh.kronos_run import context_for

torch.set_num_threads(2)
D = date(2024, 3, 15)
T0 = pd.Timestamp(D, tz="UTC")


def toy_hourly(n_days=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days * 24, freq="1h", tz="UTC")
    r = rng.normal(0, 0.01, len(idx))
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[100, c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, len(idx))))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.003, len(idx))))
    v = rng.lognormal(3, 0.5, len(idx))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v, "amount": v * c}, index=idx)


def toy_daily(n=900, seed=0):
    rng = np.random.default_rng(seed)
    idx = [date(2022, 1, 1) + timedelta(days=i) for i in range(n)]
    lrv = np.zeros(n)
    for t in range(1, n):
        lrv[t] = 0.9 * lrv[t - 1] + rng.normal(0, 0.4)
    rv = 1e-3 * np.exp(lrv)
    ret = rng.normal(0, 1, n) * np.sqrt(rv)
    return pd.DataFrame({"rv": rv, "ret": ret, "pk": rv * np.exp(rng.normal(0, 0.3, n))}, index=pd.Index(idx, name="date"))


def perturb_after(df, t, factor=3.0):
    df = df.copy()
    mask = df.index >= t
    num = df.select_dtypes("number").columns
    df.loc[mask, num] = df.loc[mask, num] * factor
    return df


def test_kronos_input_and_forecast_no_leak():
    from kvh.kronos_fast import FastKronos

    h = toy_hourly()
    a = context_for(h, D, 96)
    b = context_for(perturb_after(h, T0), D, 96)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)
    # the last candle in the window opened at 23:00 the day before, i.e. it closed exactly at the forecast time
    assert h.loc[:T0 - pd.Timedelta(hours=1)].iloc[-1]["close"] == a[0][-1, 3]
    fk = FastKronos("mini")
    pa = fk.sample_paths(a[0][None], a[1][None], a[2][None], 24, 4, 0.9, 0.9, seed=7)
    pb = fk.sample_paths(b[0][None], b[1][None], b[2][None], 24, 4, 0.9, 0.9, seed=7)
    np.testing.assert_array_equal(pa, pb)
    # sanity: a change one hour before the forecast time does change the input
    c = context_for(perturb_after(h, T0 - pd.Timedelta(hours=1)), D, 96)
    assert not np.array_equal(a[0], c[0])


@pytest.mark.parametrize("log_form", [False, True])
def test_har_no_leak(log_form):
    d = toy_daily()
    day = d.index[700]
    f1 = baselines.har_forecast(d, [day], 365, log_form=log_form, start=d.index[0])
    f2 = baselines.har_forecast(perturb_after(d, day), [day], 365, log_form=log_form, start=d.index[0])
    assert f1.iloc[0] == f2.iloc[0]
    F1, F2 = baselines.har_features(d["rv"]), baselines.har_features(perturb_after(d, day)["rv"])
    pd.testing.assert_series_equal(F1.loc[day], F2.loc[day])
    f3 = baselines.har_forecast(perturb_after(d, d.index[699]), [day], 365, log_form=log_form, start=d.index[0])
    assert f3.iloc[0] != f1.iloc[0]


def test_ewma_garch_lgbm_no_leak():
    d = toy_daily()
    day = d.index[850]  # mid-month day
    p = perturb_after(d, day)
    assert baselines.ewma_forecast(d["rv"]).loc[day] == baselines.ewma_forecast(p["rv"]).loc[day]
    g1 = baselines.garch_forecast(d, [day])
    g2 = baselines.garch_forecast(p, [day])
    assert g1.iloc[0] == pytest.approx(g2.iloc[0], rel=1e-12)
    l1 = baselines.lgbm_forecast({"A": d, "B": toy_daily(seed=1)}, [day])
    l2 = baselines.lgbm_forecast({"A": p, "B": toy_daily(seed=1)}, [day])
    assert l1["A"].iloc[0] == pytest.approx(l2["A"].iloc[0], rel=1e-12)


def test_student_no_leak():
    h = toy_hourly()
    dd = toy_daily(n=200)
    dd.index = pd.Index([date(2024, 1, 1) + timedelta(days=i) for i in range(200)], name="date")
    f1 = student.features("X", [D], daily=dd, hourly=h)
    f2 = student.features("X", [D], daily=perturb_after(dd, D), hourly=perturb_after(h, T0))
    pd.testing.assert_frame_equal(f1, f2)
    assert len(f1) == 1
    m = student.MLP(f1.shape[1], 16).eval()
    with torch.no_grad():
        y1 = m(torch.tensor(f1.to_numpy(), dtype=torch.float32))
        y2 = m(torch.tensor(f2.to_numpy(), dtype=torch.float32))
    assert torch.equal(y1, y2)
    f3 = student.features("X", [D], daily=dd, hourly=perturb_after(h, T0 - pd.Timedelta(hours=1)))
    assert not f1.equals(f3)
