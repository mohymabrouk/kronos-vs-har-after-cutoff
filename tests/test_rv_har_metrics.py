"""Tests 1-3 and 6: realised variance on a toy series, HAR recovers simulated coefficients, QLIKE/DM/MZ on fixed
inputs, and the protocol dates."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from kvh import config
from kvh.baselines import ewma_forecast, har_features, har_forecast, ols
from kvh.data import daily_from_5m
from kvh.metrics import diebold_mariano, model_confidence_set, mz_r2, newey_west_var, qlike


# ---- 1. realised variance -------------------------------------------------------------------------------------
def test_rv_from_5m_toy():
    # two UTC days of 288 bars; day 1 alternates +-1% moves, day 2 is flat except one 2% jump
    idx = pd.date_range("2024-01-01", periods=576, freq="5min", tz="UTC")
    closes, opens = [], []
    p = 100.0
    for i in range(576):
        o = p
        if i < 288:
            p = p * np.exp(0.01 if i % 2 == 0 else -0.01)
        elif i == 400:
            p = p * np.exp(0.02)
        opens.append(o)
        closes.append(p)
    df = pd.DataFrame({"open": opens, "high": np.maximum(opens, closes), "low": np.minimum(opens, closes), "close": closes}, index=idx)
    d = daily_from_5m(df)
    assert list(d.index) == [date(2024, 1, 1), date(2024, 1, 2)]
    assert d["n5"].tolist() == [288, 288]
    assert d["rv"].iloc[0] == pytest.approx(288 * 0.01 ** 2, rel=1e-9)
    assert d["rv"].iloc[1] == pytest.approx(0.02 ** 2, rel=1e-9)
    # daily close-to-close return of day 2 = 0.02 (day 1 ends where it started)
    assert d["ret"].iloc[1] == pytest.approx(0.02, abs=1e-12)


def test_rv_gap_spans_missing_bar():
    idx = pd.date_range("2024-01-01", periods=288, freq="5min", tz="UTC").delete(10)  # one missing bar
    c = 100 * np.exp(0.001 * np.arange(1, 288))
    o = np.r_[100, c[:-1]]
    d = daily_from_5m(pd.DataFrame({"open": o, "high": c, "low": o, "close": c}, index=idx))
    assert d["n5"].iloc[0] == 287
    assert d["rv"].iloc[0] == pytest.approx(287 * 0.001 ** 2, rel=1e-9)


# ---- 2. HAR recovers known coefficients ------------------------------------------------------------------------
def test_har_recovers_coefficients():
    rng = np.random.default_rng(0)
    b0, bd, bw, bm = 0.05, 0.35, 0.30, 0.20
    n = 6000
    rv = np.full(n, 0.5)
    for t in range(30, n):
        x = b0 + bd * rv[t - 1] + bw * rv[t - 7:t].mean() + bm * rv[t - 30:t].mean()
        rv[t] = max(x + rng.normal(0, 0.05), 0.01)
    s = pd.Series(rv, index=[date(2000, 1, 1) + timedelta(days=i) for i in range(n)])
    F = har_features(s).iloc[100:]
    b = ols(F.to_numpy(), s.iloc[100:].to_numpy())
    np.testing.assert_allclose(b, [b0, bd, bw, bm], atol=0.03)
    # the rolling forecaster uses only days < d: forecast equals the in-sample prediction of a fit on days < d
    daily = pd.DataFrame({"rv": s})
    d = s.index[-1]
    f = har_forecast(daily, [d], window=0, start=s.index[0])
    Fa = har_features(s).dropna()
    b_past = ols(Fa.iloc[:-1].to_numpy(), s.loc[Fa.index[:-1]].to_numpy())
    expected = b_past[0] + Fa.iloc[-1].to_numpy() @ b_past[1:]
    assert f.index[0] == d and f.iloc[0] == pytest.approx(expected, rel=1e-6)


def test_ewma_uses_only_past():
    x = pd.Series([1.0] * 40 + [100.0])
    f = ewma_forecast(x, 0.94)
    assert f.iloc[-1] == pytest.approx(1.0)  # the shock on the last day does not enter its own forecast


# ---- 3. QLIKE, DM, MZ, MCS on fixed inputs ---------------------------------------------------------------------------
def test_qlike_fixed():
    assert qlike(1.0, 1.0) == 0.0
    assert qlike(2.0, 1.0) == pytest.approx(2.0 - np.log(2.0) - 1.0)
    assert qlike(1.0, 2.0) == pytest.approx(0.5 - np.log(0.5) - 1.0)
    # QLIKE penalises under-prediction more than over-prediction by the same factor
    assert qlike(2.0, 1.0) > qlike(1.0, 2.0)


def test_dm_fixed():
    d = np.array([1.0, -1.0, 2.0, 0.0, 1.0, -0.5, 1.5, 0.5])
    # with lag 0 the HAC variance is the plain variance (divisor n)
    assert newey_west_var(d, 0) == pytest.approx(d.var())
    stat, p, mean = diebold_mariano(d, np.zeros_like(d), lag=0)
    assert mean == pytest.approx(d.mean())
    assert stat == pytest.approx(d.mean() / np.sqrt(d.var() / len(d)))
    assert 0 < p < 1
    # Bartlett weights for lag 1: v = g0 + 2 * 0.5 * g1
    dc = d - d.mean()
    g1 = (dc[1:] @ dc[:-1]) / len(d)
    assert newey_west_var(d, 1) == pytest.approx(d.var() + g1)
    # sign convention: model a better (smaller loss) -> negative statistic
    s2, _, _ = diebold_mariano(np.ones(50) * 0.5 + 0.01 * np.sin(np.arange(50)), np.ones(50), lag=3)
    assert s2 < 0


def test_mz_and_mcs():
    rng = np.random.default_rng(1)
    f = np.exp(rng.normal(0, 1, 500))
    assert mz_r2(f, f) == pytest.approx(1.0)
    rv = f * np.exp(rng.normal(0, 0.5, 500))
    assert 0.5 < mz_r2(rv, f) < 0.9
    L = np.column_stack([rng.normal(1.0, 0.2, 400), rng.normal(1.0, 0.2, 400), rng.normal(2.0, 0.2, 400)])
    p = model_confidence_set(L, ["a", "b", "c"], reps=500)
    assert p["c"] < 0.1 and p["a"] >= 0.1 and p["b"] >= 0.1


# ---- 6. protocol dates -----------------------------------------------------------------------------------------
def test_test_period_after_kronos_cutoff():
    assert config.TEST_START > config.KRONOS_DATA_END
    assert config.KRONOS_DATA_END >= config.KRONOS_WEIGHTS_UPLOADED >= config.KRONOS_PAPER_DATA_END
    assert config.VAL_END < config.TEST_START and config.VAL_START > config.DISTILL_TRAIN_START
    assert config.TEST_END <= date(*config.LAST_MONTH, 28) + timedelta(days=4)
