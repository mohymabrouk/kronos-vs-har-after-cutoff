"""Tests of the TSFM extension (PLAN.md section 13).

7. Scoring starts respect each model's weights date and stated data end, with the Kronos embargo rule; the
   generated test forecasts start there.
8. Input series and the variance transform; the TTM + log-HAR average; no look-ahead in the inputs.
9. The three wrappers (shape, determinism, a sanity forecast, the pinned weight file's SHA-256). These need the
   weights in the local HF cache, so they skip on CI.
"""

import hashlib
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import torch

from kvh import config, tsfm
from test_leak import D, T0, perturb_after, toy_daily, toy_hourly

torch.set_num_threads(2)


# ---- 7. dates ------------------------------------------------------------------------------------------------------
def test_add_months():
    assert tsfm.add_months(date(2025, 7, 1), 1) == date(2025, 8, 1)
    assert tsfm.add_months(date(2025, 1, 31), 1) == date(2025, 2, 28)
    assert tsfm.add_months(date(2024, 12, 15), 1) == date(2025, 1, 15)
    assert tsfm.add_months(date(2025, 10, 30), 1) == date(2025, 11, 30)


def test_scoring_starts_respect_release_and_cutoff():
    # the rule reproduces the original test start from the Kronos cutoff
    assert tsfm.embargo_end(config.KRONOS_DATA_END) == config.TEST_START
    for name, m in config.TSFM_MODELS.items():
        s = tsfm.scoring_start(name)
        assert config.TEST_START <= s <= config.TEST_END
        assert s >= tsfm.add_months(m["weights_date"], config.TSFM_EMBARGO_MONTHS) > m["weights_date"]
        if m["stated_data_end"] is not None:
            assert s >= tsfm.add_months(m["stated_data_end"], config.TSFM_EMBARGO_MONTHS)
        assert len(m["revision"]) == 40 and int(m["revision"], 16) >= 0  # a full commit hash, not a branch name
    assert tsfm.scoring_start("ttm") == date(2025, 8, 1)
    assert tsfm.scoring_start("timesfm") == date(2025, 10, 2)
    assert tsfm.scoring_start("chronos2") == date(2025, 11, 30)


def test_frozen_and_generated_forecasts_respect_scoring_start():
    fz_path = config.RESULTS / "frozen_tsfm.json"
    if not fz_path.exists():
        pytest.skip("results/frozen_tsfm.json not written yet")
    fz = json.loads(fz_path.read_text())
    for name, m in config.TSFM_MODELS.items():
        assert fz["models"][name]["scoring_start"] == str(tsfm.scoring_start(name))
        assert fz["models"][name]["revision"] == m["revision"]
    fc_path = config.RESULTS / "tsfm_forecasts_test.csv"
    if not fc_path.exists():
        pytest.skip("results/tsfm_forecasts_test.csv not written yet")
    fc = pd.read_csv(fc_path, usecols=["date", "model"])
    first = fc.groupby("model")["date"].min()
    for name in config.TSFM_MODELS:
        for spec in config.TSFM_SPECS:
            assert first[tsfm.label(name, spec)] == str(tsfm.scoring_start(name))


# ---- 8. transforms, average, no look-ahead -----------------------------------------------------------------------
def test_hourly_vol_series():
    x = np.array([[100.0, 102.0, 99.0, 101.0, 1, 1],
                  [101.0, 101.5, 100.0, 100.5, 1, 1],
                  [100.5, 100.5, 100.5, 100.5, 0, 0]])  # a flat-filled hour
    cc = tsfm.hourly_vol(x, "cc")
    np.testing.assert_allclose(cc, [abs(np.log(101 / 100)), abs(np.log(100.5 / 101)), 0.0])
    pk = tsfm.hourly_vol(x, "park")
    np.testing.assert_allclose(pk, [np.log(102 / 99), np.log(101.5 / 100), 0.0] / (2 * np.sqrt(np.log(2))))
    # squared, the park series is the hourly Parkinson variance used for Kronos paths
    assert pk[0] ** 2 == pytest.approx(np.log(102 / 99) ** 2 / (4 * np.log(2)))


def test_raw_variance_transform():
    h = np.full((2, 24), 0.01)
    h[1, :12] = -0.02  # negative point forecasts of a volatility are clipped at 0
    np.testing.assert_allclose(tsfm.raw_variance(h, "H"), [24e-4, 12e-4])
    np.testing.assert_allclose(tsfm.raw_variance(np.array([[0.02], [-0.01]]), "D"), [4e-4, 0.0])


def test_daily_sigma_context():
    d = toy_daily(n=1200)
    day = d.index[1100]
    s = tsfm.daily_sigma(d, day, 1000)
    np.testing.assert_allclose(s, np.sqrt(d["rv"].iloc[100:1100].to_numpy()))
    assert tsfm.daily_sigma(d, d.index[500], 1000) is None


def test_vol_average():
    a, b = np.array([1.0, 2.0, 0.5]), np.array([4.0, 2.0, 8.0])
    v = tsfm.vol_average(a, b)
    np.testing.assert_allclose(v[:2], [2.25, 2.0])
    assert np.all(np.sqrt(a * b) <= v + 1e-15) and np.all(v <= (a + b) / 2 + 1e-15)  # between geometric and arithmetic


def test_tsfm_inputs_no_leak():
    h = toy_hourly()
    before = h.copy()  # sanity: a change in the last candle that closed at the forecast time does change the input
    before.loc[T0 - pd.Timedelta(hours=1), ["high", "close"]] *= 1.01
    for name, variant in (("ttm", "cc"), ("timesfm", "park"), ("chronos2", "cc"), ("chronos2", "park")):
        a = tsfm.series_for(name, "H", variant, D, hourly=h)
        assert len(a) == config.TSFM_MODELS[name]["context_h"]
        np.testing.assert_array_equal(a, tsfm.series_for(name, "H", variant, D, hourly=perturb_after(h, T0)))
        assert not np.array_equal(a, tsfm.series_for(name, "H", variant, D, hourly=before))
    d = toy_daily(n=1200)
    day = d.index[1100]
    for name in config.TSFM_MODELS:
        a = tsfm.series_for(name, "D", "sigma", day, daily=d)
        assert len(a) == config.TSFM_MODELS[name]["context_d"]
        np.testing.assert_array_equal(a, tsfm.series_for(name, "D", "sigma", day, daily=perturb_after(d, day)))
        assert not np.array_equal(a, tsfm.series_for(name, "D", "sigma", day, daily=perturb_after(d, day - timedelta(days=1))))


# ---- 9. wrappers (need the pinned weights locally; skip on CI) ------------------------------------------------------
def _needs(name):
    if tsfm.cached_weights(name) is None:
        pytest.skip(f"pinned {name} weights not in the local HF cache")


@pytest.mark.parametrize("name", list(config.TSFM_MODELS))
def test_pinned_weights_sha256(name):
    _needs(name)
    h = hashlib.sha256()
    with open(tsfm.cached_weights(name), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    assert h.hexdigest() == config.TSFM_MODELS[name]["weights_sha256"]


@pytest.mark.parametrize("name", list(config.TSFM_MODELS))
def test_wrapper_forecasts(name):
    _needs(name)
    w = tsfm.WRAPPERS[name]()
    m = config.TSFM_MODELS[name]
    rng = np.random.default_rng(0)
    for spec, L, horizon in (("H", m["context_h"], 24), ("D", m["context_d"], 1)):
        t = np.arange(L + 24)
        clean = 0.01 * (1.5 + np.sin(2 * np.pi * t / 24))  # positive, period 24
        x = [clean[:L], clean[:L] * (1 + 0.05 * rng.standard_normal(L))]
        a = w.predict(x, horizon, spec)
        assert a.shape == (2, horizon) and np.isfinite(a).all()
        np.testing.assert_array_equal(a, w.predict(x, horizon, spec))  # deterministic
        if spec == "H":  # a clean periodic series is continued closely
            err = np.abs(a[0] - clean[L:L + 24]).max()
            assert err < 0.2 * 0.01, f"{name}: max error {err:.2e} on a sine with amplitude 1e-2"
