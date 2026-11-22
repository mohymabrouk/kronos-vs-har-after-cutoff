"""Test 5: the Kronos wrapper returns the right shape, is deterministic under a fixed seed, and matches the
vendored original implementation (greedy decoding, where both are deterministic)."""

import numpy as np
import pandas as pd
import pytest
import torch

from kvh import config
from kvh.kronos_fast import FastKronos, path_variances, time_features

torch.set_num_threads(2)


def toy_candles(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, n)
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[100, close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    vol = rng.lognormal(3, 0.5, n)
    x = np.stack([open_, high, low, close, vol, vol * close], axis=1)
    ts = pd.date_range("2024-01-01", periods=n + 24, freq="1h", tz="UTC")
    return x, ts[:n], ts[n:]


@pytest.fixture(scope="module")
def mini():
    return FastKronos("mini")


def test_shape_and_determinism(mini):
    L = 96
    x, tx, ty = toy_candles(L)
    ctx, sx, sy = x[None], time_features(tx)[None], time_features(ty)[None]
    ctx2 = np.concatenate([ctx, ctx * 1.1])
    sx2, sy2 = np.concatenate([sx, sx]), np.concatenate([sy, sy])
    a = mini.sample_paths(ctx2, sx2, sy2, 24, n_samples=5, T=0.9, top_p=0.9, seed=123)
    b = mini.sample_paths(ctx2, sx2, sy2, 24, n_samples=5, T=0.9, top_p=0.9, seed=123)
    c = mini.sample_paths(ctx2, sx2, sy2, 24, n_samples=5, T=0.9, top_p=0.9, seed=124)
    assert a.shape == (2, 5, 24, 6)
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
    v = path_variances(a, ctx2[:, -1, 3])
    assert v["cc"].shape == (2, 5) and (v["cc"] > 0).all() and (v["park"] > 0).all()


@pytest.mark.parametrize("name", ["mini", "small"])
def test_matches_original_greedy(name):
    """With a near-zero temperature both samplers are greedy; decoded paths must agree with the original."""
    import sys
    sys.path.insert(0, str(config.ROOT / "third_party"))
    from kronos.kronos import auto_regressive_inference

    fk = FastKronos(name)
    L = 64
    x, tx, ty = toy_candles(L, seed=1)
    z, mean, std = fk.normalise(x)
    ours = fk.sample_paths(x[None], time_features(tx)[None], time_features(ty)[None], 24, n_samples=2, T=1e-4, top_p=0.9, seed=0)
    orig = auto_regressive_inference(fk.tokenizer, fk.model, torch.from_numpy(z[None]), torch.from_numpy(time_features(tx)[None]),
                                     torch.from_numpy(time_features(ty)[None]), fk.max_context, 24, clip=5, T=1e-4, top_k=0,
                                     top_p=0.9, sample_count=2)
    orig = orig[:, -24:, :] * (std + 1e-5) + mean
    np.testing.assert_allclose(ours[0, 0], orig[0], rtol=2e-3, atol=1e-3 * np.abs(orig[0]).max())
    np.testing.assert_allclose(ours[0, 1], orig[0], rtol=2e-3, atol=1e-3 * np.abs(orig[0]).max())
