"""Loss functions and forecast-comparison tests: QLIKE, MSE, Mincer-Zarnowitz R^2, Diebold-Mariano with
Newey-West (HAC) errors, and the Model Confidence Set of Hansen, Lunde & Nason (2011)."""

import numpy as np
from scipy import stats


def qlike(rv, f):
    """Patton (2011) QLIKE in the normalised form rv/f - log(rv/f) - 1 (>= 0, = 0 iff f = rv)."""
    ratio = np.asarray(rv, float) / np.asarray(f, float)
    return ratio - np.log(ratio) - 1.0


def se(rv, f):
    return (np.asarray(rv, float) - np.asarray(f, float)) ** 2


def mz_r2(rv, f) -> float:
    """R^2 of the Mincer-Zarnowitz regression log(rv) = a + b log(f) + e."""
    y, x = np.log(np.asarray(rv, float)), np.log(np.asarray(f, float))
    X = np.column_stack([np.ones_like(x), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    return float(1.0 - resid.var() / y.var())


def newey_west_var(d, lag: int) -> float:
    """Long-run variance of a series (Bartlett kernel)."""
    d = np.asarray(d, float)
    d = d - d.mean()
    n = len(d)
    v = d @ d / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        v += 2.0 * w * (d[k:] @ d[:-k]) / n
    return float(v)


def diebold_mariano(loss_a, loss_b, lag: int = 7):
    """DM test of equal predictive accuracy. d = loss_a - loss_b; negative statistic = model a better.

    Returns (statistic, two-sided p-value, mean differential)."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    lrv = newey_west_var(d, lag)
    stat = d.mean() / np.sqrt(lrv / n)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(stat)))
    return float(stat), float(p), float(d.mean())


def _block_bootstrap_indices(n: int, block: int, reps: int, rng) -> np.ndarray:
    """Circular moving-block bootstrap indices [reps, n]."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(reps, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n]
    return idx % n


def model_confidence_set(losses: np.ndarray, names, alpha: float = 0.10, block: int = 7, reps: int = 2000, seed: int = 0):
    """Hansen, Lunde & Nason (2011) MCS with the T_max statistic (range-free version based on deviations from
    the cross-model average). losses: [T, k]. Returns dict name -> MCS p-value; models with p >= alpha are in."""
    rng = np.random.default_rng(seed)
    L = np.asarray(losses, float)
    n, k = L.shape
    idx = _block_bootstrap_indices(n, block, reps, rng)
    boot_means = L[idx].mean(axis=1)  # [reps, k]
    means = L.mean(axis=0)
    alive = list(range(k))
    pvals = {}
    running = 0.0
    while len(alive) > 1:
        a = np.array(alive)
        dbar = means[a] - means[a].mean()  # d_i. = Lbar_i - average over alive
        bd = boot_means[:, a] - boot_means[:, a].mean(axis=1, keepdims=True)
        var = ((bd - dbar) ** 2).mean(axis=0)
        t = dbar / np.sqrt(var)
        tmax = t.max()
        tboot = ((bd - dbar) / np.sqrt(var)).max(axis=1)
        p = float((tboot >= tmax).mean())
        running = max(running, p)
        worst = a[int(np.argmax(t))]
        pvals[names[worst]] = running
        alive.remove(worst)
    pvals[names[alive[0]]] = 1.0
    return pvals
