"""TTM under the torch version granite-tsfm declares (<2.12) against the project environment (torch 2.15).

granite-tsfm pins torch<2.12; the project lifts that bound (pyproject [tool.uv] override-dependencies) so the
environment of the original results is unchanged. This script checks that TTM gives the same forecasts:

  1. project env:  uv run python scripts/ttm_torch_check.py dump <file.npz>
     (120 random validation asset-days per input series: the TTM input and the stored TTM forecast)
  2. separate env with torch 2.11 + granite-tsfm 0.3.12 + numpy (no kvh needed):
     python scripts/ttm_torch_check.py check <file.npz>  ->  JSON with the relative differences
     The JSON of the run is kept in results/tsfm_ttm_torch_check.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO, REV = "ibm-granite/granite-timeseries-ttm-r2", "a23813d5e5859987e522d084f7d1d1ef9f134c3e"
SERIES = (("H", "cc", "h"), ("H", "park", "h"), ("D", "sigma", "d"))


def dump(path: str) -> None:
    from kvh import config, tsfm
    from kvh.data import load_daily, load_hourly

    rng = np.random.default_rng(0)
    out, cache = {}, {}
    for spec, var, _ in SERIES:
        df = tsfm.load("ttm", spec, var, config.VAL_START, config.VAL_END)
        X, Y = [], []
        for i in rng.choice(len(df), 120, replace=False):
            r = df.iloc[i]
            if r["symbol"] not in cache:
                cache[r["symbol"]] = (load_hourly(r["symbol"]), load_daily(r["symbol"]))
            h, d = cache[r["symbol"]]
            X.append(tsfm.series_for("ttm", spec, var, r["date"], h, d))
            Y.append(np.asarray(r["point"]))
        out[f"{spec}_{var}_x"], out[f"{spec}_{var}_y"] = np.stack(X), np.stack(Y)
    np.savez(path, **out)


def check(path: str) -> None:
    import torch
    from tsfm_public import TinyTimeMixerForPrediction
    from tsfm_public.toolkit.time_series_preprocessor import DEFAULT_FREQUENCY_MAPPING

    torch.set_num_threads(1)
    m = TinyTimeMixerForPrediction.from_pretrained(REPO, revision=REV, cache_dir=str(Path.home() / ".cache" / "quant-research" / "p3" / "hf")).eval()
    io = np.load(path)
    res = {"torch": torch.__version__}
    for spec, var, freq in SERIES:  # the same computation as kvh.tsfm.TTM.predict
        x, y = io[f"{spec}_{var}_x"].astype(np.float64), io[f"{spec}_{var}_y"]
        mu, sd = x.mean(1, keepdims=True), x.std(1, keepdims=True)
        sd = np.where(sd > 0, sd, 1.0)
        z = torch.tensor(((x - mu) / sd)[..., None], dtype=torch.float32)
        tok = torch.full((len(x),), DEFAULT_FREQUENCY_MAPPING[freq], dtype=torch.long)
        with torch.no_grad():
            o = m(past_values=z, freq_token=tok, return_loss=False).prediction_outputs[..., 0].numpy()
        o = o[:, :y.shape[1]].astype(np.float64) * sd + mu
        rel = np.abs(o - y) / np.abs(y)
        res[f"{spec}_{var}"] = dict(n=int(y.size), max_rel_diff=float(rel.max()), median_rel_diff=float(np.median(rel)))
    print(json.dumps(res))


if __name__ == "__main__":
    {"dump": dump, "check": check}[sys.argv[1]](sys.argv[2])
