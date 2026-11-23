"""Is TimesFM-2.5's 2025-10-01 weight file the same model as the 2025-09-02 upload? (PLAN.md 13.1)

The HF repo google/timesfm-2.5-200m-pytorch replaced model.safetensors on 2025-10-01 ("Refreshed model with speed
optimizations"; the card: "we changed the structure of the model to fuse QKV matrices into one"). This script loads
both files from the HF cache (downloading them if needed), concatenates the original query/key/value tensors of each
attention layer into the fused qkv_proj layout, and compares every tensor with the refreshed file.
Writes results/tsfm_timesfm_weights_check.json. Run: uv run python scripts/timesfm_weights_check.py
"""

import json

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from kvh import config

REPO = "google/timesfm-2.5-200m-pytorch"
ORIGINAL = "a6e2c7cdf5f478e552410974e26d6ffc9afc08a6"  # 2025-09-29, same LFS object as the first upload d35e0e45 (2025-09-02)
REFRESHED = config.TSFM_MODELS["timesfm"]["revision"]  # d418f3e8, 2025-10-01


def main() -> None:
    cache = str(config.HF_HOME)
    a = load_file(hf_hub_download(REPO, "model.safetensors", revision=ORIGINAL, cache_dir=cache))
    b = load_file(hf_hub_download(REPO, "model.safetensors", revision=REFRESHED, cache_dir=cache))
    fused, used = {}, set()
    for k in a:
        if ".query." in k:
            base, suf = k.rsplit(".query.", 1)
            parts = [f"{base}.{n}.{suf}" for n in ("query", "key", "value")]
            fused[f"{base}.qkv_proj.{suf}"] = torch.cat([a[p] for p in parts], dim=0)
            used |= set(parts)
    fused.update({k: v for k, v in a.items() if k not in used})
    same_keys = set(fused) == set(b)
    exact = sum(torch.equal(fused[k], b[k]) for k in b) if same_keys else 0
    res = dict(repo=REPO, original_revision=ORIGINAL, refreshed_revision=REFRESHED, tensors_original=len(a),
               tensors_refreshed=len(b), same_names_after_fusing=same_keys, tensors_bit_identical=int(exact),
               max_abs_diff=float(max((fused[k].float() - b[k].float()).abs().max() for k in b)) if same_keys else None)
    (config.RESULTS / "tsfm_timesfm_weights_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
