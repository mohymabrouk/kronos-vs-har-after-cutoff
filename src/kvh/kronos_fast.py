"""Kronos inference with a key/value cache, returning every sampled path (not their average).

The vendored `auto_regressive_inference` (third_party/kronos) re-runs the whole transformer over the full window
at each of the 24 generation steps, and runs it once per sample, then averages the decoded paths. For a variance
forecast we need each path, and on 2 CPU threads we need speed. This module re-implements the same computation:

* the context is tokenised once (the tokenizer encoder is deterministic);
* the Kronos transformer is run once over the context (prefill), its keys/values are cached and copied to the
  N sample paths, and each new token costs one single-position pass (exact: causal attention + RoPE with
  absolute positions 0..L+23, and L + 24 <= max_context so the original never slides its window);
* the fine (s2) token is drawn from the dependency-aware layer exactly as in `Kronos.decode_s2`;
* the tokenizer decoder (also causal) is prefilled on the context and extended by the 24 generated tokens.

`tests/test_kronos_wrapper.py` checks logits and decoded values against the vendored original.
"""

import os
import sys
import zlib

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from kvh import config

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_TP = str(config.ROOT / "third_party")
if _TP not in sys.path:
    sys.path.insert(0, _TP)
from kronos import Kronos, KronosTokenizer  # noqa: E402
from kronos.kronos import top_k_top_p_filtering  # noqa: E402

FEATS = ["open", "high", "low", "close", "volume", "amount"]


def time_features(ts: pd.DatetimeIndex) -> np.ndarray:
    """minute, hour, weekday, day, month: same as kronos.calc_time_stamps."""
    ts = pd.DatetimeIndex(ts)
    return np.stack([ts.minute, ts.hour, ts.day_of_week, ts.day, ts.month], axis=1).astype(np.float32)


def _rope_tables(head_dim: int, n: int):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(n).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rot_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class CachedStack:
    """A stack of vendored TransformerBlocks (causal self-attention with RoPE) with a key/value cache."""

    def __init__(self, layers, max_len: int):
        self.layers = layers
        att = layers[0].self_attn
        self.h, self.hd = att.n_heads, att.head_dim
        self.cos, self.sin = _rope_tables(self.hd, max_len)
        self.max_len = max_len
        self.k = self.v = None
        self.pos = 0

    def expand(self, n: int) -> None:
        """Repeat every cached sequence n times (sequence i -> rows i*n .. i*n+n-1)."""
        self.k = [k.repeat_interleave(n, dim=0) for k in self.k]
        self.v = [v.repeat_interleave(n, dim=0) for v in self.v]

    def extend(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d] at positions pos..pos+T-1. Returns the block output [B, T, d]."""
        b, t, d = x.shape
        p0, p1 = self.pos, self.pos + t
        assert p1 <= self.max_len, "sequence longer than the cache"
        cos, sin = self.cos[p0:p1][None, None], self.sin[p0:p1][None, None]
        if self.k is None:
            self.k = [None] * len(self.layers)
            self.v = [None] * len(self.layers)
        if t == 1 or p0 == 0:
            mask, causal = None, (t > 1)
        else:
            qpos = torch.arange(p0, p1)[:, None]
            kpos = torch.arange(0, p1)[None, :]
            mask, causal = (kpos <= qpos), False
        for i, layer in enumerate(self.layers):
            att = layer.self_attn
            hx = layer.norm1(x)
            q = att.q_proj(hx).view(b, t, self.h, self.hd).transpose(1, 2)
            k = att.k_proj(hx).view(b, t, self.h, self.hd).transpose(1, 2)
            v = att.v_proj(hx).view(b, t, self.h, self.hd).transpose(1, 2)
            q = q * cos + _rot_half(q) * sin
            k = k * cos + _rot_half(k) * sin
            if self.k[i] is None:
                self.k[i], self.v[i] = k, v
            else:
                self.k[i] = torch.cat([self.k[i], k], dim=2)
                self.v[i] = torch.cat([self.v[i], v], dim=2)
            o = F.scaled_dot_product_attention(q, self.k[i], self.v[i], attn_mask=mask, is_causal=causal)
            o = o.transpose(1, 2).contiguous().view(b, t, d)
            x = x + att.out_proj(o)
            x = x + layer.ffn(layer.norm2(x))
        self.pos = p1
        return x


def _sample(logits: torch.Tensor, T: float, top_p: float, gen: torch.Generator) -> torch.Tensor:
    logits = logits / T
    if top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=0, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=gen)


class FastKronos:
    def __init__(self, name: str):
        repo, rev, tok_repo, tok_rev, max_ctx = config.KRONOS_MODELS[name]
        cache_dir = str(config.HF_HOME)
        self.name = name
        self.tokenizer = KronosTokenizer.from_pretrained(tok_repo, revision=tok_rev, cache_dir=cache_dir).eval()
        self.model = Kronos.from_pretrained(repo, revision=rev, cache_dir=cache_dir).eval()
        self.max_context = max_ctx
        self.clip = config.CLIP

    def n_params(self) -> dict:
        return {"model": sum(p.numel() for p in self.model.parameters()),
                "tokenizer": sum(p.numel() for p in self.tokenizer.parameters())}

    # ---- normalisation, identical to KronosPredictor.predict -------------------------------------------------
    def normalise(self, x: np.ndarray):
        mean, std = x.mean(axis=0), x.std(axis=0)
        z = np.clip((x - mean) / (std + 1e-5), -self.clip, self.clip)
        return z.astype(np.float32), mean, std

    @torch.no_grad()
    def sample_paths(self, ctx: np.ndarray, ctx_stamp: np.ndarray, fut_stamp: np.ndarray, pred_len: int,
                     n_samples: int, T: float, top_p: float, seed: int, return_context_recon: bool = False):
        """ctx: [M, L, 6] raw OHLCVA; stamps [M, L, 5] / [M, pred_len, 5].

        Returns paths [M, n_samples, pred_len, 6] in raw units (and optionally the decoded last context bar
        [M, 6], i.e. the tokenizer's reconstruction of the candle at the forecast time).
        """
        m, L, _ = ctx.shape
        assert L + pred_len <= self.max_context, "context + horizon must fit in the model window"
        zs, means, stds = zip(*(self.normalise(c) for c in ctx))
        z = torch.from_numpy(np.stack(zs))
        means, stds = np.stack(means), np.stack(stds)
        st_x = torch.from_numpy(ctx_stamp.astype(np.float32))
        st_y = torch.from_numpy(fut_stamp.astype(np.float32))
        gen = torch.Generator().manual_seed(int(seed))
        mdl, tok = self.model, self.tokenizer

        s1, s2 = tok.encode(z, half=True)  # [M, L] each
        # Kronos transformer: prefill on the context, then copy to the N paths
        stack = CachedStack(mdl.transformer, L + pred_len)
        x = mdl.embedding([s1, s2]) + mdl.time_emb(st_x)
        h = mdl.norm(stack.extend(x))  # [M, L, d]
        stack.expand(n_samples)
        h_last = h[:, -1:, :].repeat_interleave(n_samples, dim=0)
        ca = mdl.dep_layer.cross_attn
        dk = ca.k_proj(h).view(m, L, ca.n_heads, ca.head_dim).transpose(1, 2).repeat_interleave(n_samples, dim=0)
        dv = ca.v_proj(h).view(m, L, ca.n_heads, ca.head_dim).transpose(1, 2).repeat_interleave(n_samples, dim=0)
        st_y = st_y.repeat_interleave(n_samples, dim=0)
        b = m * n_samples
        gen_s1 = torch.empty(b, pred_len, dtype=torch.long)
        gen_s2 = torch.empty(b, pred_len, dtype=torch.long)
        for i in range(pred_len):
            s1_logits = mdl.head(h_last)[:, -1, :]
            a1 = _sample(s1_logits, T, top_p, gen)  # [b, 1]
            sib = mdl.embedding.emb_s1(a1)  # [b, 1, d]
            q = ca.q_proj(sib).view(b, 1, ca.n_heads, ca.head_dim).transpose(1, 2)  # RoPE at position 0 = identity
            o = F.scaled_dot_product_attention(q, dk, dv)
            o = ca.out_proj(o.transpose(1, 2).contiguous().view(b, 1, -1))
            x2 = mdl.dep_layer.norm(h_last + o)
            a2 = _sample(mdl.head.cond_forward(x2)[:, -1, :], T, top_p, gen)
            gen_s1[:, i], gen_s2[:, i] = a1[:, 0], a2[:, 0]
            if i == pred_len - 1:
                break
            xn = mdl.embedding([a1, a2]) + mdl.time_emb(st_y[:, i:i + 1, :])
            h_new = mdl.norm(stack.extend(xn))
            h_last = h_new
            dk = torch.cat([dk, ca.k_proj(h_new).view(b, 1, ca.n_heads, ca.head_dim).transpose(1, 2)], dim=2)
            dv = torch.cat([dv, ca.v_proj(h_new).view(b, 1, ca.n_heads, ca.head_dim).transpose(1, 2)], dim=2)

        # tokenizer decoder: causal, so prefill on the context once and extend by the generated block
        dec = CachedStack(tok.decoder, L + pred_len)
        zc = tok.post_quant_embed(tok.indices_to_bits([s1, s2], half=True))
        ctx_out = dec.extend(zc)
        recon_last = tok.head(ctx_out[:, -1, :]).numpy()  # [M, 6] normalised
        dec.expand(n_samples)
        zg = tok.post_quant_embed(tok.indices_to_bits([gen_s1, gen_s2], half=True))
        out = tok.head(dec.extend(zg)).numpy().reshape(m, n_samples, pred_len, 6)
        paths = out * (stds[:, None, None, :] + 1e-5) + means[:, None, None, :]
        if return_context_recon:
            return paths, recon_last * (stds + 1e-5) + means
        return paths


def seed_for(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) ^ config.SEED


def path_variances(paths: np.ndarray, anchor_close: np.ndarray) -> dict:
    """Per-path variance of the next 24 hours from sampled hourly candles.

    paths [M, N, 24, 6]; anchor_close [M] (close at the forecast time used as the first price).
    Returns arrays [M, N]:
      cc   = sum of squared hourly close-to-close log returns
      park = Parkinson: sum_h ln(H_h/L_h)^2 / (4 ln 2), with H/L widened to contain open and close
    """
    tiny = 1e-12
    o, h, l, c = (paths[..., k] for k in range(4))
    c = np.maximum(c, tiny)
    prev = np.concatenate([np.broadcast_to(np.maximum(anchor_close, tiny)[:, None, None], c.shape[:2] + (1,)), c[..., :-1]], axis=-1)
    cc = (np.log(c / prev) ** 2).sum(-1)
    hi = np.maximum.reduce([o, h, l, c])
    lo = np.maximum(np.minimum.reduce([o, h, l, c]), tiny)
    park = (np.log(hi / lo) ** 2).sum(-1) / (4 * np.log(2))
    return {"cc": cc, "park": park}
