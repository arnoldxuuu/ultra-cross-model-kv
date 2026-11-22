"""Mapped generation in plain HF (no vLLM): a genevals GenFn that runs anywhere torch runs (CUDA, ROCm, CPU).

For quality cells on the free AMD MI300X/MI350X boxes, where the vLLM connector (CUDA FlashAttention layout) cannot
run. Per prompt: if shorter than min_len -> native greedy; else the first len-tail tokens get their KV from
kv_fn (source prefill + maps + restore + optional sink template), the target prefills the tail natively on top of that
cache, then decodes greedily. Batch 1 (exactness over speed; quality cells are n <= 500).
"""
from __future__ import annotations

import torch

from . import core


def source_kv_fn(S, T, maps_k, maps_v, mode="none", kw=None, sink=None, feat="kv"):
    """kv_fn(ids [1,m]) -> pre-RoPE target K, V [Lt,1,H,m,D] from the source model + ridge maps."""
    def fn(ids):
        m = ids.shape[1]
        cs, ss = core.rope_cos_sin(S, m, ids.device)
        Ks, Vs, _ = core.capture_kv(S, ids, cs, ss)
        if feat == "kvcat":
            Ks = Vs = torch.cat([Ks, Vs], dim=2)
        H = maps_k[0]["H"]
        K = core.restore_scale(core.apply_maps(maps_k, Ks), maps_k, mode, H, kw)
        V = core.apply_maps(maps_v, Vs)
        if sink is not None:  # calibration-mean K/V at the first A positions
            A = min(sink["K"].shape[2], m)
            K[:, :, :, :A] = sink["K"][:, None, :, :A].to(K)
            V[:, :, :, :A] = sink["V"][:, None, :, :A].to(V)
        return K, V
    return fn


@torch.no_grad()
def greedy_from_cache(T, cache, last_logits, max_new, stop_ids=()):
    out, tok = [], int(last_logits[0, -1].argmax())
    for _ in range(max_new):
        out.append(tok)
        if tok in stop_ids:
            break
        o = T(input_ids=torch.tensor([[tok]], device=last_logits.device), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        tok = int(o.logits[0, -1].argmax())
    return out


def make_hf_gen(T, kv_fn, tail=16, min_len=32, stop_ids=(), device=None):
    """GenFn(prompts: list[list[int]], max_new) -> list[list[int]] for xkv.genevals."""
    dev = device or next(T.parameters()).device

    @torch.no_grad()
    def gen(prompts, max_new):
        outs = []
        for p in prompts:
            ids = torch.tensor([list(p)], device=dev)
            n = ids.shape[1]
            m = n - tail if n >= min_len else 0
            if m <= 0:
                o = T(input_ids=ids, use_cache=True)
                outs.append(greedy_from_cache(T, o.past_key_values, o.logits, max_new, stop_ids))
                continue
            K, V = kv_fn(ids[:, :m])
            ct, st = core.rope_cos_sin(T, n, dev)
            cache = core.build_cache(K, V, ct, st)
            o = T(input_ids=ids[:, m:], past_key_values=cache, use_cache=True)
            outs.append(greedy_from_cache(T, o.past_key_values, o.logits, max_new, stop_ids))
        return outs
    return gen
