"""Tensor-parallel source prefill inside vLLM's TP workers (kvx connector, TP > 1).

Under TP=2 every rank ran the WHOLE source prefill (Qwen2.5-3B, 3.1 s at 128K): the bottleneck of the 72B 128K run
(5.05x vs vLLM TP2). Here each rank keeps 1/world of every decoder layer of a Llama-style HF model:
  q/k/v_proj rows of its heads (+ biases), o_proj columns (+ all-reduce), MLP gate/up rows, down_proj columns
  (+ all-reduce). HF attention infers the head count from the projection width (view(..., -1, head_dim)) and the GQA
  ratio is unchanged, so the forward runs as is. Each rank's KV cache then holds only its KV heads; gather_heads()
  all-gathers them so every rank's mapper sees all source heads (it maps only its own target heads).
Requires num_heads and num_key_value_heads divisible by world.
"""
from __future__ import annotations

import torch
import torch.distributed as dist


def _rows(lin, idx):
    lin.weight.data = lin.weight.data[idx].contiguous()
    if lin.bias is not None:
        lin.bias.data = lin.bias.data[idx].contiguous()
    lin.out_features = len(idx)


def _cols(lin, idx):
    lin.weight.data = lin.weight.data[:, idx].contiguous()
    lin.in_features = len(idx)


def _allreduce_hook(group, keep_bias):
    def hook(mod, args, out):
        if keep_bias is not None:  # bias added once, after the reduction
            out = out - keep_bias
        dist.all_reduce(out, group=group)
        if keep_bias is not None:
            out = out + keep_bias
        return out
    return hook


def shard_source(model, rank: int, world: int, group=None):
    """Shard a Llama-style HF causal LM (model.model.layers[*].self_attn / .mlp) in place across `world` ranks."""
    if world <= 1:
        return model
    cfg = model.config
    H, Hk = cfg.num_attention_heads, cfg.num_key_value_heads
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    assert H % world == 0 and Hk % world == 0, (H, Hk, world)
    hq, hk = H // world, Hk // world
    q_idx = torch.arange(rank * hq * D, (rank + 1) * hq * D)
    k_idx = torch.arange(rank * hk * D, (rank + 1) * hk * D)
    for layer in model.model.layers:
        a, m = layer.self_attn, layer.mlp
        _rows(a.q_proj, q_idx)
        _rows(a.k_proj, k_idx)
        _rows(a.v_proj, k_idx)
        _cols(a.o_proj, q_idx)
        ob = a.o_proj.bias.data.clone() if a.o_proj.bias is not None else None
        a.o_proj.register_forward_hook(_allreduce_hook(group, ob))
        inter = m.gate_proj.out_features
        assert inter % world == 0, inter
        f = inter // world
        f_idx = torch.arange(rank * f, (rank + 1) * f)
        _rows(m.gate_proj, f_idx)
        _rows(m.up_proj, f_idx)
        _cols(m.down_proj, f_idx)
        db = m.down_proj.bias.data.clone() if m.down_proj.bias is not None else None
        m.down_proj.register_forward_hook(_allreduce_hook(group, db))
        for attr in ("num_heads", "num_key_value_heads"):  # older HF modules keep counts as attributes
            if hasattr(a, attr):
                setattr(a, attr, getattr(a, attr) // world)
    model._kvx_tp = (rank, world)
    return model


def gather_heads(x, world: int, group=None, dim: int = 1):
    """All-gather a per-rank head shard (e.g. K [B, H/world, T, D]) into the full tensor along `dim` (rank order)."""
    if world <= 1:
        return x
    parts = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(parts, x.contiguous(), group=group)
    return torch.cat(parts, dim=dim)
