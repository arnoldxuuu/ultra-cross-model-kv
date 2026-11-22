"""Source-model prefill through vLLM's own flash-attention varlen kernel (FA3 on Hopper, FA4 on Blackwell), registered as a
transformers attention implementation. Single sequence, causal, no padding (the connector's source prefill).
HF SDPA ran the 1.7B's 128K prefill at ~50% of H100 peak even with cuDNN; this is the kernel native vLLM uses."""
import torch

NAME = "vllm_fa"
_REGISTERED = False


def _fa_func():
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    try:
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
    except Exception:
        get_flash_attn_version = None
    return flash_attn_varlen_func, get_flash_attn_version


def vllm_fa_attention(module, query, key, value, attention_mask=None, scaling=None, dropout=0.0, **kwargs):
    """query [B, Hq, T, D], key/value [B, Hk, S, D] -> (out [B, T, Hq, D], None). B must be 1; causal over S == T."""
    fn, ver = _fa_func()
    B, Hq, T, D = query.shape
    assert B == 1, "vllm_fa source attention supports one sequence"
    S = key.shape[2]
    q = query[0].transpose(0, 1).contiguous()  # [T, Hq, D]
    k = key[0].transpose(0, 1).contiguous()
    v = value[0].transpose(0, 1).contiguous()
    cu_q = torch.tensor([0, T], dtype=torch.int32, device=q.device)
    cu_k = torch.tensor([0, S], dtype=torch.int32, device=q.device)
    kw = {}
    if ver is not None:
        v_ = ver(head_size=D)
        if v_ is not None:
            kw["fa_version"] = v_
    out = fn(q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=T, max_seqlen_k=S,
             softmax_scale=scaling if scaling is not None else D ** -0.5, causal=True, **kw)
    if isinstance(out, tuple):
        out = out[0]
    return out[None], None


def use_vllm_fa(model):
    """Register the kernel with transformers and switch `model` to it. Returns True on success."""
    global _REGISTERED
    from transformers import AttentionInterface
    if not _REGISTERED:
        AttentionInterface.register(NAME, vllm_fa_attention)
        _REGISTERED = True
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(NAME)
    else:
        model.config._attn_implementation = NAME
    return True
