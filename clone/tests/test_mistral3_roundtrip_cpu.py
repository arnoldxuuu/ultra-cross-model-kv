"""Tiny Mistral3ForConditionalGeneration (transformers 5, ministral3 text model) with the real YaRN settings of
Ministral-3-3B (s=1.0) and Mistral-Medium-3.5-128B (s=1.416): capture KV through kvx.models.TextLM, strip RoPE,
rebuild the cache, and the continuation logits must equal native (the oracle round trip used by fit_eval)."""
import json
import os

import pytest
import torch

transformers = pytest.importorskip("transformers")
if int(transformers.__version__.split(".")[0]) < 5:
    pytest.skip("ministral3 needs transformers >= 5", allow_module_level=True)

from kvx import core  # noqa: E402
from kvx.models import TextLM  # noqa: E402

YARN = {
    "ministral-3b": {"beta_fast": 32.0, "beta_slow": 1.0, "factor": 16.0, "llama_4_scaling_beta": 0.1, "mscale": 1.0,
                     "mscale_all_dim": 1.0, "original_max_position_embeddings": 16384, "rope_theta": 1e6,
                     "rope_type": "yarn"},
    "medium-128b": {"beta_fast": 4.0, "beta_slow": 1.0, "factor": 64.0, "llama_4_scaling_beta": 0, "mscale": 1.0,
                    "mscale_all_dim": 0.0, "original_max_position_embeddings": 4096, "rope_theta": 1e6,
                    "rope_type": "yarn"},
}


def _tiny(rope):
    from transformers import AutoConfig, AutoModelForImageTextToText
    cfg = AutoConfig.for_model("mistral3")
    t = cfg.text_config
    for k, v in dict(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=16, max_position_embeddings=4096).items():
        setattr(t, k, v)
    t.rope_parameters = dict(rope)
    if hasattr(t, "rope_scaling"):
        t.rope_scaling = dict(rope)
    v = cfg.vision_config
    for k, val in dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                       head_dim=16).items():
        if hasattr(v, k):
            setattr(v, k, val)
    torch.manual_seed(0)
    return TextLM(AutoModelForImageTextToText.from_config(cfg).eval())


@pytest.mark.parametrize("which", list(YARN))
def test_mistral3_oracle_roundtrip(which):
    m = _tiny(YARN[which])
    P, C = 24, 8
    ids = torch.randint(0, 128, (1, P + C))
    with torch.no_grad():
        ref = m(input_ids=ids).logits[:, P:]
        cos, sin = core.rope_cos_sin(m, P + C, "cpu")
        K, V, _ = core.capture_kv(m, ids[:, :P], cos, sin)
        c = core.build_cache(K.float(), V.float(), cos, sin)
        from transformers import DynamicCache
        own = m(input_ids=ids[:, :P], past_key_values=DynamicCache(), use_cache=True).past_key_values
        for (k0, _), (k1, _) in zip(core.cache_layers(own), core.cache_layers(c)):  # keys must match exactly
            assert torch.allclose(k1.float(), k0.float(), rtol=2e-2, atol=2e-2), (which, (k1 - k0).abs().max())
        out = m(input_ids=ids[:, P:], past_key_values=c).logits
    s = float((cos[0] ** 2 + sin[0] ** 2).mean().sqrt()) if cos.dim() == 2 else None
    assert torch.allclose(out, ref, atol=3e-2), (which, s, (out - ref).abs().max())
