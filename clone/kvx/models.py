"""Load a causal LM; for multimodal checkpoints (Gemma-3 4B+), expose the text model as a plain causal LM."""
from types import SimpleNamespace

import torch


class TextLM(torch.nn.Module):
    """Wrap a *ForConditionalGeneration: .model = inner text model (layers, rotary_emb), .config = text config."""

    def __init__(self, cg):
        super().__init__()
        inner = cg.model
        self.model = inner.language_model
        self.lm_head = cg.lm_head
        self.config = cg.config.text_config
        self._cg = cg  # keep weights (vision tower unused)

    def forward(self, input_ids=None, past_key_values=None, use_cache=True, logits_to_keep=0,
                output_hidden_states=False, **kw):
        out = self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache,
                         output_hidden_states=output_hidden_states, **kw)
        h = out.last_hidden_state
        if logits_to_keep:
            h = h[:, -logits_to_keep:]
        logits = self.lm_head(h)
        cap = getattr(self.config, "final_logit_softcapping", None)
        if cap:
            logits = torch.tanh(logits / cap) * cap
        return SimpleNamespace(logits=logits, past_key_values=out.past_key_values,
                               hidden_states=getattr(out, "hidden_states", None))

    def set_attn_implementation(self, impl):
        self._cg.set_attn_implementation(impl)


SHARED_HUB = "/hf/hub"  # the shared weights volume (KVX_WEIGHTS_VOLUME) mounted read-only at /hf


def model_path(name):
    """Local snapshot dir of `name` on the shared weights volume; never downloads into a seat volume.
    Off Modal (Vast box via run_local.py) the shared mount is absent: return the repo id (Hub download on the box)."""
    import os
    if not os.path.isdir(SHARED_HUB):
        return name
    rd = f"{SHARED_HUB}/models--{name.replace('/', '--')}"
    if not os.path.isdir(rd):
        raise RuntimeError(f"{name} is not on the shared weights volume: stage it first with "
                           f"`modal run ~/rsi-weights/stage.py::main --repos {name} --hub` (CPU container)")
    from huggingface_hub import snapshot_download
    return snapshot_download(name, cache_dir=SHARED_HUB, local_files_only=True)


def apply_yarn(cfg, factor, orig_ctx=32768):
    """Set YaRN on an HF config for both transformers 4 (rope_scaling) and 5 (rope_parameters); keeps rope_theta and the
    other rope keys (partial_rotary_factor, mrope_section: Qwen3.5/3.8). Multimodal checkpoints: on text_config."""
    tc = getattr(cfg, "text_config", None) or cfg
    rp = dict(getattr(tc, "rope_parameters", None) or {})
    theta = rp.get("rope_theta") or getattr(tc, "rope_theta", None) or 10000.0
    y = dict(rp, rope_type="yarn", factor=float(factor), original_max_position_embeddings=int(orig_ctx),
             rope_theta=float(theta))
    y.pop("type", None)
    tc.rope_scaling = dict(y)
    try:
        tc.rope_parameters = dict(y)
    except Exception:
        pass
    tc.max_position_embeddings = int(orig_ctx * factor)
    return cfg


def load_lm(name, dtype=torch.bfloat16, attn="sdpa", yarn=0.0, orig_ctx=32768, device=None):
    """yarn > 0: load with YaRN rope scaling (factor `yarn` over `orig_ctx`), e.g. Qwen3 at 128K: yarn=4.
    device: load the weights straight onto that device (device_map) instead of staging the whole model in host RAM
    (a 32B in bf16 is 65 GB: too much for a 64 GB host)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    name = model_path(name)
    cfg = AutoConfig.from_pretrained(name)
    kw = dict(dtype=dtype, attn_implementation=attn)
    if device is not None:
        kw["device_map"] = str(device)
    if yarn > 0:
        apply_yarn(cfg, yarn, orig_ctx)
        return AutoModelForCausalLM.from_pretrained(name, config=cfg, **kw)
    if hasattr(cfg, "text_config") and cfg.architectures and "ConditionalGeneration" in cfg.architectures[0]:
        from transformers import AutoModelForImageTextToText
        cg = AutoModelForImageTextToText.from_pretrained(name, **kw)
        return TextLM(cg)
    return AutoModelForCausalLM.from_pretrained(name, **kw)
