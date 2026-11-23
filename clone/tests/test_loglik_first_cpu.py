"""score_item(first=f): the first f context tokens use the target's native KV. With f covering the whole mapped
prefix, any mapping (even zeros) must give the native scores."""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")  # before triton is imported (other CPU tests rely on it)
import torch  # noqa: E402
from kvx import bench, core  # noqa: E402


class _Tok:
    name_or_path = "qwen-tiny"
    bos_token_id = None
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [3 + (ord(c) % 90) for c in text]}


def _tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=128)
    m = Qwen3ForCausalLM(cfg).eval()
    m.config._attn_implementation = "eager"
    return m


def test_first_covering_prefix_equals_native():
    T = _tiny()
    tok = _Tok()
    cs, ss = core.rope_cos_sin(T, 128, "cpu")
    ctx, conts = "the quick brown fox jumps", [" over", " under it"]
    zero = lambda Ks, Vs: (torch.zeros_like(Ks), torch.zeros_like(Vs))
    with torch.no_grad():
        nat = bench.score_item(tok, T, T, ctx, conts, "native", cs, ss, cs, ss, "cpu", tail=2)
        full = bench.score_item(tok, T, T, ctx, conts, zero, cs, ss, cs, ss, "cpu", tail=2, first=10 ** 6)
        none = bench.score_item(tok, T, T, ctx, conts, zero, cs, ss, cs, ss, "cpu", tail=2, first=0)
    assert all(abs(a - b) < 1e-3 for a, b in zip(nat, full)), (nat, full)
    assert any(abs(a - b) > 1e-2 for a, b in zip(nat, none))
