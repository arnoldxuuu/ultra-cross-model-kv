"""TP source prefill on CPU: 2 gloo processes; sharded forward == unsharded (logits and gathered KV)."""
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _tiny():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(3)
    cfg = Qwen2Config(vocab_size=101, hidden_size=64, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128)
    return Qwen2ForCausalLM(cfg).eval()


def _worker(rank, world, port, path, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from kvx import core
    from kvx.tp_source import gather_heads, shard_source
    m = _tiny()
    m.load_state_dict(torch.load(path))
    ids = torch.randint(0, 101, (1, 17), generator=torch.Generator().manual_seed(5))
    shard_source(m, rank, world)
    with torch.no_grad():
        o = m(input_ids=ids, use_cache=True)
    lay = core.cache_layers(o.past_key_values)
    K = torch.stack([gather_heads(k, world) for k, _ in lay])
    V = torch.stack([gather_heads(v, world) for _, v in lay])
    if rank == 0:
        torch.save({"logits": o.logits, "K": K, "V": V}, out)
    dist.destroy_process_group()


def test_tp2_source_matches_unsharded():
    m = _tiny()
    ids = torch.randint(0, 101, (1, 17), generator=torch.Generator().manual_seed(5))
    from kvx import core
    with torch.no_grad():
        o = m(input_ids=ids, use_cache=True)
    lay = core.cache_layers(o.past_key_values)
    K0 = torch.stack([k for k, _ in lay])
    V0 = torch.stack([v for _, v in lay])
    with tempfile.TemporaryDirectory() as d:
        sd, out = os.path.join(d, "sd.pt"), os.path.join(d, "out.pt")
        torch.save(m.state_dict(), sd)
        mp.spawn(_worker, args=(2, 29511, sd, out), nprocs=2, join=True)
        r = torch.load(out)
    assert torch.allclose(r["logits"], o.logits, atol=1e-4), (r["logits"] - o.logits).abs().max()
    assert torch.allclose(r["K"], K0, atol=1e-5) and torch.allclose(r["V"], V0, atol=1e-5)
