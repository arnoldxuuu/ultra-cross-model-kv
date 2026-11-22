"""CPU unit tests for kvx math (no GPU): RoPE strip/re-apply, ridge vs least squares, top-k selection,
restore modes, the connector's paged-KV write indexing, and the Triton epilogue (TRITON_INTERPRET=1).
Run: modal run tests/modal_unit_cpu.py   (or pytest -q tests/test_core_cpu.py where torch is installed)"""
import math
import os

import pytest
import torch

from kvx import core

torch.manual_seed(0)


def _cos_sin(T, D, theta=10000.0):
    inv = 1.0 / theta ** (torch.arange(0, D, 2, dtype=torch.float64) / D)
    f = torch.outer(torch.arange(T, dtype=torch.float64), inv)
    emb = torch.cat([f, f], -1)
    return emb.cos().float(), emb.sin().float()


def test_rope_roundtrip_and_relative_property():
    T, D = 37, 64
    cos, sin = _cos_sin(T, D)
    k = torch.randn(2, 4, T, D)
    assert torch.allclose(core.unrope(core.rope(k, cos, sin), cos, sin), k, atol=1e-5)
    # RoPE scores depend only on the position difference: q@m . k@n == q@(m+s) . k@(n+s)
    q, kk = torch.randn(D), torch.randn(D)
    def at(x, p):
        return core.rope(x[None], cos[p:p + 1], sin[p:p + 1])[0]
    a = (at(q, 9) * at(kk, 3)).sum()
    b = (at(q, 30) * at(kk, 24)).sum()
    assert abs(a - b) < 1e-4


def test_cs_per_layer_tables():
    cos = torch.arange(2 * 5 * 4, dtype=torch.float32).view(2, 5, 4)
    c, s = core._cs(cos, cos, 1, 3)
    assert c.shape == (3, 4) and torch.equal(c, cos[1, :3])
    c2, _ = core._cs(cos[0], cos[0], 1, 3)
    assert torch.equal(c2, cos[0, :3])


def _toy_stats(N=4000, Ls=3, Lt=2, H=2, D=4, noise=0.05):
    ds, dt = H * D, H * D
    X = torch.randn(N, Ls * ds) * 3 + 5  # large mean: exercises the shifted accumulation
    Wtrue = torch.zeros(Ls * ds, Lt * dt)
    Wtrue[ds:2 * ds, :dt] = torch.randn(ds, dt)  # target 0 depends only on source layer 1
    Wtrue[2 * ds:, dt:] = torch.randn(ds, dt)    # target 1 depends only on source layer 2
    Y = X @ Wtrue + 1.5 + noise * torch.randn(N, Lt * dt)
    return X, Y, Wtrue, (Ls, ds, Lt, dt, H)


def test_gram_ridge_matches_lstsq_and_picks_true_layers():
    X, Y, Wtrue, (Ls, ds, Lt, dt, H) = _toy_stats()
    st = core.GramStats(Ls, ds, Lt, dt, "cpu")
    for i in range(0, X.shape[0], 1000):  # streamed in chunks like calibration
        st.add(X[i:i + 1000], Y[i:i + 1000])
    st.finalize()
    assert torch.allclose(st.mx, X.mean(0), atol=1e-3)
    R2 = core.layer_r2(st, 1e-6, H)
    assert R2.shape == (Lt, H, Ls)
    assert R2[0].mean(0).argmax().item() == 1 and R2[1].mean(0).argmax().item() == 2
    maps = core.fit_maps(st, R2, 1, 1e-6, H)
    assert maps[0]["sel"] == [1] and maps[1]["sel"] == [2]
    # top-1 map for target 0 == least squares of Y0 on [X_s1, 1]
    Xa = torch.cat([X[:, ds:2 * ds], torch.ones(len(X), 1)], 1).double()
    sol = torch.linalg.lstsq(Xa, Y[:, :dt].double()).solution
    assert torch.allclose(maps[0]["W"].double(), sol[:-1], atol=2e-3)
    assert torch.allclose(maps[0]["b"].double(), sol[-1], atol=2e-2)
    assert (maps[0]["r2"] > 0.99).all()


def test_weighted_add_equals_row_duplication():
    X, Y, _, (Ls, ds, Lt, dt, H) = _toy_stats(N=300)
    w = torch.randint(1, 4, (300,)).float()
    a = core.GramStats(Ls, ds, Lt, dt, "cpu"); a.add(X, Y, w); a.finalize()
    rep = torch.repeat_interleave(torch.arange(300), w.long())
    b = core.GramStats(Ls, ds, Lt, dt, "cpu"); b.add(X[rep], Y[rep]); b.finalize()
    assert a.n == b.n
    assert torch.allclose(a.xx, b.xx, rtol=1e-3, atol=1e-1)
    assert torch.allclose(a.xy, b.xy, rtol=1e-3, atol=1e-1)


def test_apply_maps_layout_matches_manual():
    X, Y, _, (Ls, ds, Lt, dt, H) = _toy_stats()
    st = core.GramStats(Ls, ds, Lt, dt, "cpu"); st.add(X, Y); st.finalize()
    maps = core.fit_maps(st, core.layer_r2(st, 1e-6, H), 2, 1e-6, H)
    B, T, D = 2, 5, dt // H
    Xs = torch.randn(Ls, B, H, T, D)
    out = core.apply_maps(maps, Xs)
    assert out.shape == (Lt, B, H, T, D)
    b, t, l = 1, 3, 1
    feat = torch.cat([Xs[s, b, :, t].reshape(-1) for s in maps[l]["sel"]])
    ref = (feat @ maps[l]["W"] + maps[l]["b"]).view(H, D)
    assert torch.allclose(out[l, b, :, t], ref, atol=1e-4)


def test_restore_rms_matches_k_norm():
    L, B, H, T, D = 2, 1, 3, 6, 8
    w = torch.rand(L, D) + 0.5
    raw = torch.randn(L, B, H, T, D) * 7
    native = w.view(L, 1, 1, 1, D) * raw * torch.rsqrt(raw.pow(2).mean(-1, keepdim=True) + 1e-6)
    maps = [dict(mu=torch.zeros(H * D)) for _ in range(L)]
    # a key that is native up to a per-row scale must be restored exactly
    scaled = native * (torch.rand(L, B, H, T, 1) * 3 + 0.2)
    out = core.restore_scale(scaled, maps, "rms", H, kw=w)
    assert torch.allclose(out, native, atol=1e-4)
    assert core.restore_scale(scaled, maps, "none", H) is scaled


def test_restore_global_gain():
    L, B, H, T, D = 1, 1, 2, 4, 4
    mu = torch.randn(H * D)
    K = torch.randn(L, B, H, T, D)
    out = core.restore_scale(K, [dict(mu=mu)], "g1.1", H)
    m = mu.view(1, H, 1, D)
    assert torch.allclose(out[0], m + 1.1 * (K[0] - m), atol=1e-6)


def test_fp8_round_error_bound():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("no fp8 dtype")
    x = torch.randn(64, 256)
    r = core._fp8_round(x, 1)
    rel = ((r - x).norm() / x.norm()).item()
    assert rel < 0.04  # e4m3 has 3 mantissa bits: ~2-3% relative error


def _slots(block_ids, block, m):
    blocks = torch.tensor(block_ids)
    return (torch.arange(block)[None, :] + blocks[:, None] * block).flatten()[:m]


@pytest.mark.parametrize("T", [16, 33, 47])
def test_paged_write_matches_reference_loop(T):
    """The connector's vLLM 0.34 write kv[bi, :, off, :D] = K[T,H,D] lands token t at block bids[t//bs], row t%bs."""
    nb, H, bs, D = 9, 3, 16, 8
    bids = [7, 2, 5, 0]
    m = (T // bs) * bs  # _n_mapped: whole blocks only
    slots = _slots(bids, bs, m)
    bi, off = slots // bs, slots % bs
    K, V = torch.randn(m, H, D), torch.randn(m, H, D)
    kv = torch.zeros(nb, H, bs, 2 * D)
    kv[bi, :, off, :D] = K
    kv[bi, :, off, D:] = V
    ref = torch.zeros_like(kv)
    for t in range(m):
        ref[bids[t // bs], :, t % bs, :D] = K[t]
        ref[bids[t // bs], :, t % bs, D:] = V[t]
    assert torch.equal(kv, ref)
    # old layout [2, nb, bs, H, D]
    kv2 = torch.zeros(2, nb, bs, H, D)
    kv2[0][bi, off] = K
    assert torch.equal(kv2[0][bids[0], :, :, :], ref[bids[0]].permute(1, 0, 2)[..., :D]) if m else True


def test_n_mapped_leaves_tail_and_whole_blocks():
    from types import SimpleNamespace
    pytest.importorskip("vllm")
    from kvx.vllm_connector import CrossModelConnector
    c = SimpleNamespace(min_len=64, tail=1, block=16)
    f = CrossModelConnector._n_mapped
    assert f(c, 63) == 0
    assert f(c, 64) == 48 and f(c, 65) == 64 and f(c, 1000) == 992


def test_triton_epilogue_matches_torch():
    os.environ.setdefault("TRITON_INTERPRET", "1")
    triton = pytest.importorskip("triton")
    from kvx.triton_kernels import restore_rope_
    T, Lt, H, D = 5, 2, 3, 16
    cos, sin = _cos_sin(T, D)
    w = torch.rand(Lt, D) + 0.5
    Y = torch.randn(T, Lt, H, D)
    ref = []
    for l in range(Lt):
        k = Y[:, l].permute(1, 0, 2)[None, None]  # [1,1,H,T,D]
        k = core.restore_scale(k, [dict(mu=torch.zeros(H * D))], "rms", H, kw=w[l:l + 1])[0, 0]
        ref.append(core.rope(k, cos, sin).permute(1, 0, 2))
    ref = torch.stack(ref, 1)
    out = restore_rope_(Y.clone().contiguous(), w, cos.contiguous(), sin.contiguous())
    assert torch.allclose(out, ref, atol=1e-4)


def test_paged_write_with_prefix_hit_leaves_shared_blocks():
    """Connector start offset (vLLM prefix-cache hit of p tokens): rows < p keep native KV, rows >= p get mapped KV."""
    nb, H, bs, D = 8, 2, 16, 4
    bids, m, p = [3, 6, 1, 4], 64, 32
    native = torch.randn(nb, H, bs, 2 * D)
    kv = native.clone()
    slots = _slots(bids, bs, m)[p:]
    bi, off = slots // bs, slots % bs
    K = torch.randn(m, H, D)
    kv[bi, :, off, :D] = K[p:]
    assert torch.equal(kv[3], native[3]) and torch.equal(kv[6], native[6])  # shared prefix blocks untouched
    assert torch.equal(kv[1, :, :, :D], K[32:48].permute(1, 0, 2))


def test_fp8sim_close_to_dense_with_large_mean():
    X, Y, _, (Ls, ds, Lt, dt, H) = _toy_stats()
    st = core.GramStats(Ls, ds, Lt, dt, "cpu"); st.add(X, Y); st.finalize()
    maps = core.fit_maps(st, core.layer_r2(st, 1e-6, H), 2, 1e-6, H)
    Xs = (torch.randn(Ls, 1, H, 64, dt // H) * 3 + 5)
    ref, q = core.apply_maps(maps, Xs), core.apply_maps_fp8sim(maps, Xs)
    rel = ((q - ref).norm() / (ref - ref.mean()).norm()).item()
    assert rel < 0.08, rel


def _tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(1)
    cfg = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=256)
    m = Qwen3ForCausalLM(cfg).eval()
    m.config._attn_implementation = "sdpa"
    return m


def test_recompute_all_positions_equals_native():
    from kvx.recompute import recompute_cache
    m = _tiny_qwen3()
    B, P, C = 2, 24, 6
    ids = torch.randint(0, 97, (B, P + C))
    cos, sin = core.rope_cos_sin(m, P + C, "cpu")
    with torch.no_grad():
        ref = m(input_ids=ids).logits[:, P:]
    K, V = torch.randn(3, B, 2, P, 16), torch.randn(3, B, 2, P, 16)  # garbage "mapped" cache
    sel = torch.arange(P)[None].expand(B, P).contiguous()
    c = recompute_cache(m, ids[:, :P], K, V, sel, cos, sin)
    with torch.no_grad():
        out = m(input_ids=ids[:, P:], past_key_values=c).logits
    assert torch.allclose(out, ref, atol=2e-2), (out - ref).abs().max()


def test_recompute_last_n_equals_tail_prefill():
    from kvx.recompute import recompute_cache
    m = _tiny_qwen3()
    B, P, N = 1, 20, 5
    ids = torch.randint(0, 97, (B, P))
    cos, sin = core.rope_cos_sin(m, P, "cpu")
    K, V = torch.randn(3, B, 2, P, 16), torch.randn(3, B, 2, P, 16)
    c1 = recompute_cache(m, ids, K, V, torch.arange(P - N, P)[None], cos, sin)
    c2 = core.build_cache(K[:, :, :, :P - N].contiguous(), V[:, :, :, :P - N].contiguous(), cos, sin)
    with torch.no_grad():
        m(input_ids=ids[:, P - N:], past_key_values=c2)
    for (k1, v1), (k2, v2) in zip(core.cache_layers(c1), core.cache_layers(c2)):
        assert k1.shape == k2.shape
        assert torch.allclose(k1.float(), k2.float(), atol=2e-2) and torch.allclose(v1.float(), v2.float(), atol=2e-2)


def test_rank_disagreement_scores_out_of_span_tokens():
    from kvx.recompute import PredBasis, rank_disagreement, select_top
    torch.manual_seed(2)
    L, H, D = 2, 2, 4
    basis_dirs = torch.randn(H * D, 2)
    calib = (torch.randn(L, 4, 50, 2) @ basis_dirs.T).view(L, 4, 50, H, D).permute(0, 1, 3, 2, 4)  # rank-2 data
    pb = PredBasis(); pb.add("k", calib); pb.finalize(r=2)
    Y = (torch.randn(L, 1, 10, 2) @ basis_dirs.T).view(L, 1, 10, H, D).permute(0, 1, 3, 2, 4).clone()
    Y[:, :, :, 7] += 5 * torch.randn(L, 1, H, D)  # token 7 leaves the calibration span
    s = rank_disagreement(pb, {"k": Y})
    assert s.argmax().item() == 7 and s[0, :7].max() < 1e-3
    assert select_top(s, 0.2, keep_first=1).tolist() == [[0, 7]]


def _import_connector_with_stub_vllm():
    import sys, types, logging
    names = ["vllm", "vllm.distributed", "vllm.distributed.kv_transfer", "vllm.distributed.kv_transfer.kv_connector",
             "vllm.distributed.kv_transfer.kv_connector.v1", "vllm.distributed.kv_transfer.kv_connector.v1.base",
             "vllm.logger"]
    for n in names:
        sys.modules.setdefault(n, types.ModuleType(n))
    base = sys.modules["vllm.distributed.kv_transfer.kv_connector.v1.base"]
    base.KVConnectorBase_V1 = getattr(base, "KVConnectorBase_V1", type("KVConnectorBase_V1", (), {}))
    base.KVConnectorMetadata = getattr(base, "KVConnectorMetadata", type("KVConnectorMetadata", (), {}))
    base.KVConnectorRole = getattr(base, "KVConnectorRole", type("KVConnectorRole", (), {}))
    sys.modules["vllm.logger"].init_logger = logging.getLogger
    import importlib
    return importlib.import_module("kvx.vllm_connector")


@pytest.mark.parametrize("chunk", [1, 2, 3])
def test_connector_chunked_map_equals_unchunked_and_injects_sink(chunk):
    """_map_layers in target-layer chunks (memory fix for 72B at 32K) == one GEMM; sink template overwrites rows < A."""
    vc = _import_connector_with_stub_vllm()
    S = _tiny_qwen3()
    T_, Ls, Hs, Ds, Lt, H, D = 12, 3, 2, 16, 5, 2, 8
    cs, ss = core.rope_cos_sin(S, T_, "cpu")
    ct, st = _cos_sin(T_, D)
    def make(ch, sink):
        c = vc.CrossModelConnector.__new__(vc.CrossModelConnector)
        c.core, c.S, c.feat, c.fp8, c.chunk, c.kw = core, S, "kv", False, ch, None
        c.ds, c.H, c.D = Hs * Ds, H, D
        g = torch.Generator().manual_seed(3)
        c.Wk = torch.randn(Ls * Hs * Ds, Lt * H * D, generator=g) * 0.1
        c.Wv = torch.randn(Ls * Hs * Ds, Lt * H * D, generator=g) * 0.1
        c.bk, c.bv = torch.randn(Lt * H * D, generator=g), torch.randn(Lt * H * D, generator=g)
        c.cs, c.ss, c.ct, c.st = cs, ss, ct, st
        c.sinkK = c.sinkV = None
        if sink:
            c.sinkK, c.sinkV = torch.full((2, Lt, H, D), 7.0), torch.full((2, Lt, H, D), -3.0)
        return c
    os.environ.setdefault("TRITON_INTERPRET", "1")
    ids = torch.randint(0, 97, (T_,)).tolist()
    ref = [(l, K.clone(), V.clone()) for l, K, V in make(Lt, False)._map_layers(ids)]
    got = [(l, K.clone(), V.clone()) for l, K, V in make(chunk, False)._map_layers(ids)]
    assert [l for l, _, _ in got] == list(range(Lt))
    for (_, K1, V1), (_, K2, V2) in zip(ref, got):
        assert torch.allclose(K1, K2, atol=1e-5) and torch.allclose(V1, V2, atol=1e-5)
    sk = [(l, K.clone(), V.clone()) for l, K, V in make(chunk, True)._map_layers(ids)]
    for (_, K1, V1), (_, K2, V2) in zip(ref, sk):
        assert (K2[:2] == 7).all() and (V2[:2] == -3).all()
        assert torch.allclose(K1[2:], K2[2:], atol=1e-5) and torch.allclose(V1[2:], V2[2:], atol=1e-5)


@pytest.mark.parametrize("arch", ["qwen2", "qwen3"])
def test_target_cos_sin_default_rope_matches_hf(arch):
    vc = _import_connector_with_stub_vllm()
    from transformers import Qwen2Config, Qwen2ForCausalLM
    if arch == "qwen2":
        cfg = Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                          num_attention_heads=4, num_key_value_heads=2, rope_theta=1e6, max_position_embeddings=256)
        m = Qwen2ForCausalLM(cfg).eval()
    else:
        m = _tiny_qwen3(); cfg = m.config
    c_ref, s_ref = core.rope_cos_sin(m, 50, "cpu")
    c, s = vc.target_cos_sin(cfg, 50, "cpu")
    assert torch.allclose(c, c_ref, atol=1e-5) and torch.allclose(s, s_ref, atol=1e-5)


@pytest.mark.parametrize("local_hit", [0, 32])
def test_connector_scheduler_flow_writes_whole_mapped_prefix(local_hit):
    """Replay vLLM's scheduler call order. NewRequestData.num_computed_tokens already counts the connector's own
    external tokens, so the write start must come from the local prefix-cache hit (bug: start == m, nothing written)."""
    from types import SimpleNamespace as NS
    vc = _import_connector_with_stub_vllm()
    c = vc.CrossModelConnector.__new__(vc.CrossModelConnector)
    c.min_len, c.tail, c.block, c._need_load, c._local_hit = 64, 16, 16, {}, {}
    ids = list(range(100))
    req = NS(request_id="r", prompt_token_ids=ids)
    n_ext, _ = c.get_num_new_matched_tokens(req, local_hit)
    assert n_ext == 80 - local_hit
    c.update_state_after_alloc(req, None, n_ext)
    new = NS(req_id="r", prompt_token_ids=ids, block_ids=([3, 5, 7, 9, 11, 2, 4],), num_computed_tokens=local_hit + n_ext)
    meta = c.build_connector_meta(NS(scheduled_new_reqs=[new]))
    r = meta.requests[0]
    assert r.start == local_hit and len(r.token_ids) == 80 and len(r.slot_mapping) == 80
    assert r.slot_mapping[local_hit].item() == [3, 5, 7, 9, 11][local_hit // 16] * 16


@pytest.mark.parametrize("T,Lt,rms", [(37, 3, True), (9, 4, False)])
def test_triton_rows_epilogue_many_programs(T, Lt, rms):
    os.environ.setdefault("TRITON_INTERPRET", "1")
    pytest.importorskip("triton")
    from kvx.triton_kernels import restore_rope_
    H, D = 2, 16
    cos, sin = _cos_sin(T, D)
    w = torch.rand(Lt, D) + 0.5
    Y = torch.randn(T, Lt, H, D)
    ref = []
    for l in range(Lt):
        k = Y[:, l].permute(1, 0, 2)[None, None]
        if rms:
            k = core.restore_scale(k, [dict(mu=torch.zeros(H * D))], "rms", H, kw=w[l:l + 1])
        ref.append(core.rope(k[0, 0], cos, sin).permute(1, 0, 2))
    ref = torch.stack(ref, 1)
    out = restore_rope_(Y.clone().contiguous(), w if rms else None, cos.contiguous(), sin.contiguous(), rows=8)
    assert torch.allclose(out, ref, atol=1e-4)


def test_fp8_linear_emulated_matches_simulated_quant():
    from kvx.fp8 import FP8Linear, quantize_linears
    torch.manual_seed(5)
    lin = torch.nn.Linear(64, 48, bias=True)
    x = torch.randn(3, 7, 64)
    q = FP8Linear(lin, emulate=True)
    y = q(x)
    assert y.shape == (3, 7, 48)
    rel = ((y - lin(x)).norm() / lin(x).norm()).item()
    assert rel < 0.06, rel  # e4m3 on both operands
    m = _tiny_qwen3()
    n = quantize_linears(m, emulate=True)
    assert n == 3 * 7  # q,k,v,o + gate,up,down per layer; lm_head skipped
    ids = torch.randint(0, 97, (1, 10))
    assert torch.isfinite(m(input_ids=ids).logits).all()


def test_yarn_config_and_connector_rope_match_hf():
    vc = _import_connector_with_stub_vllm()
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from kvx.models import apply_yarn
    cfg = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=64, rope_theta=1e6)
    apply_yarn(cfg, 4.0, 64)
    assert cfg.max_position_embeddings == 256
    m = Qwen3ForCausalLM(cfg).eval()
    c_ref, s_ref = core.rope_cos_sin(m, 200, "cpu")  # HF's own YaRN rotary
    c, s = vc.target_cos_sin(cfg, 200, "cpu")
    assert torch.allclose(c, c_ref, atol=1e-5) and torch.allclose(s, s_ref, atol=1e-5)
    plain = Qwen3ForCausalLM(Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, rope_theta=1e6)).eval()
    c_plain, _ = core.rope_cos_sin(plain, 200, "cpu")
    assert not torch.allclose(c_plain, c_ref, atol=1e-3)  # YaRN really changed the tables


def test_connector_init_worker_local_dir_rms_yarn(tmp_path):
    """Full _init_worker on CPU with the target given as a LOCAL snapshot dir (as vLLM passes it with hf_overrides),
    rms restore (k_norm read from safetensors) and YaRN on both models; then map a prompt end to end."""
    vc = _import_connector_with_stub_vllm()
    m = _tiny_qwen3()
    d = tmp_path / "tiny"
    m.save_pretrained(d, safe_serialization=True)
    Ls, H, D = 3, 2, 16
    ds = H * D
    g = torch.Generator().manual_seed(7)
    mk = lambda: [dict(sel=[0, 2], W=torch.randn(2 * ds, ds, generator=g) * 0.1, b=torch.randn(ds, generator=g),
                       mu=torch.zeros(ds), r2=torch.ones(H), vnat=torch.ones(H), H=H, D=D) for _ in range(Ls)]
    torch.save({"k": mk(), "v": mk(), "feat": "kv", "src": str(d), "tgt": str(d), "k_layers": 2}, tmp_path / "maps.pt")
    c = vc.CrossModelConnector.__new__(vc.CrossModelConnector)
    c.src_name, c.tgt_name, c.maps_path = str(d), str(d), str(tmp_path / "maps.pt")
    c.restore, c.src_device, c.fp8, c.sink_path, c.chunk, c.max_len = "rms", None, False, None, 2, 64
    c.yarn, c.orig_ctx, c.fp8_source = 2.0, 32, False
    os.environ.setdefault("TRITON_INTERPRET", "1")
    c._init_worker(torch.device("cpu"))
    assert c.kw is not None and c.kw.shape == (Ls, D)
    assert torch.allclose(c.kw.float(), torch.stack([l.self_attn.k_norm.weight for l in m.model.layers]).float(), atol=1e-2)
    out = list(c._map_layers(torch.randint(0, 97, (40,)).tolist()))
    assert [l for l, _, _ in out] == [0, 1, 2]
    assert all(K.shape == (40, H, D) and torch.isfinite(K).all() and torch.isfinite(V).all() for _, K, V in out)


@pytest.mark.parametrize("token_major,n", [(True, 48), (False, 48), (True, 40)])
def test_write_kv_fa_fast_path_matches_reference(token_major, n):
    vc = _import_connector_with_stub_vllm()
    nb, H, bs, D = 9, 2, 16, 4
    bids = [7, 2, 5, 0]
    slots = _slots(bids, bs, n)
    bi, off = slots // bs, slots % bs
    K, V = torch.randn(n, H, D), torch.randn(n, H, D)
    if token_major:  # what vLLM allocates: physical [blocks, block, H, 2D], handed over as a [blocks, H, block, 2D] view
        kv = torch.zeros(nb, bs, H, 2 * D).permute(0, 2, 1, 3)
    else:
        kv = torch.zeros(nb, H, bs, 2 * D)
    vc.write_kv_fa(kv, bi, off, K, V, bs)
    ref = torch.zeros(nb, H, bs, 2 * D)
    for t in range(n):
        ref[bids[t // bs], :, t % bs, :D] = K[t]
        ref[bids[t // bs], :, t % bs, D:] = V[t]
    assert torch.equal(kv.contiguous(), ref)


def test_connector_tp2_ranks_concat_to_tp1(tmp_path):
    """Tensor parallel: each rank maps only its KV heads; rank outputs joined along heads == the TP=1 output."""
    vc = _import_connector_with_stub_vllm()
    m = _tiny_qwen3()
    d = tmp_path / "tiny"
    m.save_pretrained(d, safe_serialization=True)
    Ls, H, D = 3, 2, 16
    ds = H * D
    g = torch.Generator().manual_seed(9)
    mk = lambda: [dict(sel=[0, 1, 2], W=torch.randn(3 * ds, ds, generator=g) * 0.1, b=torch.randn(ds, generator=g),
                       mu=torch.zeros(ds), r2=torch.ones(H), vnat=torch.ones(H), H=H, D=D) for _ in range(Ls)]
    torch.save({"k": mk(), "v": mk(), "feat": "kv"}, tmp_path / "maps.pt")
    A = 2
    torch.save({"K": torch.randn(Ls, H, A, D), "V": torch.randn(Ls, H, A, D)}, tmp_path / "sink.pt")
    os.environ.setdefault("TRITON_INTERPRET", "1")
    ids = torch.randint(0, 97, (24,)).tolist()

    def run(tp):
        c = vc.CrossModelConnector.__new__(vc.CrossModelConnector)
        c.src_name, c.tgt_name, c.maps_path = str(d), str(d), str(tmp_path / "maps.pt")
        c.restore, c.src_device, c.fp8, c.sink_path, c.chunk, c.max_len = "rms", None, False, str(tmp_path / "sink.pt"), 2, 64
        c.yarn, c.orig_ctx, c.fp8_source, c.src_sdpa, c.tp_override = 0.0, 32768, False, "cudnn", tp
        c.src_tp = False  # mapper column split only; the sharded source prefill is tests/test_tp_source_cpu.py
        c._init_worker(torch.device("cpu"))
        return {l: (K.clone(), V.clone()) for l, K, V in c._map_layers(ids)}
    full = run((0, 1))
    r0, r1 = run((0, 2)), run((1, 2))
    for l in range(Ls):
        K = torch.cat([r0[l][0], r1[l][0]], 1)
        V = torch.cat([r0[l][1], r1[l][1]], 1)
        assert r0[l][0].shape == (24, 1, D)
        assert torch.allclose(K, full[l][0], atol=1e-5) and torch.allclose(V, full[l][1], atol=1e-5)


def _tiny_qwen3_yarn(factor=64.0):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from kvx.models import apply_yarn
    torch.manual_seed(3)
    cfg = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=64, rope_theta=1e6)
    apply_yarn(cfg, factor, 16)
    m = Qwen3ForCausalLM(cfg).eval()
    m.config._attn_implementation = "sdpa"
    return m


def test_unrope_exact_with_yarn_attention_factor():
    m = _tiny_qwen3_yarn(64.0)
    T = 40
    cos, sin = core.rope_cos_sin(m, T, "cpu")
    mm = (cos * cos + sin * sin).sqrt()
    assert abs(mm.mean().item() - (0.1 * math.log(64.0) + 1)) < 1e-3  # YaRN m really is in the tables
    k = torch.randn(1, 2, T, 16)
    assert torch.allclose(core.unrope(core.rope(k, cos, sin), cos, sin), k, atol=1e-5)


def test_capture_kv_yarn_returns_true_pre_rope_keys():
    m = _tiny_qwen3_yarn(64.0)
    ids = torch.randint(0, 97, (1, 24))
    grab = {}
    a = m.model.layers[1].self_attn
    a.k_norm.register_forward_hook(lambda mod, i, o: grab.__setitem__("k", o.detach()))  # [B,T,H,D] pre-RoPE
    cos, sin = core.rope_cos_sin(m, 24, "cpu")
    K, V, _ = core.capture_kv(m, ids, cos, sin)
    ref = grab["k"].transpose(1, 2)  # [B,H,T,D]
    rel = ((K[1].float() - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, rel  # bf16 storage; the old unrope gave m^2 = 2.0x keys (rel ~1.0)


@pytest.mark.parametrize("layout", ["fa", "split5", "split4", "lead2"])
def test_write_kv_every_layout_matches_reference(layout):
    """ROCm backends (ROCM_ATTN, TRITON_ATTN, ROCM_AITER_FA) use [blocks, 2, block, ...]: the generic writer must put
    token t's K/V at (block bids[t//bs], offset t%bs) in each layout."""
    vc = _import_connector_with_stub_vllm()
    nb, H, bs, D, n = 9, 4, 16, 8, 40
    bids = [7, 2, 5]
    slots = _slots(bids, bs, n)
    bi, off = slots // bs, slots % bs
    K, V = torch.randn(n, H, D), torch.randn(n, H, D)
    shape = dict(fa=(nb, H, bs, 2 * D), split5=(nb, 2, bs, H, D), split4=(nb, 2, bs, H * D), lead2=(2, nb, bs, H, D))[layout]
    kv = torch.zeros(shape)
    assert vc.kv_layout(kv, H, D, "FLASH_ATTN" if layout == "fa" else "ROCM") == layout
    vc.write_kv(kv, bi, off, K, V, bs, layout)
    for t in range(n):
        b, o = bids[t // bs], t % bs
        if layout == "fa":
            got_k, got_v = kv[b, :, o, :D], kv[b, :, o, D:]
        elif layout == "split5":
            got_k, got_v = kv[b, 0, o], kv[b, 1, o]
        elif layout == "split4":
            got_k, got_v = kv[b, 0, o].view(H, D), kv[b, 1, o].view(H, D)
        else:
            got_k, got_v = kv[0, b, o], kv[1, b, o]
        assert torch.equal(got_k, K[t]) and torch.equal(got_v, V[t])
    untouched = [b for b in range(nb) if b not in bids]
    idx = dict(fa=0, split5=0, split4=0, lead2=1)[layout]
    assert float(kv.select(idx, untouched[0]).abs().sum()) == 0.0


def test_kv_layout_rejects_unknown():
    vc = _import_connector_with_stub_vllm()
    with pytest.raises(RuntimeError):
        vc.kv_layout(torch.zeros(3, 5, 16, 7), 4, 8, "X")


def test_kv_layout_h2_needs_backend():
    """H == 2 (e.g. TP2 of a 4-KV-head model): [blocks, 2, block, 2D] is FlashAttention's fa OR split4. Guessing
    without the backend wrote every TP2 K/V of the hybrid connector in the wrong layout (oracle KL 1.41)."""
    vc = _import_connector_with_stub_vllm()
    kv = torch.zeros(3, 2, 16, 16)
    assert vc.kv_layout(kv, 2, 8, "FLASH_ATTN") == "fa"
    assert vc.kv_layout(kv, 2, 8, "ROCM_ATTN") == "split4"
    with pytest.raises(RuntimeError):
        vc.kv_layout(kv, 2, 8, "")


def test_hfgen_oracle_mapping_equals_native_greedy():
    """With an oracle kv_fn (the target's own pre-RoPE KV), mapped generation must equal native greedy exactly."""
    from kvx.hfgen import make_hf_gen
    m = _tiny_qwen3()
    m.float()

    def oracle(ids):
        cos, sin = core.rope_cos_sin(m, ids.shape[1], "cpu")
        K, V, _ = core.capture_kv(m, ids, cos, sin)
        return K.float(), V.float()

    native = make_hf_gen(m, oracle, tail=10 ** 9, min_len=10 ** 9)  # never maps
    mapped = make_hf_gen(m, oracle, tail=3, min_len=4)
    prompts = [torch.randint(0, 97, (n,)).tolist() for n in (5, 12, 20)]
    a, b = native(prompts, 8), mapped(prompts, 8)
    assert a == b, (a, b)
