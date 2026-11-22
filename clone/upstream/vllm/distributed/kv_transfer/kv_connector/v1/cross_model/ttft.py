"""TTFT: native target prefill vs (source prefill + map + target tail recompute), HF forwards, batch 1.

python -m kvx.ttft --src Qwen/Qwen3-1.7B --tgt Qwen/Qwen3-32B --maps /vol/runs/<run>/maps_k8.pt --lens 4096,8192,16384,32768
"""
import argparse, json, time
import torch
from . import core


def timed(fn, reps=3, warm=1):
    for _ in range(warm):
        r = fn()
        del r
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
        del r
    ts.sort()
    return ts[len(ts) // 2]


@torch.no_grad()
def map_cache_fast(S, maps_k, maps_v, ids, cs, ss, ct, st, kw, H):
    """Source prefill -> pre-RoPE source KV (bf16 math) -> per-layer GEMM -> target cache (RoPE re-applied)."""
    from transformers import DynamicCache
    out = S(input_ids=ids, use_cache=True, logits_to_keep=1)
    T = ids.shape[1]
    c_s, s_s = cs[:T].to(torch.bfloat16), ss[:T].to(torch.bfloat16)
    ks, vs = [], []
    for k, v in core.cache_layers(out.past_key_values):
        ks.append(core.unrope(k, c_s, s_s))  # [1, H, T, D]
        vs.append(v)
    # [T, Ls, H*D]
    Xk = torch.stack(ks)[:, 0].permute(2, 0, 1, 3).reshape(T, len(ks), -1)
    Xv = torch.stack(vs)[:, 0].permute(2, 0, 1, 3).reshape(T, len(vs), -1)
    c_t, s_t = ct[:T].to(torch.bfloat16), st[:T].to(torch.bfloat16)
    cache = DynamicCache()
    for l, (mk, mv) in enumerate(zip(maps_k, maps_v)):
        K = torch.addmm(mk["b16"], Xk[:, mk["sel"]].reshape(T, -1), mk["W16"]).view(T, mk["H"], mk["D"])
        V = torch.addmm(mv["b16"], Xv[:, mv["sel"]].reshape(T, -1), mv["W16"]).view(T, mv["H"], mv["D"])
        if kw is not None:  # rms restore against the target k_norm weight
            w = kw[l]
            u = (K / w).float()
            K = (w * (u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + 1e-6))).to(torch.bfloat16)
        K = K.permute(1, 0, 2)[None]
        V = V.permute(1, 0, 2)[None]
        cache.update(core.rope(K, c_t, s_t), V, l)
    return cache


@torch.no_grad()
def map_cache_dense(S, Wk, bk, Wv, bv, Hs_, ids, cs, ss, ct, st, kw, H, D):
    """One GEMM per kind: X_all [T, Ls*ds] @ W_big [Ls*ds, Lt*H*D] (unselected source blocks are zero)."""
    from transformers import DynamicCache
    out = S(input_ids=ids, use_cache=True, logits_to_keep=1)
    T = ids.shape[1]
    c_s, s_s = cs[:T].to(torch.bfloat16), ss[:T].to(torch.bfloat16)
    lay = core.cache_layers(out.past_key_values)
    Xk = torch.cat([core.unrope(k, c_s, s_s)[0].transpose(0, 1).reshape(T, -1) for k, _ in lay], 1)
    Xv = torch.cat([v[0].transpose(0, 1).reshape(T, -1) for _, v in lay], 1)
    Yk = torch.addmm(bk, Xk, Wk).view(T, -1, H, D)  # [T, Lt, H, D]
    Yv = torch.addmm(bv, Xv, Wv).view(T, -1, H, D)
    if kw is not None:
        w = kw.view(1, -1, 1, D)
        u = Yk / w
        Yk = w * u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + 1e-6)
    c_t, s_t = ct[:T].to(torch.bfloat16)[:, None], st[:T].to(torch.bfloat16)[:, None]
    Yk = core.rope(Yk.view(T, -1, D), c_t, s_t).view(T, -1, H, D)
    cache = DynamicCache()
    for l in range(Yk.shape[1]):
        cache.update(Yk[:, l].transpose(0, 1)[None], Yv[:, l].transpose(0, 1)[None], l)
    return cache


@torch.no_grad()
def map_cache_blocks(S, maps_k, maps_v, ids, cs, ss, ct, st, kw, H, D):
    """Per target layer: Y_t = b_t + sum_j X[:, s_j] W_{t,j} with addmm on strided views (no gather copies):
    k-proportional FLOPs (dense does all Ls source layers)."""
    from transformers import DynamicCache
    out = S(input_ids=ids, use_cache=True, logits_to_keep=1)
    T = ids.shape[1]
    c_s, s_s = cs[:T].to(torch.bfloat16), ss[:T].to(torch.bfloat16)
    lay = core.cache_layers(out.past_key_values)
    Xk = torch.cat([core.unrope(k, c_s, s_s)[0].transpose(0, 1).reshape(T, -1) for k, _ in lay], 1)
    Xv = torch.cat([v[0].transpose(0, 1).reshape(T, -1) for _, v in lay], 1)
    ds = Xk.shape[1] // len(lay)
    c_t, s_t = ct[:T].to(torch.bfloat16)[:, None], st[:T].to(torch.bfloat16)[:, None]
    cache = DynamicCache()
    for l, (mk, mv) in enumerate(zip(maps_k, maps_v)):
        outs = []
        for X, m in ((Xk, mk), (Xv, mv)):
            Y = m["b16"].expand(T, -1).clone()
            for j, s_ in enumerate(m["sel"]):
                Y.addmm_(X[:, s_ * ds:(s_ + 1) * ds], m["W16"][j * ds:(j + 1) * ds])
            outs.append(Y.view(T, H, D))
        K, V = outs
        if kw is not None:
            w = kw[l]
            u = K / w
            K = w * u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + 1e-6)
        K = core.rope(K, c_t, s_t)
        cache.update(K.transpose(0, 1)[None], V.transpose(0, 1)[None], l)
    return cache


def dense_weights(maps, Ls, ds, dev):
    Lt = len(maps)
    dt = maps[0]["W16"].shape[1]
    W = torch.zeros(Ls * ds, Lt * dt, device=dev, dtype=torch.bfloat16)
    b = torch.cat([m["b16"] for m in maps])
    for t, m in enumerate(maps):
        for j, s_ in enumerate(m["sel"]):
            W[s_ * ds:(s_ + 1) * ds, t * dt:(t + 1) * dt] = m["W16"][j * ds:(j + 1) * ds]
    return W, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tgt", required=True)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--lens", default="4096,8192,16384,32768")
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--out", default="ttft.json")
    ap.add_argument("--dense", type=int, default=0)
    ap.add_argument("--profile", type=int, default=0)
    ap.add_argument("--blocks", type=int, default=0)
    ap.add_argument("--attn", default="sdpa")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM
    dev = "cuda"
    S = AutoModelForCausalLM.from_pretrained(a.src, dtype=torch.bfloat16, attn_implementation=a.attn).to(dev).eval()
    T = AutoModelForCausalLM.from_pretrained(a.tgt, dtype=torch.bfloat16, attn_implementation=a.attn).to(dev).eval()
    mp = torch.load(a.maps, map_location="cpu")
    for m in mp["k"] + mp["v"]:  # keep only bf16 weights on the GPU
        m["W16"], m["b16"] = m.pop("W").to(dev, torch.bfloat16), m.pop("b").to(dev, torch.bfloat16)
        for key in ("mu", "r2", "vnat"):
            m.pop(key, None)
    try:
        kw = [l.self_attn.k_norm.weight.detach().to(torch.bfloat16) for l in T.model.layers]
    except AttributeError:
        kw = None
    H = T.config.num_key_value_heads
    Lmax = max(int(x) for x in a.lens.split(","))
    cs, ss = core.rope_cos_sin(S, Lmax, dev)
    ct, st = core.rope_cos_sin(T, Lmax, dev)
    torch.set_grad_enabled(False)
    Ls, ds = S.config.num_hidden_layers, mp["k"][0]["W16"].shape[0] // len(mp["k"][0]["sel"])
    D = mp["k"][0]["D"]
    dense = None
    if a.dense:
        Wk, bk = dense_weights(mp["k"], Ls, ds, dev)
        Wv, bv = dense_weights(mp["v"], Ls, ds, dev)
        kwt = torch.stack(kw) if kw is not None else None
        dense = (Wk, bk, Wv, bv, kwt)
        for m in mp["k"] + mp["v"]:
            m.pop("W16", None)  # free per-layer copies (H100 headroom)
        torch.cuda.empty_cache()
    res = dict(src=a.src, tgt=a.tgt, tail=a.tail, gpu=torch.cuda.get_device_name(0), dense=bool(a.dense),
               blocks=bool(a.blocks), attn=a.attn, rows=[])
    for n in [int(x) for x in a.lens.split(",")]:
        ids = torch.randint(100, 30000, (1, n), device=dev)
        P = n - a.tail
        # same call shape as the mapped path (use_cache=True keeps SDPA on its causal fast path)
        t_native = timed(lambda: T(input_ids=ids, logits_to_keep=1))
        t_src = timed(lambda: S(input_ids=ids[:, :P], logits_to_keep=1))
        if a.blocks:
            mapper = lambda: map_cache_blocks(S, mp["k"], mp["v"], ids[:, :P], cs, ss, ct, st, kw, H, D)
        elif dense is None:
            mapper = lambda: map_cache_fast(S, mp["k"], mp["v"], ids[:, :P], cs, ss, ct, st, kw, H)
        else:
            Wk, bk, Wv, bv, kwt = dense
            mapper = lambda: map_cache_dense(S, Wk, bk, Wv, bv, None, ids[:, :P], cs, ss, ct, st, kwt, H, D)
        t_srcmap = timed(mapper)

        def full():
            c = mapper()
            return T(input_ids=ids[:, P:], past_key_values=c, logits_to_keep=1)
        t_full = timed(full)
        prof = {}
        if a.profile and dense is not None:
            Wk, bk, Wv, bv, kwt = dense
            ids_p = ids[:, :P]
            o = S(input_ids=ids_p, use_cache=True, logits_to_keep=1)
            lay = core.cache_layers(o.past_key_values)
            c_s, s_s = cs[:P].to(torch.bfloat16), ss[:P].to(torch.bfloat16)
            def unrope_cat():
                return (torch.cat([core.unrope(k, c_s, s_s)[0].transpose(0, 1).reshape(P, -1) for k, _ in lay], 1),
                        torch.cat([v[0].transpose(0, 1).reshape(P, -1) for _, v in lay], 1))
            prof["unrope_cat_s"] = timed(unrope_cat)
            Xk, Xv = unrope_cat()
            prof["gemm_kv_s"] = timed(lambda: (torch.addmm(bk, Xk, Wk), torch.addmm(bv, Xv, Wv)))
            Yk = torch.addmm(bk, Xk, Wk).view(P, -1, H, D)
            def restore_rope():
                w = kwt.view(1, -1, 1, D)
                u = Yk / w
                y = w * u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + 1e-6)
                c_t, s_t = ct[:P].to(torch.bfloat16)[:, None], st[:P].to(torch.bfloat16)[:, None]
                return core.rope(y.view(P, -1, D), c_t, s_t)
            prof["restore_rope_s"] = timed(restore_rope)
            c = mapper()
            def tail_only():
                out = T(input_ids=ids[:, P:], past_key_values=c, logits_to_keep=1)
                c.crop(P)
                return out
            prof["tail_forward_s"] = timed(tail_only)
            del o, lay, Xk, Xv, Yk, c
        row = dict(len=n, native_s=t_native, **prof, src_prefill_s=t_src, src_plus_map_s=t_srcmap, mapped_ttft_s=t_full,
                   speedup=t_native / t_full)
        print(json.dumps(row), flush=True)
        res["rows"].append(row)
        torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
