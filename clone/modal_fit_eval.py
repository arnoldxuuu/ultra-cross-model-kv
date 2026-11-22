"""Fit the cross-model ridge KV mapper and measure next-token KL against native prefill.

modal run modal_fit_eval.py --src Qwen/Qwen3-1.7B --tgt Qwen/Qwen3-8B --gpu L40S
"""
import json, os, time
try:
    import modal
except ImportError:  # rented box via run_local.py: provide a no-op stand-in
    class _Stub:
        def __getattr__(self, name):
            return _Stub()
        def __call__(self, *a, **k):
            if len(a) == 1 and callable(a[0]) and not k:
                return a[0]
            return _Stub()
    modal = _Stub()

GPU = os.environ.get("KVX_GPU", "L40S")
GPU_SPEC = GPU.split(",") if "," in GPU else GPU
APP = os.environ.get("KVX_APP", "kvx-fit")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "transformers==4.56.2", "accelerate", "datasets==3.6.0", "numpy", "hf_transfer")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", **{k: os.environ[k] for k in ("KVX_VOLUME", "KVX_WEIGHTS_VOLUME") if k in os.environ}})
    .add_local_python_source("kvx")
)
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"), create_if_missing=True)
hf_shared = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))  # shared weights volume (KVX_WEIGHTS_VOLUME), read-only
app = modal.App(APP, image=image)


def _commit():
    try:
        vol.commit()
    except Exception:
        pass  # running outside Modal (Vast box via run_local.py)


EVAL_DOCS = (20000, 30000)  # stream doc-index range reserved for eval; calibration never reads it


def load_texts(tok, n, L, skip_docs=0, seed_name="sample-10BT", exclude=None, info=None):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=seed_name, split="train", streaming=True)
    out = []
    for i, ex in enumerate(ds):
        if i < skip_docs or (exclude and exclude[0] <= i < exclude[1]):
            continue
        ids = tok(ex["text"], add_special_tokens=False)["input_ids"]
        if info is not None:
            info["last_doc"] = i
        if tok.bos_token_id is not None and "qwen" not in tok.name_or_path.lower():
            ids = [tok.bos_token_id] + ids  # Llama/Mistral/Gemma expect BOS (attention sink)
        if len(ids) >= L:
            out.append(ids[:L])
            if len(out) >= n:
                break
    return out


def _cache_path(tok, tag):
    import hashlib
    h = hashlib.sha1((tok.name_or_path.split("/")[-1].split("-")[0] + str(len(tok))).encode()).hexdigest()[:8]
    return f"/vol/data/{tag}_{h}.pt"


def cached(tok, tag, build):
    """Token lists cached under /vol/data (per tokenizer family); avoids minutes of GPU-idle streaming."""
    import torch
    p = _cache_path(tok, tag)
    if os.path.exists(p):
        return torch.load(p)
    out = build()
    os.makedirs("/vol/data", exist_ok=True)
    torch.save(out, p)
    _commit()
    return out


def load_concat(tok, n, L, skip_docs=0):
    """n sequences of L tokens made by concatenating FineWeb-Edu docs with EOS between them."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    out, buf = [], []
    eos = tok.eos_token_id
    for i, ex in enumerate(ds):
        if i < skip_docs:
            continue
        buf += tok(ex["text"], add_special_tokens=False)["input_ids"] + [eos]
        while len(buf) >= L:
            out.append(buf[:L]); buf = buf[L:]
            if len(out) >= n:
                return out
    return out


def load_pg19_train(tok, n, L, per_book=8):
    """n calibration sequences of L tokens from PG19 TRAIN books (eval uses pg19-test), up to per_book chunks each."""
    from datasets import load_dataset
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    out = []
    for ex in ds:
        ids = tok(ex["text"][: L * per_book * 6], add_special_tokens=False)["input_ids"]
        if tok.bos_token_id is not None and "qwen" not in tok.name_or_path.lower():
            ids = [tok.bos_token_id] + ids
        for c in range(min(per_book, len(ids) // L)):
            out.append(ids[c * L:(c + 1) * L])
            if len(out) >= n:
                return out
    return out


def load_long(tok, n, Lx):
    """n sequences of Lx tokens: PG19 test books, else concatenated FineWeb-Edu docs."""
    from datasets import load_dataset
    out = []
    try:
        ds = load_dataset("emozilla/pg19-test", split="test")
        for ex in ds:
            ids = tok(ex["text"][: Lx * 8], add_special_tokens=False)["input_ids"]
            if len(ids) >= Lx:
                out.append(ids[:Lx])
                if len(out) >= n:
                    return out, "pg19-test"
    except Exception as e:
        print("pg19 unavailable:", e, flush=True)
    raise RuntimeError("not enough long docs")


def fit_eval_impl(src: str, tgt: str, n_calib: int = 500, L: int = 1024, stride: int = 4,
             ks: str = "1,2,4,8,all", lam: float = 0.01, n_eval: int = 64, P: int = 960,
             modes: str = "none,var,rms,g1.1,g1.2", run_id: str = "x", bs: int = 8,
             distill_steps: int = 0, distill_k: int = 0, rank: int = 16, distill_lr: float = 1e-3, distill_docs: int = 100, lam_rel: float = 1e-3, bench_n: int = 0,
             tails: str = "", tail_k: int = 0, bench_tail: int = 0,
             eval_L: int = 0, vks: str = "", mlp_epochs: int = 0, mlp_kinds: str = "v",
             feat: str = "kv", calib_concat: int = 0, save_maps: int = 0, sinks: str = "",
             layer_scan: int = 0, native_first: str = "", kspecs: str = "", lowranks: str = "", fp8sim: int = 0,
             src_stride: int = 1, calib_src: str = "fineweb", one_pass_: int = -1, vweight: float = 0.0, yarn: float = 0.0,
             maps_from: str = "", recompute: str = "", rd_rank: int = 128, rd_docs: int = 32,
             offload_maps: int = 0):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kvx import core
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"  # cpu: smoke tests only
    if maps_from:  # fail before the (long) model load: the file must exist and match this run
        import glob
        mf = sorted(glob.glob(f"/vol/runs/{maps_from}/maps_k*.pt"))
        if len(mf) != 1:
            raise RuntimeError(f"--maps-from {maps_from}: expected one maps_k*.pt, found {mf}")
        _d = torch.load(mf[0], map_location="cpu")
        bad = {k_: (_d.get(k_), v_) for k_, v_ in (("src", src), ("tgt", tgt), ("feat", feat)) if _d.get(k_) != v_}
        if bad:
            raise RuntimeError(f"--maps-from {maps_from} does not match this run: {bad}")
        if sinks and not tails:
            raise RuntimeError("sinks/sinkmean are scored inside the tails block: pass --tails")
        maps_file = mf[0]
        del _d
    t0 = time.time()
    from kvx.models import model_path
    tok = AutoTokenizer.from_pretrained(model_path(tgt))
    tok.name_or_path = tgt  # keep the repo id: token-cache keys and BOS rules read it
    from kvx.models import load_lm
    S = load_lm(src, yarn=yarn, device=dev).eval()
    dev_t = "cuda:1" if torch.cuda.device_count() > 1 else dev
    T = load_lm(tgt, yarn=yarn, device=dev_t).eval()
    if dev_t != dev:
        core.place_on(T, dev_t, "cuda:0")
    S.requires_grad_(False); T.requires_grad_(False)
    _commit()
    eval_L = eval_L or L
    cs, ss = core.rope_cos_sin(S, max(L, eval_L), dev)
    ct, st_ = core.rope_cos_sin(T, max(L, eval_L), dev_t)
    ct, st_ = ct.to(dev), st_.to(dev)
    hd = lambda c: getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    Hs, Ds = S.config.num_key_value_heads, hd(S.config)
    Ht, Dt = T.config.num_key_value_heads, hd(T.config)
    Ls, Lt = S.config.num_hidden_layers, T.config.num_hidden_layers
    cinfo = {}
    bos = int(tok.bos_token_id is not None and "qwen" not in tok.name_or_path.lower())
    calib = torch.tensor(cached(tok, f"pg19train_{n_calib}x{L}_bos{bos}", lambda: load_pg19_train(tok, n_calib, L))
                         if calib_src == "pg19" else
                         torch.cat([torch.tensor(cached(tok, f"calib_{n_calib // 2}x{L}_bos{bos}",
                                                        lambda: load_texts(tok, n_calib // 2, L, exclude=EVAL_DOCS))),
                                    torch.tensor(cached(tok, f"pg19train_{n_calib // 2}x{L}_bos{bos}",
                                                        lambda: load_pg19_train(tok, n_calib // 2, L, per_book=32)))]).tolist()
                         if calib_src == "mix" else
                         load_concat(tok, n_calib, L) if calib_concat else
                         cached(tok, f"calib_{n_calib}x{L}_bos{bos}",
                                lambda: load_texts(tok, n_calib, L, exclude=EVAL_DOCS, info=cinfo)))
    if eval_L > L:
        ev, eval_src = load_long(tok, n_eval, eval_L)
        evals = torch.tensor(ev)
        if tok.bos_token_id is not None and "qwen" not in tok.name_or_path.lower():
            evals[:, 0] = tok.bos_token_id
    else:
        evals = torch.tensor(cached(tok, f"eval_{n_eval}x{L}_bos{bos}", lambda: load_texts(tok, n_eval, L, skip_docs=20000)))
        eval_src = "fineweb-edu"
    t_data = time.time() - t0
    print(f"[{t_data:.0f}s] data+models loaded", flush=True)
    res = dict(calib_last_doc=cinfo.get("last_doc"), feat=feat, eval_L=eval_L, eval_src=eval_src, lam_rel=lam_rel, src=src, tgt=tgt, n_calib=len(calib), L=L, stride=stride, lam=lam, P=P, n_eval=len(evals),
               Ls=Ls, Lt=Lt, Hs=Hs, Ht=Ht, D=Dt, t_load=t_data)
    klist = [Ls if k == "all" else int(k) for k in ks.split(",")]
    maps = {}
    r2sum = {}
    t1 = time.time()
    dsrc = S.config.hidden_size if feat == "hidden" else (2 * Hs * Ds if feat == "kvcat" else Hs * Ds)
    one_pass = (one_pass_ == 1) if one_pass_ >= 0 else os.environ.get("KVX_ONE_PASS", "1") == "1"  # K and V stats from the same forwards
    stats = {}
    xstats = {}
    sinkmean = None
    if maps_from:  # reuse a saved fit (no calibration pass); sink template from the first tokens of the calib docs
        d = torch.load(maps_file)
        assert len(d["k"]) == Lt, (len(d["k"]), Lt)
        to = lambda ms: [{k_: (v_.to(dev) if torch.is_tensor(v_) else v_) for k_, v_ in m.items()} for m in ms]
        maps[("k", d["k_layers"])], maps[("v", d["k_layers"])] = to(d["k"]), to(d["v"])
        klist = [d["k_layers"]]
        res["maps_from"] = maps_from
        if sinks:
            A_ = max(int(x) for x in sinks.split(","))
            for i in range(0, len(calib), 64):
                ids = calib[i:i + 64, :A_].to(dev)
                with torch.no_grad():
                    Kt, Vt, _ = core.capture_kv(T, ids, ct, st_)
                sk = (Kt.float().sum(1), Vt.float().sum(1), ids.shape[0])
                sinkmean = sk if sinkmean is None else (sinkmean[0] + sk[0], sinkmean[1] + sk[1], sinkmean[2] + sk[2])
                del Kt, Vt
    for kind in (() if maps_from else ("k", "v")):
      if kind in stats:
        st = stats.pop(kind)
      else:
        kinds = ("k", "v") if one_pass else (kind,)
        sts = {kk: core.GramStats(Ls, dsrc, Lt, Ht * Dt, dev) for kk in kinds}
        cbs = max(1, bs * 1024 // L)  # keep calib batch at ~8K tokens
        for i in range(0, len(calib), cbs):
            ids = calib[i:i + cbs].to(dev)
            with torch.no_grad():
                Ks, Vs, Hs_ = core.capture_kv(S, ids, cs, ss, hidden=(feat == "hidden"))
                Kt, Vt, _ = core.capture_kv(T, ids, ct, st_)
                if sinks:  # calibration mean of the target's first-position K/V (sink template, input-invariant)
                    A_ = max(int(x) for x in sinks.split(","))
                    sk = (Kt[:, :, :, :A_].float().sum(1), Vt[:, :, :, :A_].float().sum(1), ids.shape[0])
                    sinkmean = sk if sinkmean is None else (sinkmean[0] + sk[0], sinkmean[1] + sk[1], sinkmean[2] + sk[2])
                wv = None
                if vweight > 0 and "v" in kinds:
                    wv = torch.cat([core.attention_received(T, ids[b:b + 1], power=vweight) for b in range(ids.shape[0])])
                    wv = wv[:, ::stride].reshape(-1)
            for kk in kinds:
                if feat == "kvcat":  # each source layer contributes its K and V
                    Xs = torch.cat([Ks, Vs], dim=2)[:, :, :, ::stride]
                else:
                    Xs = (Hs_ if feat == "hidden" else (Ks if kk == "k" else Vs))[:, :, :, ::stride]
                Yt = (Kt if kk == "k" else Vt)[:, :, :, ::stride]
                X = core.flat_tokens(Xs).reshape(-1, Ls * dsrc).float()
                Y = core.flat_tokens(Yt).reshape(-1, Lt * Ht * Dt).float()
                sts[kk].add(X, Y, wv if kk == "v" else None)
                del X, Y
            del Ks, Vs, Kt, Vt, Hs_
        st = sts.pop(kind)
        stats.update(sts)
      if True:
        print(f"[{time.time()-t0:.0f}s] {kind} stats done n={st.n}", flush=True)
        st.finalize()
        R2 = core.layer_r2(st, lam, Ht, lam_rel)
        if src_stride > 1:  # restrict candidate source layers (dense mapper cost ~ number of source layers used)
            keep = torch.zeros(Ls, dtype=torch.bool, device=R2.device)
            keep[src_stride - 1::src_stride] = True  # last layer of each stride group, ends at the top layer
            R2 = R2.masked_fill(~keep.view(1, 1, -1), -1e9)
        r2sum[kind] = dict(best_single_mean=R2.max(-1).values.mean().item(),
                           best_layer=R2.mean(1).argmax(-1).tolist())
        vk_list = [Ls if x == "all" else int(x) for x in vks.split(",")] if vks else []
        for k in sorted(set(klist + (vk_list if kind == "v" else []))):
            maps[(kind, k)] = core.fit_maps(st, R2, k, lam, Ht, lam_rel)
            if offload_maps:  # 32B on one 96 GB GPU: finished K maps (6.4 GB at k24) crowded the V solve
                maps[(kind, k)] = [{kk: (vv.cpu() if torch.is_tensor(vv) else vv) for kk, vv in m.items()}
                                   for m in maps[(kind, k)]]
            r2sum[kind][f"k{k}"] = torch.stack([m["r2"] for m in maps[(kind, k)]]).mean().item()
            r2sum[kind][f"k{k}_src_layers_used"] = len({s_ for m in maps[(kind, k)] for s_ in m["sel"]})
        print(f"[{time.time()-t0:.0f}s] {kind} maps fit {r2sum[kind]}", flush=True)
        xstats[kind] = (st.mx.clone(), (st.xx.diagonal() / st.n).clamp(min=1e-12).sqrt().clone())
        del st
        torch.cuda.empty_cache()
    # layer-adaptive k: spec "split/klo/khi" -> k=klo for target layers < split, khi otherwise (stitched)
    for spec in (kspecs.split(",") if kspecs else []):
        sp, klo, khi = [int(x) for x in spec.split("/")]
        for kind in ("k", "v"):
            maps[(kind, spec)] = [maps[(kind, klo if t < sp else khi)][t] for t in range(Lt)]
    res["t_fit"] = time.time() - t1
    if save_maps:
        sk = tail_k or klist[min(3, len(klist) - 1)]
        cpu = lambda ms: [{k: (v.cpu() if torch.is_tensor(v) else v) for k, v in m.items()} for m in ms]
        os.makedirs(f"/vol/runs/{run_id}", exist_ok=True)
        # save_maps=2: every k of --ks (same Gram stats, free), e.g. small-k maps for the fast serving mapper;
        # the tail k keeps the plain name maps_k{sk}.pt, others go to maps_all/ so --maps-from still finds one file
        for kk in ([sk] + [x for x in klist if x != sk and isinstance(x, int)] if save_maps >= 2 else [sk]):
            if ("k", kk) not in maps:
                continue
            path = f"/vol/runs/{run_id}/maps_k{kk}.pt" if kk == sk else f"/vol/runs/{run_id}/maps_all/maps_k{kk}.pt"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({"k": cpu(maps[("k", kk)]), "v": cpu(maps[("v", kk)]), "src": src, "tgt": tgt, "k_layers": kk,
                        "feat": feat}, path)
        _commit()
    if offload_maps:  # stats are freed now: bring the maps back for the evaluation
        torch.cuda.empty_cache()
        maps = {key: [{kk: (vv.to(dev) if torch.is_tensor(vv) else vv) for kk, vv in m.items()} for m in ms]
                for key, ms in maps.items()}
    res["r2"] = r2sum
    kw = None
    kw_full = None
    try:
        kw = [l.self_attn.k_norm.weight.detach().to(dev) for l in T.model.layers]
        if T.config.model_type.startswith("gemma"):  # Gemma (1 + w) k_norm: rms restore gave NaN -> disabled
            kw = None
        if kw is not None and kw[0].numel() != Dt:  # OLMo-2 style k_norm over all heads: only "rmsfull" applies
            kw_full = kw
            kw = None
    except AttributeError:
        pass
    mode_list = [m for m in modes.split(",") if not (m == "rms" and kw is None) and not (m in ("rmsfull", "rmsfullc") and kw_full is None)]
    # low-rank dense mapper: W_dense [Ls*ds, Lt*dt] ~ (U S) V^T, mapper FLOPs T*r*(Ls*ds + Lt*dt)
    lowrank = {}
    if lowranks:
        lk = tail_k or klist[min(3, len(klist) - 1)]
        for kind in ("k", "v"):
            ms = maps[(kind, lk)]
            ds_, dt_ = ms[0]["W"].shape[0] // len(ms[0]["sel"]), ms[0]["W"].shape[1]
            Wd = torch.zeros(Ls * ds_, Lt * dt_, device=dev)
            for t, m in enumerate(ms):
                for j, s_ in enumerate(m["sel"]):
                    Wd[s_ * ds_:(s_ + 1) * ds_, t * dt_:(t + 1) * dt_] = m["W"][j * ds_:(j + 1) * ds_]
            b = torch.cat([m["b"] for m in ms])
            mx, sx = xstats[kind]
            # data-aware: center inputs (mean folded into the bias exactly) and scale rows by input std before SVD
            b_c = b + mx @ Wd
            Wsc = Wd * sx[:, None]
            for r in [int(x) for x in lowranks.split(",")]:
                U, S_, V = torch.svd_lowrank(Wsc, q=r, niter=4)
                A = (U * S_) / sx[:, None]  # maps centered x -> rank-r code
                lowrank[(kind, r)] = (A.contiguous(), V.T.contiguous(), b_c, mx)
            del Wd
        res["lowrank_k"] = lk
    res_dist = None
    mlp_maps = {}
    if mlp_epochs > 0:
        from kvx import mlp as kmlp
        mk_ = tail_k or klist[min(3, len(klist) - 1)]
        t5 = time.time()
        for kind in mlp_kinds.split(","):
            Xc, Yc = [], []
            with torch.no_grad():
                cbs = max(1, bs * 1024 // L)
                for i in range(0, len(calib), cbs):
                    ids = calib[i:i + cbs].to(dev)
                    Ks, Vs, _ = core.capture_kv(S, ids, cs, ss)
                    Kt, Vt, _ = core.capture_kv(T, ids, ct, st_)
                    Xs = (Ks if kind == "k" else Vs)[:, :, :, ::stride]
                    Yt = (Kt if kind == "k" else Vt)[:, :, :, ::stride]
                    Xc.append(core.flat_tokens(Xs).to(torch.bfloat16).cpu())
                    Yc.append(core.flat_tokens(Yt).to(torch.bfloat16).cpu())
                    del Ks, Vs, Kt, Vt
            Xc, Yc = torch.cat(Xc), torch.cat(Yc)
            nets, mst = kmlp.train_mlps(Xc, Yc, maps[(kind, mk_)], epochs=mlp_epochs,
                                        log=lambda m: print(m, flush=True))
            mlp_maps[kind] = nets
            res.setdefault("mlp", {})[kind] = dict(k=mk_, epochs=mlp_epochs,
                                                    val_r2_ridge=sum(a for a, _ in mst) / len(mst),
                                                    val_r2_mlp=sum(b for _, b in mst) / len(mst))
            del Xc, Yc
        res["mlp"]["t"] = time.time() - t5
        print(f"[{time.time()-t0:.0f}s] mlp {res['mlp']}", flush=True)
    if distill_steps > 0:
        try:
            from kvx import distill
            dk = distill_k or klist[min(3, len(klist) - 1)]
            ddocs = torch.tensor(load_texts(tok, distill_docs, L, skip_docs=30000)).to(dev)
            t3 = time.time()
            rk, rv, hist = distill.train(S, T, maps[("k", dk)], maps[("v", dk)], ddocs, P, cs, ss, ct, st_,
                                         rank=rank, steps=distill_steps, lr=distill_lr, bs=4,
                                         restore=(lambda K: core.restore_scale(K, maps[("k", dk)], "rms", Ht, kw)) if kw is not None else None)
            res_dist = (dk, rk, rv)
            res["distill"] = dict(k=dk, rank=rank, steps=distill_steps, lr=distill_lr, docs=len(ddocs),
                                  t=time.time() - t3, loss_first20=sum(hist[:20]) / 20,
                                  loss_last20=sum(hist[-20:]) / 20)
        except Exception:
            import traceback
            res["distill_error"] = traceback.format_exc()[-2000:]
            torch.cuda.empty_cache()
    rd_basis = None
    if recompute:  # RaReCache rank-disagreement basis from mapped predictions on calibration docs (never the target)
        from kvx.recompute import PredBasis
        rk = tail_k or klist[min(3, len(klist) - 1)]
        rd_basis = PredBasis()
        with torch.no_grad():
            for i in range(0, min(rd_docs, len(calib)), 4):
                ids_ = calib[i:i + 4, :P].to(dev)
                Ks_, Vs_, _ = core.capture_kv(S, ids_, cs, ss)
                if feat == "kvcat":
                    Ks_ = Vs_ = torch.cat([Ks_, Vs_], dim=2)
                rd_basis.add("k", core.apply_maps(maps[("k", rk)], Ks_))
                rd_basis.add("v", core.apply_maps(maps[("v", rk)], Vs_))
                del Ks_, Vs_
        rd_basis.finalize(rd_rank)
        print(f"[{time.time()-t0:.0f}s] rank-disagreement basis r={rd_rank} from {rd_basis.n} tokens", flush=True)
    if sinkmean is not None:  # serving artifact: calibration-mean target K (pre-RoPE) / V at the first A positions
        os.makedirs(f"/vol/runs/{run_id}", exist_ok=True)
        torch.save({"K": (sinkmean[0] / sinkmean[2]).cpu(), "V": (sinkmean[1] / sinkmean[2]).cpu(), "tgt": tgt},
                   f"/vol/runs/{run_id}/sink_template.pt")
        _commit()
    # ---- eval
    t2 = time.time()
    acc = {}

    def add(name, kl, agree):
        a = acc.setdefault(name, dict(kl=0.0, kl_first=0.0, agree=0.0, agree_first=0.0, n=0))
        a["kl"] += kl.mean(1).sum().item(); a["kl_first"] += kl[:, 0].sum().item()
        a["agree"] += agree.float().mean(1).sum().item(); a["agree_first"] += agree[:, 0].float().sum().item()
        a["n"] += kl.shape[0]

    ebs = max(1, bs * 1024 // eval_L)
    if eval_L != L:
        P = eval_L - (1024 - P); res["P"] = P  # continuation length fixed at 1024-P (64), independent of calib L
    with torch.no_grad():
        for i in range(0, len(evals), ebs):
            ids = evals[i:i + ebs].to(dev)
            Ks, Vs, Hp = core.capture_kv(S, ids[:, :P], cs, ss, hidden=(feat == "hidden"))
            if feat == "hidden":
                Ks = Vs = Hp  # mapper inputs are the residual stream for both K and V
            elif feat == "kvcat":
                Ks = Vs = torch.cat([Ks, Vs], dim=2)
            Vmin = min(S.config.vocab_size, T.config.vocab_size)  # padded vocabs differ across sizes (Qwen2.5)
            lg_t = T(input_ids=ids, logits_to_keep=eval_L - P).logits[..., :Vmin].float()
            lp = torch.log_softmax(lg_t, -1)
            top = lg_t.argmax(-1)
            def score(name, logits):
                logits = logits[..., :Vmin]
                add(name, core.kl_rows(lp, logits), logits.argmax(-1) == top)
            score("ref_nocontext", T(input_ids=ids[:, P:]).logits)
            score("ref_source_model", S(input_ids=ids, logits_to_keep=eval_L - P).logits)
            for k in klist:
                Km = core.apply_maps(maps[("k", k)], Ks)
                Vm = core.apply_maps(maps[("v", k)], Vs)
                if eval_L > 8192:
                    Vm = Vm.to(torch.bfloat16)
                for mode in mode_list:
                    Kr = core.restore_scale(Km, maps[("k", k)], mode, Ht, kw_full if mode in ("rmsfull", "rmsfullc") else kw)
                    if eval_L > 8192:
                        Kr = Kr.to(torch.bfloat16)
                    c = core.build_cache(Kr, Vm, ct, st_)
                    del Kr
                    score(f"map_k{k}_{mode}", T(input_ids=ids[:, P:], past_key_values=c).logits)
                    del c
                # oracle splits: exact K + mapped V, mapped K + exact V (skipped for long evals: memory)
                if k == klist[min(3, len(klist) - 1)] and eval_L <= 8192:
                    Kt, Vt, _ = core.capture_kv(T, ids[:, :P], ct, st_)
                    score(f"oracleK_mapV_k{k}", T(input_ids=ids[:, P:], past_key_values=core.build_cache(Kt.float(), Vm, ct, st_)).logits)
                    score(f"mapK_oracleV_k{k}", T(input_ids=ids[:, P:], past_key_values=core.build_cache(Km, Vt, ct, st_)).logits)
                    score("oracle_roundtrip", T(input_ids=ids[:, P:], past_key_values=core.build_cache(Kt.float(), Vt, ct, st_)).logits)
                del Km, Vm
            for vk in ([Ls if x == "all" else int(x) for x in vks.split(",")] if vks else []):
                tk = tail_k or klist[min(3, len(klist) - 1)]
                tmode = "rms" if kw is not None else "none"
                Km = core.restore_scale(core.apply_maps(maps[("k", tk)], Ks), maps[("k", tk)], tmode, Ht, kw)
                Vm = core.apply_maps(maps[("v", vk)], Vs)
                score(f"map_k{tk}_{tmode}_vk{vk}", T(input_ids=ids[:, P:], past_key_values=core.build_cache(Km, Vm, ct, st_)).logits)
                del Km, Vm
            if mlp_maps:
                from kvx import mlp as kmlp
                tk = tail_k or klist[min(3, len(klist) - 1)]
                tmode = "rms" if kw is not None else "none"
                Kb = core.apply_maps(maps[("k", tk)], Ks) if "k" not in mlp_maps else kmlp.apply_mlp_maps(maps[("k", tk)], mlp_maps["k"], Ks)
                Km = core.restore_scale(Kb, maps[("k", tk)], tmode, Ht, kw)
                Vm = core.apply_maps(maps[("v", tk)], Vs) if "v" not in mlp_maps else kmlp.apply_mlp_maps(maps[("v", tk)], mlp_maps["v"], Vs)
                tag = "mlp" + "".join(sorted(mlp_maps))
                score(f"map_k{tk}_{tmode}_{tag}", T(input_ids=ids[:, P:], past_key_values=core.build_cache(Km, Vm, ct, st_)).logits)
                for N in ([int(x) for x in tails.split(",")] if tails else []):
                    c = core.build_cache(Km[:, :, :, :P - N].contiguous(), Vm[:, :, :, :P - N].contiguous(), ct, st_)
                    score(f"map_k{tk}_{tmode}_{tag}_tail{N}", T(input_ids=ids[:, P - N:], past_key_values=c).logits[:, N:])
                del Km, Vm, Kb
            if tails:
                tk = tail_k or klist[min(3, len(klist) - 1)]
                tmode = "rms" if kw is not None else "none"
                Km = core.restore_scale(core.apply_maps(maps[("k", tk)], Ks), maps[("k", tk)], tmode, Ht, kw)
                Vm = core.apply_maps(maps[("v", tk)], Vs)
                if eval_L > 8192:
                    Km, Vm = Km.to(torch.bfloat16), Vm.to(torch.bfloat16)
                    torch.cuda.empty_cache()
                for N in [int(x) for x in tails.split(",")]:
                    c = core.build_cache(Km[:, :, :, :P - N].contiguous(), Vm[:, :, :, :P - N].contiguous(), ct, st_)
                    score(f"map_k{tk}_{tmode}_tail{N}", T(input_ids=ids[:, P - N:], past_key_values=c).logits[:, N:])
                    del c
                if layer_scan or native_first:
                    Ktn, Vtn, _ = core.capture_kv(T, ids[:, :P], ct, st_)
                    if layer_scan:
                        for l in range(Lt):
                            Kc, Vc = Km.clone(), Vm.clone()
                            Kc[l] = Ktn[l].float(); Vc[l] = Vtn[l].float()
                            score(f"scan_exact_L{l}", T(input_ids=ids[:, P:], past_key_values=core.build_cache(Kc, Vc, ct, st_)).logits)
                            del Kc, Vc
                    for m in ([int(x) for x in native_first.split(",")] if native_first else []):
                        Kc, Vc = Km.clone(), Vm.clone()
                        Kc[:m] = Ktn[:m].float(); Vc[:m] = Vtn[:m].float()
                        for N in (0, 64):
                            c = core.build_cache(Kc[:, :, :, :P - N].contiguous(), Vc[:, :, :, :P - N].contiguous(), ct, st_)
                            score(f"map_k{tk}_{tmode}_nativefirst{m}_tail{N}", T(input_ids=ids[:, P - N:], past_key_values=c).logits[:, N:])
                        del Kc, Vc
                    del Ktn, Vtn
                if fp8sim:
                    Kq = core.restore_scale(core.apply_maps_fp8sim(maps[("k", tk)], Ks), maps[("k", tk)], tmode, Ht, kw)
                    Vq = core.apply_maps_fp8sim(maps[("v", tk)], Vs)
                    for N in (0, 64):
                        c = core.build_cache(Kq[:, :, :, :P - N].contiguous(), Vq[:, :, :, :P - N].contiguous(), ct, st_)
                        score(f"map_k{tk}_{tmode}_fp8sim_tail{N}", T(input_ids=ids[:, P - N:], past_key_values=c).logits[:, N:])
                    del Kq, Vq
                for r in ([int(x) for x in lowranks.split(",")] if lowranks else []):
                    outs = []
                    for kind, Xs_ in (("k", Ks), ("v", Vs)):
                        A, Bm, b, mx = lowrank[(kind, r)]
                        X = core.flat_tokens(Xs_).float().reshape(Ks.shape[1] * Ks.shape[3], -1) - mx
                        Y = (X @ A) @ Bm + b
                        outs.append(Y.view(Ks.shape[1], Ks.shape[3], Lt, Ht, Dt).permute(2, 0, 3, 1, 4))
                    Klr = core.restore_scale(outs[0], maps[("k", res["lowrank_k"])], tmode, Ht, kw)
                    for N in (0, 64):
                        c = core.build_cache(Klr[:, :, :, :P - N].contiguous(), outs[1][:, :, :, :P - N].contiguous(), ct, st_)
                        score(f"map_k{res['lowrank_k']}_{tmode}_lr{r}_tail{N}", T(input_ids=ids[:, P - N:], past_key_values=c).logits[:, N:])
                    del outs, Klr
                for spec in (kspecs.split(",") if kspecs else []):
                    Ka = core.restore_scale(core.apply_maps(maps[("k", spec)], Ks), maps[("k", spec)], tmode, Ht, kw)
                    Va = core.apply_maps(maps[("v", spec)], Vs)
                    tag = spec.replace("/", "-")
                    for N in (0, 64):
                        c = core.build_cache(Ka[:, :, :, :P - N].contiguous(), Va[:, :, :, :P - N].contiguous(), ct, st_)
                        score(f"map_kA{tag}_{tmode}_tail{N}", T(input_ids=ids[:, P - N:], past_key_values=c).logits[:, N:])
                    del Ka, Va
                for A in ([int(x) for x in sinks.split(",")] if sinks else []):
                    # first A tokens computed natively by the target (attention sinks), rest mapped
                    Ka, Va, _ = core.capture_kv(T, ids[:, :A], ct, st_)
                    Kc, Vc = Km.clone(), Vm.clone()
                    Kc[:, :, :, :A] = Ka.float(); Vc[:, :, :, :A] = Va.float()
                    for N in [0] + ([int(x) for x in tails.split(",")][-1:]):
                        c = core.build_cache(Kc[:, :, :, :P - N].contiguous(), Vc[:, :, :, :P - N].contiguous(), ct, st_)
                        score(f"map_k{tk}_{tmode}_sink{A}_tail{N}", T(input_ids=ids[:, P - N:], past_key_values=c).logits[:, N:])
                    del Kc, Vc
                    if sinkmean is not None:  # no target prefill at all: calibration-mean sink template
                        n_ = sinkmean[2]
                        mK = (sinkmean[0][:, :, :A] / n_).to(Km.device)[:, None]
                        mV = (sinkmean[1][:, :, :A] / n_).to(Vm.device)[:, None]
                        for which in ("kv", "v"):
                            Kc, Vc = Km.clone(), Vm.clone()
                            Vc[:, :, :, :A] = mV.to(Vc.dtype)
                            if which == "kv":
                                Kc[:, :, :, :A] = mK.to(Kc.dtype)
                            c = core.build_cache(Kc.contiguous(), Vc.contiguous(), ct, st_)
                            score(f"map_k{tk}_{tmode}_sinkmean{which}{A}", T(input_ids=ids[:, P:], past_key_values=c).logits)
                            if which == "kv" and tails:  # deployable config: template sinks + native tail
                                N = int(tails.split(",")[-1])
                                c = core.build_cache(Kc[:, :, :, :P - N].contiguous(), Vc[:, :, :, :P - N].contiguous(), ct, st_)
                                score(f"map_k{tk}_{tmode}_sinkmeankv{A}_tail{N}",
                                      T(input_ids=ids[:, P - N:], past_key_values=c).logits[:, N:])
                            del Kc, Vc
                if rd_basis is not None:  # selective native recompute at budget rho: rank-disagreement vs last-N vs random
                    from kvx.recompute import rank_disagreement, select_top, recompute_cache
                    Kp, Vp = Km[:, :, :, :P].contiguous(), Vm[:, :, :, :P].contiguous()
                    rs = rank_disagreement(rd_basis, {"k": core.apply_maps(maps[("k", tk)], Ks), "v": Vp.float()})
                    gen = torch.Generator(device="cpu").manual_seed(i)
                    for rho in [float(x) for x in recompute.split(",")]:
                        n_ = max(1, int(-(-rho * P // 1)))
                        sels = {"rd": select_top(rs, rho), "rdS4": select_top(rs, rho, keep_first=4),
                                "last": torch.arange(P - n_, P, device=ids.device)[None].expand(ids.shape[0], n_),
                                "rand": torch.stack([torch.randperm(P, generator=gen)[:n_].sort().values
                                                     for _ in range(ids.shape[0])]).to(ids.device)}
                        for nm, sel in sels.items():
                            c = recompute_cache(T, ids[:, :P], Kp, Vp, sel.contiguous(), ct, st_)
                            score(f"map_k{tk}_{tmode}_rc{nm}{rho}", T(input_ids=ids[:, P:], past_key_values=c).logits)
                            del c
                    del Kp, Vp, rs
                del Km, Vm
            if res_dist is not None:
                dk, rk, rv = res_dist
                Kd = rk.apply(maps[("k", dk)], Ks).detach()
                if kw is not None:
                    Kd = core.restore_scale(Kd, maps[("k", dk)], "rms", Ht, kw)
                score(f"map_k{dk}_distill_r{rank}_rms", T(input_ids=ids[:, P:], past_key_values=core.build_cache(
                    Kd, rv.apply(maps[("v", dk)], Vs).detach(), ct, st_)).logits)
    for a in acc.values():
        n = a.pop("n")
        for key in a:
            a[key] /= n
    res["eval"] = acc
    res["t_eval"] = time.time() - t2
    if bench_n > 0:
        from kvx import bench
        t4 = time.time()
        cs4, ss4 = core.rope_cos_sin(S, 4096, dev)
        ct4, st4 = [x.to(dev) for x in core.rope_cos_sin(T, 4096, dev_t)]
        bk = klist[min(3, len(klist) - 1)]
        def mk(mode, k=bk):
            def f(Ks, Vs):
                if feat == "kvcat":
                    Ks = Vs = torch.cat([Ks, Vs], dim=2)
                return (core.restore_scale(core.apply_maps(maps[("k", k)], Ks), maps[("k", k)], mode, Ht, kw),
                        core.apply_maps(maps[("v", k)], Vs))
            return f
        methods = {"native": "native", "source": "source", f"map_k{bk}_none": mk("none")}
        if kw is not None:
            methods[f"map_k{bk}_rms"] = mk("rms")
        if res_dist is not None:
            dk, rk, rv = res_dist
            methods[f"map_k{dk}_distill"] = lambda Ks, Vs: (rk.apply(maps[("k", dk)], Ks), rv.apply(maps[("v", dk)], Vs))
        res["bench"] = {}
        for name, items, norm in (("hellaswag", bench.hellaswag_items(bench_n), True),
                                  ("mmlu5", bench.mmlu_items(bench_n), False)):
            try:
                res["bench"][name] = bench.run(tok, S, T, items, methods, cs4, ss4, ct4, st4, dev, norm=norm)
                if bench_tail > 0:
                    tm = {f"map_k{bk}_rms_tail{bench_tail}": methods[f"map_k{bk}_rms"]} if kw is not None else {f"map_k{bk}_none_tail{bench_tail}": methods[f"map_k{bk}_none"]}
                    res["bench"][name].update(bench.run(tok, S, T, items, tm, cs4, ss4, ct4, st4, dev, norm=norm, tail=bench_tail))
            except Exception:
                import traceback
                res["bench"][name + "_error"] = traceback.format_exc()[-1500:]
        res["t_bench"] = time.time() - t4
    import torch as _t
    res["gpu"] = _t.cuda.get_device_name(0) if _t.cuda.is_available() else "cpu"
    res["t_total"] = time.time() - t0
    os.makedirs(f"/vol/runs/{run_id}", exist_ok=True)
    json.dump(res, open(f"/vol/runs/{run_id}/result.json", "w"), indent=1)
    _commit()
    return res


def fit_eval_safe(**kw):
    """Re-raise remote errors as plain RuntimeError so the local CLI (no torch) can show them."""
    try:
        return fit_eval_impl(**kw)
    except Exception:
        import traceback
        raise RuntimeError(traceback.format_exc()[-3000:]) from None


fit_eval = app.function(gpu=GPU_SPEC, volumes={"/vol": vol, "/hf": hf_shared.read_only()}, timeout=3300, memory=98304)(fit_eval_safe)



@app.local_entrypoint()
def main(src: str = "Qwen/Qwen3-1.7B", tgt: str = "Qwen/Qwen3-8B", n_calib: int = 500, ks: str = "1,2,4,8,all",
         n_eval: int = 64, modes: str = "none,var,rms,g1.1,g1.2", run_id: str = "", lam: float = 0.01, prefix: int = 960,
         distill_steps: int = 0, distill_k: int = 0, rank: int = 16, distill_lr: float = 1e-3, lam_rel: float = 1e-3,
         bench_n: int = 0, tails: str = "", tail_k: int = 0, bench_tail: int = 0, eval_l: int = 0,
         distill_docs: int = 100, vks: str = "", mlp_epochs: int = 0, mlp_kinds: str = "v", feat: str = "kv",
         calib_l: int = 1024, calib_concat: int = 0, save_maps: int = 0, sinks: str = "",
         layer_scan: int = 0, native_first: str = "", kspecs: str = "", lowranks: str = "", fp8sim: int = 0,
         src_stride: int = 1, calib_src: str = "fineweb", one_pass: int = -1, vweight: float = 0.0, yarn: float = 0.0,
         maps_from: str = "", recompute: str = "", rd_rank: int = 128, rd_docs: int = 32,
         offload_maps: int = 0):
    run_id = run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    r = fit_eval.remote(src=src, tgt=tgt, n_calib=n_calib, ks=ks, n_eval=n_eval, modes=modes, run_id=run_id, lam=lam, P=prefix,
                         distill_steps=distill_steps, distill_k=distill_k, rank=rank, distill_lr=distill_lr, lam_rel=lam_rel, bench_n=bench_n,
                         tails=tails, tail_k=tail_k, bench_tail=bench_tail,
                         eval_L=eval_l, distill_docs=distill_docs, vks=vks,
                         mlp_epochs=mlp_epochs, mlp_kinds=mlp_kinds, feat=feat, L=calib_l, calib_concat=calib_concat, save_maps=save_maps, sinks=sinks,
                         layer_scan=layer_scan, native_first=native_first, kspecs=kspecs, lowranks=lowranks, fp8sim=fp8sim, src_stride=src_stride, calib_src=calib_src, one_pass_=one_pass, vweight=vweight, yarn=yarn, maps_from=maps_from, recompute=recompute, rd_rank=rd_rank, rd_docs=rd_docs,
                         offload_maps=offload_maps)
    os.makedirs(f"out/{run_id}", exist_ok=True)
    json.dump(r, open(f"out/{run_id}/result.json", "w"), indent=1)
    print(json.dumps(r, indent=1))


def prime_impl(tok_name: str, ncalibs: str = "500,2000,4000", n_eval: int = 64, L: int = 1024):
    """CPU job: fill /vol/data token caches so GPU jobs skip streaming."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_name)
    bos = int(tok.bos_token_id is not None and "qwen" not in tok.name_or_path.lower())
    out = {}
    for n in [int(x) for x in ncalibs.split(",")]:
        info = {}
        c = cached(tok, f"calib_{n}x{L}_bos{bos}", lambda: load_texts(tok, n, L, exclude=EVAL_DOCS, info=info))
        out[f"calib_{n}"] = (len(c), info.get("last_doc"))
    e = cached(tok, f"eval_{n_eval}x{L}_bos{bos}", lambda: load_texts(tok, n_eval, L, skip_docs=20000))
    out["eval"] = len(e)
    return out


prime = app.function(volumes={"/vol": vol}, timeout=3300, cpu=4.0, memory=16384)(prime_impl)


@app.local_entrypoint()
def prime_main(tok_name: str = "Qwen/Qwen3-8B", ncalibs: str = "500,2000,4000"):
    print(prime.remote(tok_name, ncalibs))
