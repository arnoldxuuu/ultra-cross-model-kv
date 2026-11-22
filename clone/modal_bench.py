"""Benchmarks from saved maps (no refit): KVX_GPU=B200:2 modal run modal_bench.py --maps-run l8-l70 --k 20"""
import json, os
try:
    import modal
except ImportError:  # rented box (run_local.py): no-op stand-in
    from modal_fit_eval import modal

GPU = os.environ.get("KVX_GPU", "L40S")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "transformers==4.56.2", "accelerate", "datasets==3.6.0", "numpy", "hf_transfer")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .env({k: v for k, v in __import__("os").environ.items() if k.startswith("KVX_") and k.endswith(("_VOLUME", "_SECRET"))})
    .add_local_python_source("kvx")
)
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"), create_if_missing=True)
hf_shared = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))  # shared weights volume (KVX_WEIGHTS_VOLUME), read-only
app = modal.App(os.environ.get("KVX_APP", "kvx-bench"), image=image)


@app.function(gpu=GPU.split(",") if "," in GPU else GPU, volumes={"/vol": vol, "/hf": hf_shared.read_only()}, timeout=3300)
def run(maps_run: str, k: int, bench_n: int, tail: int, mode: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from kvx import core, bench
        torch.set_grad_enabled(False)
        mp = torch.load(f"/vol/runs/{maps_run}/maps_k{k}.pt", map_location="cuda:0")
        src, tgt = mp["src"], mp["tgt"]
        from kvx.models import load_lm, model_path
        tok = AutoTokenizer.from_pretrained(model_path(tgt)); tok.name_or_path = tgt
        S = load_lm(src).to("cuda:0").eval()
        dev_t = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        T = load_lm(tgt).to(dev_t).eval()
        if dev_t != "cuda:0":
            core.place_on(T, dev_t, "cuda:0")
        try:
            kw = [l.self_attn.k_norm.weight.detach().to("cuda:0") for l in T.model.layers]
        except AttributeError:
            kw = None
        H = T.config.num_key_value_heads
        cs, ss = core.rope_cos_sin(S, 4096, "cuda:0")
        ct, st = [x.to("cuda:0") for x in core.rope_cos_sin(T, 4096, dev_t)]
        m = lambda Ks, Vs: (core.restore_scale(core.apply_maps(mp["k"], Ks), mp["k"], mode, H, kw), core.apply_maps(mp["v"], Vs))
        methods = {"native": "native", "source": "source", f"map_k{k}_{mode}": m}
        res = dict(src=src, tgt=tgt, k=k, mode=mode, n=bench_n)
        for name, items, norm in (("hellaswag", bench.hellaswag_items(bench_n), True), ("mmlu5", bench.mmlu_items(bench_n), False)):
            r = bench.run(tok, S, T, items, methods, cs, ss, ct, st, "cuda:0", norm=norm)
            r.update(bench.run(tok, S, T, items, {f"map_k{k}_{mode}_tail{tail}": m}, cs, ss, ct, st, "cuda:0", norm=norm, tail=tail))
            res[name] = r
            print(name, r, flush=True)
        json.dump(res, open(f"/vol/runs/{maps_run}/bench_k{k}_{mode}.json", "w"), indent=1)
        vol.commit()
        return res
    except Exception:
        import traceback
        raise RuntimeError(traceback.format_exc()[-3000:]) from None


@app.local_entrypoint()
def main(maps_run: str = "l8-l70", k: int = 20, bench_n: int = 200, tail: int = 64, mode: str = "none"):
    r = run.remote(maps_run, k, bench_n, tail, mode)
    os.makedirs(f"out/{maps_run}", exist_ok=True)
    json.dump(r, open(f"out/{maps_run}/bench_k{k}_{mode}.json", "w"), indent=1)
    print(json.dumps(r, indent=1))
