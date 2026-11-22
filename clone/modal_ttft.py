"""TTFT on Modal: KVX_GPU=H100 modal run modal_ttft.py --maps-run q17-q32 --k 16 --lens 4096,8192,16384"""
import json, os, time
try:
    import modal
except ImportError:  # rented box (run_local.py): no-op stand-in
    from modal_fit_eval import modal

GPU = os.environ.get("KVX_GPU", "H100")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "transformers==4.56.2", "accelerate", "numpy", "hf_transfer", "packaging", "ninja", "wheel")
    .pip_install("https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_CACHE": "/hf/hub", "HF_HUB_OFFLINE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .env({k: v for k, v in __import__("os").environ.items() if k.startswith("KVX_") and k.endswith(("_VOLUME", "_SECRET"))})
    .add_local_python_source("kvx")
)
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"), create_if_missing=True)
hf_shared = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))  # shared weights volume (KVX_WEIGHTS_VOLUME), read-only
app = modal.App(os.environ.get("KVX_APP", "kvx-ttft"), image=image)


@app.function(gpu=GPU.split(",") if "," in GPU else GPU, volumes={"/vol": vol, "/hf": hf_shared.read_only()}, timeout=3000)
def ttft(src: str, tgt: str, maps_run: str, k: int, lens: str, tail: int, dense: int = 0, profile: int = 0,
         blocks: int = 0, attn: str = "sdpa"):
    import sys
    from kvx import ttft as tt
    out = f"/vol/runs/{maps_run}/ttft_{GPU.replace(',', '_')}_tail{tail}_d{dense}_b{blocks}_{attn}.json"
    sys.argv = ["ttft", "--src", src, "--tgt", tgt, "--maps", f"/vol/runs/{maps_run}/maps_k{k}.pt",
                "--lens", lens, "--tail", str(tail), "--out", out, "--dense", str(dense), "--profile", str(profile),
                "--blocks", str(blocks), "--attn", attn]
    try:
        tt.main()
    except Exception:
        import traceback
        raise RuntimeError(traceback.format_exc()[-3000:]) from None
    vol.commit()
    return json.load(open(out))


@app.local_entrypoint()
def main(src: str = "Qwen/Qwen3-1.7B", tgt: str = "Qwen/Qwen3-32B", maps_run: str = "q17-q32", k: int = 16,
         lens: str = "4096,8192,16384", tail: int = 64, dense: int = 0, profile: int = 0,
         blocks: int = 0, attn: str = "sdpa"):
    r = ttft.remote(src, tgt, maps_run, k, lens, tail, dense, profile, blocks, attn)
    os.makedirs(f"out/{maps_run}", exist_ok=True)
    json.dump(r, open(f"out/{maps_run}/ttft_{GPU.replace(',', '_')}_tail{tail}_d{dense}_b{blocks}_{attn}.json", "w"), indent=1)
    print(json.dumps(r, indent=1))
