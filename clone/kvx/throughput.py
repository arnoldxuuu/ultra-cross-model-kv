"""Prefill throughput and $/1M prompt tokens at concurrency c > 1, native vLLM vs the cross-model connector, same box.

Closed batches: c prompts of `length` random tokens go in one generate() call (c requests in flight), repeated `reps`
times after one warm-up batch. Throughput = prompt tokens / wall time. max_new=1 measures prefill (the connector's job);
max_new>1 adds decode, which is the target's own work either way. $/1M = box $/h / (tokens/s * 3600) * 1e6.

/venv-serve/bin/python run_local.py kvx.throughput:throughput_impl --json '{"tgt": "Qwen/Qwen3-8B", "mode": "mapped",
  "src": "Qwen/Qwen3-1.7B", "maps": "/vol/runs/q17-q8-mix/maps_k24.pt", "grid": "8192:1,4,16;16384:1,4,16", ...}'
"""
import json
import os
import random
import time


def parse_grid(grid):
    """'8192:1,4,16;32768:1,4' -> [(8192, 1), (8192, 4), (8192, 16), (32768, 1), (32768, 4)]"""
    out = []
    for part in grid.split(";"):
        if not part.strip():
            continue
        n, cs = part.split(":")
        out += [(int(n), int(c)) for c in cs.split(",")]
    return out


def usd_per_m(tok_s, usd_h):
    return None if not tok_s else usd_h / (tok_s * 3600.0) * 1e6


def throughput_impl(tgt: str, mode: str = "native", src: str = "", maps: str = "", sink: str = "",
                    grid: str = "8192:1,4,16", reps: int = 3, max_new: int = 1, tail: int = 64, min_len: int = 512,
                    util: float = 0.70, chunk: int = 8, usd_h: float = 2.31, tag: str = "", seed: int = 0,
                    vocab_lo: int = 1000, vocab_hi: int = 30000):
    from vllm import SamplingParams
    from .oteval import make_llm
    pts = parse_grid(grid)
    max_len = max(n for n, _ in pts) + max_new + 64
    llm = make_llm(tgt, mode, src=src, maps=maps, tail=tail, min_len=min_len, max_len=max_len, util=util,
                   sink=sink, chunk=chunk)
    sp = SamplingParams(max_tokens=int(max_new), temperature=0.0, ignore_eos=True)
    rng = random.Random(seed)
    mk = lambda n: [rng.randrange(vocab_lo, vocab_hi) for _ in range(n)]  # fresh tokens: no prefix reuse
    rows = []
    for n, c in pts:
        try:
            llm.generate([{"prompt_token_ids": mk(n)} for _ in range(c)], sp, use_tqdm=False)  # warm-up
            ts = []
            for _ in range(reps):
                batch = [{"prompt_token_ids": mk(n)} for _ in range(c)]
                t0 = time.perf_counter()
                llm.generate(batch, sp, use_tqdm=False)
                ts.append(time.perf_counter() - t0)
            wall = sorted(ts)[len(ts) // 2]  # median batch
            tok_s = n * c / wall
            row = dict(len=n, c=c, max_new=max_new, batch_wall_s=round(wall, 4), walls=[round(t, 4) for t in ts],
                       prompt_tok_s=round(tok_s, 1), usd_per_m_prompt=round(usd_per_m(tok_s, usd_h), 4))
        except Exception as e:  # keep the other grid points (OOM at large c*len is expected on one GPU)
            row = dict(len=n, c=c, error=f"{type(e).__name__}: {str(e)[:300]}")
        rows.append(row)
        print(json.dumps(row), flush=True)
    import torch
    res = dict(tgt=tgt, mode=mode, src=src, maps=maps, tail=tail, util=util, chunk=chunk, usd_h=usd_h, reps=reps,
               gpu=torch.cuda.get_device_name(0), rows=rows, protocol="closed batches of c requests, median of reps")
    os.makedirs("/vol/runs/throughput", exist_ok=True)
    path = f"/vol/runs/throughput/{tag or mode}.json"
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path, flush=True)
    return res
