"""OneTriangle's quality suites on our pairs, through vLLM (native) or vLLM + CrossModelConnector (mapped).

Benchmark loaders and scorers are Magni's (xkv.genevals, MIT, github.com/amw720386/ultra-cross-model-kv): pass its
src/ on PYTHONPATH. Each arm is one vLLM engine; `gen(prompts, max_new)` takes token-id prompts and returns token ids,
which is the GenFn genevals expects.

Matched setting for OneTriangle's ladder: they map short prompts too (ARC-C 1.7B->32B .455 vs native .565), so the
mapped arm defaults to min_len=32, tail=16 (nearly the whole prompt mapped). min_len/tail are recorded with every row.
"""
from __future__ import annotations

import json
import os
import time

# OneTriangle's cell names -> (genevals runner, their n, extra kwargs)
SUITES = {
    "ARC-Challenge": ("run_arc_chat", 500, {}),
    "MATH-500": ("run_math500", 500, {}),
    "ZebraLogic": ("run_zebralogic", 200, {}),
    "QuALITY": ("run_quality", 500, {}),
    "LongBench v2": ("run_longbench_v2", 97, {}),
    "DevEval": ("run_deveval", 500, {}),
    "ContractNLI": ("run_contract_nli", 500, {}),
    "BAMBOO PaperQA": ("run_bamboo_paperqa", 84, {}),
    "LongHealth": ("run_longhealth", 100, {}),
    "HellaSwag": ("run_hellaswag_chat", 500, {}),
    "PIQA": ("run_piqa_chat", 500, {}),
}


def _register_extra_benches():
    """HellaSwag and PIQA as chat multiple choice with letter scoring (OneTriangle's routed 72B tab and 1.7B->8B tab
    score letters, not log-likelihood). Registered into Magni's genevals registry."""
    from xkv import genevals as G
    if "hellaswag_chat" in G._BENCH:
        return

    def _hs(tok, limit, seed, max_ctx, max_new=16):
        docs = G._load_hf("Rowan/hellaswag", None, "validation")

        def build(i):
            d = docs[i]
            endings = list(d["endings"])
            letters = G._letters(len(endings))
            gold = letters[int(d["label"])]
            user = (f"Choose the most plausible continuation.\n\nContext: {d['ctx']}\n{G._mc_block(endings, letters)}"
                    f"\n\nRespond with only the letter of the correct answer.")
            return G._mk(tok, G.chat_prompt(tok, user), gold, max_new, oracle=gold, meta={"letters": letters})
        return G._select(len(docs), build, limit, seed, max_ctx)

    def _piqa(tok, limit, seed, max_ctx, max_new=16):
        docs = G._load_hf("baber/piqa", None, "validation")

        def build(i):
            d = docs[i]
            opts = [d["sol1"], d["sol2"]]
            letters = G._letters(2)
            gold = letters[int(d["label"])]
            user = (f"Which solution achieves the goal?\n\nGoal: {d['goal']}\n{G._mc_block(opts, letters)}"
                    f"\n\nRespond with only the letter of the correct answer.")
            return G._mk(tok, G.chat_prompt(tok, user), gold, max_new, oracle=gold, meta={"letters": letters})
        return G._select(len(docs), build, limit, seed, max_ctx)

    G._BENCH["hellaswag_chat"] = G.Bench("hellaswag_chat", "acc", _hs, G._mc_score)
    G._BENCH["piqa_chat"] = G.Bench("piqa_chat", "acc", _piqa, G._mc_score)
    G.run_hellaswag_chat = lambda gen, tok, limit=100, seed=0, max_ctx=None, batch=8, max_new=None: G._run(
        "hellaswag_chat", gen, tok, limit, seed, max_ctx, batch, {"max_new": max_new})
    G.run_piqa_chat = lambda gen, tok, limit=100, seed=0, max_ctx=None, batch=8, max_new=None: G._run(
        "piqa_chat", gen, tok, limit, seed, max_ctx, batch, {"max_new": max_new})


def vllm_gen(llm):
    from vllm import SamplingParams

    def gen(prompts, max_new):
        sp = SamplingParams(max_tokens=int(max_new), temperature=0.0)
        outs = llm.generate([{"prompt_token_ids": list(p)} for p in prompts], sp, use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outs]
    return gen


def make_llm(tgt, mode, src=None, maps=None, tail=16, min_len=32, max_len=32768 + 1024, util=0.85, fp8=True,
             restore="rms", sink="", yarn=0.0, tp=1, backend="FLASH_ATTN", src_tp=True, chunk=20):
    from vllm import LLM
    from vllm.config import KVTransferConfig
    kw = dict(model=tgt, dtype="bfloat16", max_model_len=max_len, gpu_memory_utilization=util,
              enable_prefix_caching=False, tensor_parallel_size=tp, attention_backend=backend)
    if mode == "mapped":
        extra = {"src": src, "maps": maps, "tail": tail, "min_len": min_len, "restore": restore, "fp8": fp8,
                 "src_tp": int(bool(src_tp)), "map_chunk_layers": int(chunk)}
        if sink:
            extra["sink_template"] = sink
        if yarn:
            extra.update(yarn=yarn)
        kw["kv_transfer_config"] = KVTransferConfig(kv_connector="CrossModelConnector",
                                                    kv_connector_module_path="kvx.vllm_connector",
                                                    kv_role="kv_both", kv_connector_extra_config=extra)
    return LLM(**kw)


def run_suites(gen, tok, names, limits=None, seed=0, batch=128, max_ctx=None, thinking=False, max_new=None):
    from xkv import genevals as G
    _register_extra_benches()
    os.environ["XKV_THINKING"] = "1" if thinking else "0"
    out = {}
    for name in names:
        fn, n, kw = SUITES[name]
        n = (limits or {}).get(name, n)
        t0 = time.time()
        try:
            mn = (max_new or {}).get(name)
            r = getattr(G, fn)(gen, tok, limit=n, seed=seed, max_ctx=max_ctx, batch=batch, **kw,
                               **({"max_new": mn} if mn else {}))
        except Exception as e:  # keep the other suites going; record why this one failed
            import traceback
            r = {"error": traceback.format_exc()[-1500:]}
        r["_wall_s"] = round(time.time() - t0, 1)
        r["_n_requested"] = n
        out[name] = r
        print(json.dumps({name: {k: v for k, v in r.items() if not isinstance(v, (list, dict))}}), flush=True)
    return out


def oteval_impl(tgt: str, mode: str = "native", src: str = "", maps: str = "", suites: str = "ARC-Challenge",
                limits: str = "{}", tail: int = 16, min_len: int = 32, thinking: int = 0, tag: str = "",
                max_new: str = "{}", max_len: int = 33792, util: float = 0.85, restore: str = "rms", sink: str = "", tp: int = 1,
                src_tp: int = 1, chunk: int = 20):
    """One arm (native or mapped) of OneTriangle's suites for one pair. Writes /vol/runs/oteval/<tag>.json."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tgt)
    llm = make_llm(tgt, mode, src=src, maps=maps, tail=tail, min_len=min_len, max_len=max_len, util=util,
                   restore=restore, sink=sink, tp=tp, src_tp=src_tp, chunk=chunk)
    res = run_suites(vllm_gen(llm), tok, [s.strip() for s in suites.split(",") if s.strip()],
                     limits=json.loads(limits), thinking=bool(thinking), max_new=json.loads(max_new), max_ctx=max_len - 1024)
    meta = dict(tgt=tgt, src=src, mode=mode, maps=maps, tail=tail, min_len=min_len, thinking=bool(thinking),
                engine="vllm", restore=restore, sink=sink, tp=tp, src_tp=bool(src_tp))
    os.makedirs("/vol/runs/oteval", exist_ok=True)
    p = f"/vol/runs/oteval/{tag or (tgt.split('/')[-1] + '-' + mode)}.json"
    json.dump(dict(meta=meta, results=res), open(p, "w"), indent=1, default=str)
    return dict(meta=meta, results={k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))}
                                    for k, v in res.items()}, saved=p)


def oteval_hf_impl(src: str, tgt: str, maps: str = "", mode: str = "mapped", suites: str = "ARC-Challenge",
                   limits: str = "{}", tail: int = 16, min_len: int = 32, thinking: int = 0, tag: str = "",
                   restore: str = "rms", sink: str = "", max_ctx: int = 32768, device: str = "cuda"):
    """Same suites through plain HF (kvx/hfgen.py): runs on ROCm (AMD MI300X/MI350X) or CUDA without vLLM.
    mode=native ignores maps; mode=mapped needs maps (fit_eval save_maps) and, for rms restore, the target k_norm."""
    import torch
    from transformers import AutoTokenizer
    from .hfgen import make_hf_gen, source_kv_fn
    from .models import load_lm, model_path
    tok = AutoTokenizer.from_pretrained(model_path(tgt))
    T = load_lm(tgt, device=device).eval()
    stops = {t for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if isinstance(t, int) and t >= 0}
    if mode == "mapped":
        S = load_lm(src, device=device).eval()
        mp = torch.load(maps, map_location=device)
        kw = None
        if restore == "rms":
            kw = [l.self_attn.k_norm.weight.detach() for l in T.model.layers]
        sk = torch.load(sink, map_location=device) if sink else None
        kv_fn = source_kv_fn(S, T, mp["k"], mp["v"], mode=restore, kw=kw, sink=sk, feat=mp.get("feat", "kv"))
        gen = make_hf_gen(T, kv_fn, tail=tail, min_len=min_len, stop_ids=stops)
    else:
        gen = make_hf_gen(T, None, tail=10 ** 9, min_len=10 ** 9, stop_ids=stops)
    res = run_suites(gen, tok, [s.strip() for s in suites.split(",") if s.strip()], limits=json.loads(limits),
                     thinking=bool(thinking), max_ctx=max_ctx, batch=1)
    meta = dict(tgt=tgt, src=src, mode=mode, maps=maps, tail=tail, min_len=min_len, thinking=bool(thinking),
                engine="hf", restore=restore, sink=sink, device=device,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    os.makedirs("/vol/runs/oteval", exist_ok=True)
    p = f"/vol/runs/oteval/{tag or (tgt.split('/')[-1] + '-hf-' + mode)}.json"
    json.dump(dict(meta=meta, results=res), open(p, "w"), indent=1, default=str)
    return dict(meta=meta, results={k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))}
                                    for k, v in res.items()}, saved=p)


def loglik_impl(src: str, tgt: str, maps: str, suites: str = "ARC-Challenge,HellaSwag,PIQA", n: int = 500,
                tail: int = 1, restore: str = "rms", tag: str = "", device: str = "cuda"):
    """OneTriangle's short-context suites scored lm-eval style (log-likelihood of each choice; acc_norm for ARC-C and
    HellaSwag, acc for PIQA). Their native numbers (8B: ARC-C .54, HellaSwag .745, PIQA .79) match this scoring, not
    chat letters. Mapped = the whole context except `tail` tokens comes from the source through the maps."""
    import torch
    from transformers import AutoTokenizer
    from . import bench, core
    from .models import load_lm, model_path
    tok = AutoTokenizer.from_pretrained(model_path(tgt))
    tok.name_or_path = tgt
    S = load_lm(src, device=device).eval()
    T = load_lm(tgt, device=device).eval()
    mp = torch.load(maps, map_location=device)
    Ht = mp["k"][0]["H"]
    kw = [l.self_attn.k_norm.weight.detach() for l in T.model.layers] if restore == "rms" else None
    cs, ss = core.rope_cos_sin(S, 4096, device)
    ct, st = core.rope_cos_sin(T, 4096, device)

    def mapped(Ks, Vs):
        if mp.get("feat") == "kvcat":
            Ks = Vs = torch.cat([Ks, Vs], dim=2)
        return (core.restore_scale(core.apply_maps(mp["k"], Ks), mp["k"], restore, Ht, kw),
                core.apply_maps(mp["v"], Vs))
    loaders = {"ARC-Challenge": (bench.arc_items, True), "HellaSwag": (bench.hellaswag_items, True),
               "PIQA": (bench.piqa_items, False)}
    res = {}
    with torch.no_grad():
        for name in [x.strip() for x in suites.split(",") if x.strip()]:
            fn, norm = loaders[name]
            res[name] = bench.run(tok, S, T, fn(n), {"native": "native", "source": "source", "mapped": mapped},
                                  cs, ss, ct, st, device, norm=norm, tail=tail)
            print(json.dumps({name: res[name]}), flush=True)
    meta = dict(src=src, tgt=tgt, maps=maps, n=n, tail=tail, restore=restore, scoring="loglik (lm-eval style)")
    os.makedirs("/vol/runs/oteval", exist_ok=True)
    p = f"/vol/runs/oteval/{tag or (tgt.split('/')[-1] + '-loglik')}.json"
    json.dump(dict(meta=meta, results=res), open(p, "w"), indent=1)
    return dict(meta=meta, results=res, saved=p)
