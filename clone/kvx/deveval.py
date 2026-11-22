"""DevEval (code) for OneTriangle's "long context" rows: local-file infilling prompts (seketeam/DevEval
Experiments/prompt/local_infilling, the instruction format: file contents above and below + the signature/docstring to
complete), chat template, greedy, reasoning off. The file context is what gets mapped; the tail (the code to complete)
runs natively. Output: {namespace, completion (function body), raw} jsonl for kvx.deveval_eval.

  /venv-serve/bin/python run_local.py kvx.deveval:deveval_gen_impl --json '{"tgt": "Qwen/Qwen3-14B", "mode": "mapped",
     "src": "Qwen/Qwen3-1.7B", "maps": ".../maps_k24.pt", "sink": "...", "items": "/vol/deveval/items500.jsonl"}'
"""
import ast
import json
import os
import re
import textwrap
import time

FENCE = re.compile(r"```[ \t]*(?:python|py|Python)?[ \t]*\n(.*?)(?:```|$)", re.S)


def extract_body(text, function_name):
    """Model reply -> the function body (dedented), the part DevEval splices over body_position (after the docstring)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"<think>.*$", "", text, flags=re.S)  # reasoning cut off by max_new: no answer
    blocks = FENCE.findall(text)
    code = max(blocks, key=len) if blocks else text
    code = textwrap.dedent(code.strip("\n"))
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    if tree is not None:
        fn = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == function_name), None)
        if fn is not None:
            body = fn.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str) and len(body) > 1:
                body = body[1:]  # the docstring sits above body_position in the repo file
            lines = code.split("\n")
            return textwrap.dedent("\n".join(lines[body[0].lineno - 1:body[-1].end_lineno])) + "\n"
    # no parseable def: a bare body, or a def we could not parse -> drop a leading def line if present
    lines = code.split("\n")
    if lines and re.match(rf"\s*(async\s+)?def\s+{re.escape(function_name)}\b", lines[0]):
        lines = lines[1:]
    return textwrap.dedent("\n".join(lines)) + "\n"


def load_items(items_path, prompt_file):
    """items: jsonl rows of data.jsonl (namespace, function_name via namespace tail...) chosen by the ground-truth pass."""
    items = [json.loads(l) for l in open(items_path)]
    prompts = {}
    want = {it["namespace"] for it in items}
    for l in open(prompt_file):
        d = json.loads(l)
        if d["namespace"] in want:
            prompts[d["namespace"]] = d["prompt"]
    return [dict(it, prompt=prompts[it["namespace"]]) for it in items if it["namespace"] in prompts]


def deveval_gen_impl(tgt: str, mode: str = "native", src: str = "", maps: str = "", sink: str = "",
                     items: str = "/vol/deveval/items500.jsonl",
                     prompt_file: str = "/vol/deveval/local_infilling_prompt.jsonl", tail: int = 256,
                     min_len: int = 512, util: float = 0.80, chunk: int = 8, max_new: int = 1024,
                     max_len: int = 33792, tag: str = "", no_context: int = 0, thinking: int = 0):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from .oteval import make_llm
    tok = AutoTokenizer.from_pretrained(tgt)
    its = load_items(items, prompt_file)
    if no_context:  # OneTriangle's "no context" bar: the target sees only the code to complete
        for it in its:
            p = it["prompt"]
            k = p.find("The code to be completed is:")
            it["prompt"] = f"Please complete the {it['namespace'].split('.')[-1]} function.\n\n" + p[k:] if k >= 0 else p
    ids, keep = [], []
    for it in its:
        msgs = [{"role": "user", "content": it["prompt"]}]
        try:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=bool(thinking))
        except TypeError:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        x = tok(s, add_special_tokens=False)["input_ids"]
        if len(x) + max_new > max_len:
            continue
        ids.append(x)
        keep.append(it)
    llm = make_llm(tgt, mode, src=src, maps=maps, tail=tail, min_len=min_len, max_len=max_len, util=util, sink=sink,
                   chunk=chunk)
    sp = SamplingParams(max_tokens=int(max_new), temperature=0.0)  # greedy in both modes (reasoning on: extract_body drops <think>)
    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": x} for x in ids], sp, use_tqdm=False)
    wall = time.time() - t0
    os.makedirs("/vol/runs/deveval", exist_ok=True)
    path = f"/vol/runs/deveval/{tag or mode}.jsonl"
    with open(path, "w") as f:
        for it, o in zip(keep, outs):
            raw = o.outputs[0].text
            fn = it["namespace"].split(".")[-1]
            f.write(json.dumps(dict(namespace=it["namespace"], completion=extract_body(raw, fn), raw=raw)) + "\n")
    meta = dict(tgt=tgt, mode=mode, src=src, maps=maps, tail=tail, thinking=int(bool(thinking)), max_new=max_new, n_items=len(its), n_generated=len(keep),
                skipped_too_long=len(its) - len(keep), wall_s=round(wall, 1), out=path,
                prompt_tokens_mean=round(sum(map(len, ids)) / max(1, len(ids)), 1))
    json.dump(meta, open(path.replace(".jsonl", ".meta.json"), "w"), indent=1)
    print(json.dumps(meta), flush=True)
    return meta
