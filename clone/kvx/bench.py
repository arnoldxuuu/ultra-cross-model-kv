"""Accuracy retention with a mapped context prefix (HellaSwag acc_norm, MMLU 5-shot).

The context minus its last token comes from the source model through the mapper; the target
processes the last context token plus the scored continuation natively, as in serving.
"""
import torch
import torch.nn.functional as F
from . import core


def hellaswag_items(n):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    out = []
    for ex in ds.select(range(n)):
        ctx = ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
        out.append((ctx, [" " + e for e in ex["endings"]], int(ex["label"])))
    return out


def arc_items(n, split="test"):
    """ARC-Challenge, lm-eval style: "Question: ...\nAnswer:" + " " + choice text; acc_norm with norm=True."""
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    out = []
    for ex in ds.select(range(min(n, len(ds)))):
        labels = list(ex["choices"]["label"])
        out.append(("Question: " + ex["question"] + "\nAnswer:", [" " + t for t in ex["choices"]["text"]],
                    labels.index(ex["answerKey"])))
    return out


def piqa_items(n, split="validation"):
    """PIQA, lm-eval style: "Question: <goal>\nAnswer:" + " " + solution; acc (norm=False)."""
    from datasets import load_dataset
    ds = load_dataset("baber/piqa", split=split)
    return [("Question: " + ex["goal"] + "\nAnswer:", [" " + ex["sol1"], " " + ex["sol2"]], int(ex["label"]))
            for ex in ds.select(range(min(n, len(ds))))]


def mmlu_items(n, shots=5, seed=0):
    from datasets import load_dataset
    test = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=seed).select(range(n))
    dev = load_dataset("cais/mmlu", "all", split="dev")
    by_sub = {}
    for ex in dev:
        by_sub.setdefault(ex["subject"], []).append(ex)
    L = "ABCD"

    def fmt(ex, ans=True):
        s = ex["question"].strip() + "\n" + "".join(f"{L[i]}. {c}\n" for i, c in enumerate(ex["choices"])) + "Answer:"
        return s + (f" {L[ex['answer']]}\n\n" if ans else "")

    out = []
    for ex in test:
        sub = ex["subject"]
        head = f"The following are multiple choice questions (with answers) about {sub.replace('_', ' ')}.\n\n"
        ctx = head + "".join(fmt(d) for d in by_sub[sub][:shots]) + fmt(ex, ans=False)
        out.append((ctx, [" A", " B", " C", " D"], int(ex["answer"])))
    return out


@torch.no_grad()
def score_item(tok, S, T, ctx, conts, method, cs, ss, ct, st, dev, tail=1):
    """Return per-continuation (sum logprob, n chars). method: callable(Ks,Vs)->(K,V) mapped, 'native' or 'source'."""
    c_ids = tok(ctx, add_special_tokens=False)["input_ids"]
    if tok.bos_token_id is not None and "qwen" not in tok.name_or_path.lower():
        c_ids = [tok.bos_token_id] + c_ids  # match calibration (BOS-first) for Llama/Mistral/Gemma/DeepSeek
    e_ids = [tok(e, add_special_tokens=False)["input_ids"] for e in conts]
    n = len(conts)
    tail = max(1, min(tail, len(c_ids) - 1))
    nt = tail
    tail = [c_ids[-nt:] + e for e in e_ids]
    M = max(len(t) for t in tail)
    pad = tok.pad_token_id or 0
    if method in ("native", "source"):
        model = T if method == "native" else S
        seqs = [c_ids[:-nt] + t + [pad] * (M - len(t)) for t in tail]
        logits = model(input_ids=torch.tensor(seqs, device=dev)).logits[:, len(c_ids) - 1:].float()
    else:
        pre = torch.tensor([c_ids[:-nt]], device=dev)
        Ks, Vs, _ = core.capture_kv(S, pre, cs, ss)
        K, V = method(Ks, Vs)
        K = K.expand(-1, n, -1, -1, -1).contiguous()
        V = V.expand(-1, n, -1, -1, -1).contiguous()
        cache = core.build_cache(K, V, ct, st)
        seqs = [t + [pad] * (M - len(t)) for t in tail]
        logits = T(input_ids=torch.tensor(seqs, device=dev), past_key_values=cache).logits[:, nt - 1:].float()
    lp = F.log_softmax(logits, -1)
    scores = []
    for i, e in enumerate(e_ids):
        tgt = torch.tensor(e, device=dev)
        scores.append(lp[i, torch.arange(len(e), device=dev), tgt].sum().item())
    return scores


def run(tok, S, T, items, methods, cs, ss, ct, st, dev, norm=True, tail=1):
    acc = {m: 0 for m in methods}
    for ctx, conts, label in items:
        for name, meth in methods.items():
            sc = score_item(tok, S, T, ctx, conts, meth, cs, ss, ct, st, dev, tail=tail)
            if norm:
                sc = [s / max(1, len(c)) for s, c in zip(sc, conts)]
            acc[name] += int(max(range(len(sc)), key=lambda i: sc[i]) == label)
    return {m: a / len(items) for m, a in acc.items()}
