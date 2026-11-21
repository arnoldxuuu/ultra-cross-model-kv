"""make_batch_gen with a session cache: xkv.genrun.make_batch_gen's transfer/native prefill per prompt, but a prompt
that extends a stored one is appended natively on the kept cache (hybrid/session.py). Same signature plus `store`."""
from __future__ import annotations

import os

import torch

from .session import SessionStore, session_gen


def make_session_batch_gen(strategy, sm, tm, mapper, tok, tail=128, window=8192, batch=8, chunk=8192, log=None,
                           full_until=None, turn_tail_max=None, turn_tail_msgs=1, store: SessionStore | None = None):
    from .decode import Decoder
    from .genrun import stop_ids
    from .perf import _bucket
    from .mapper import adaptive_tail, native_prefill, transfer_prefill

    store = store or SessionStore()
    stops = stop_ids(tok)
    repair, rstate, runit = 0, "final", False
    if "+rep" in strategy:
        strategy, r = strategy.split("+rep")
        rstate = "zero" if "z" in r else "final"
        runit = "m" in r
        repair = int(r.rstrip("zm"))
    if strategy.startswith("transfer@"):
        strategy, tail = "transfer", strategy.split("@")[1]
        tail = tail if tail == "auto" else int(tail)
    model = sm if strategy == "source" else tm
    im_start = tok.convert_tokens_to_ids("<|im_start|>")

    def tail_for(p):
        base = adaptive_tail(len(p)) if tail == "auto" else tail
        if not turn_tail_max:
            return base
        starts = [i for i, t in enumerate(p) if t == im_start]
        if len(starts) < 2:
            return base
        k = min(turn_tail_msgs + 1, len(starts) - 1)
        return max(base, min(turn_tail_max, len(p) - starts[-k]))

    def base_prefill(p, span):
        ids = torch.tensor([p], device="cuda")
        if strategy == "transfer":
            return transfer_prefill(sm, tm, mapper, ids, tail=min(tail_for(p), ids.shape[1]), window=window,
                                    chunk=chunk, full_until=full_until, native_spans=span, repair=repair,
                                    repair_state=rstate, repair_sep=im_start if runit else None,
                                    graph_tail=False if os.environ.get("XKV_NO_GRAPH_TAIL") == "1" else None)
        return native_prefill(model, ids, chunk=chunk)

    prefill = session_gen(base_prefill, tm, store, to_tensor=lambda x: torch.tensor([x], device="cuda"), chunk=chunk)
    hows = {"session": 0, "base": 0}

    def gen(prompts, max_new, spans=None):
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        out = [None] * len(prompts)
        for g0 in range(0, len(order), batch):
            idx = order[g0:g0 + batch]
            cap = _bucket(max(len(prompts[i]) for i in idx) + max_new + 16)
            dec = Decoder(model, len(idx), cap)
            for b, i in enumerate(idx):
                c, lg, how = prefill(prompts[i], spans[i] if spans else None)
                hows[how] += 1
                dec.load(b, c, int(lg[0, -1].argmax()))  # the decoder copies; the stored cache stays at prompt end
            outs, _ = dec.generate(max_new, stops)
            for b, i in enumerate(idx):
                out[i] = outs[b]
            del dec
            torch.cuda.empty_cache()
            if log:
                log(f"session {strategy} {min(g0 + batch, len(order))}/{len(order)} {hows} store={store.stats} "
                    f"{store.nbytes() / 2**30:.1f} GiB")
        return out

    gen.store, gen.hows = store, hows
    return gen
