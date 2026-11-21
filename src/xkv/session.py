"""Session cache for multi-turn agents (tau2, Terminal-Bench) on a transferred hybrid cache.

Agent prompts only grow: request t+1 = request t + the assistant's reply + new messages. Magni's engine re-runs the whole
transfer (4B prefill, maps, repair) on every request; on Terminal-Bench (prompts up to ~75K tokens) that made every
turn slow and every trial time out (0.10 vs native 0.70). Here the target's prompt-end cache (attention KV + DeltaNet
recurrent and conv states) is kept after each request. When the next prompt starts with the stored prompt token for
token, only the new suffix is prefilled natively on top of it: the DeltaNet state stays exact, and the cost per turn is
a native append of the new tokens instead of a full re-transfer. The first request of a conversation (and any request
whose prefix does not match exactly) is transferred as before.

The decoder copies the prefill cache into its own batch cache (xkv.decode.BatchCache.load), so the stored cache stays
at the prompt end while the reply is generated.
"""
from __future__ import annotations

from collections import OrderedDict


class SessionStore:
    """LRU of (prompt token ids, prompt-end Cache, last logits) keyed by conversation prefix; bounded by bytes."""

    def __init__(self, max_bytes: int = 40 << 30, min_reuse: int = 1024):
        self.max_bytes, self.min_reuse = max_bytes, min_reuse
        self.entries: OrderedDict[int, tuple] = OrderedDict()
        self._next = 0
        self.stats = {"hits": 0, "misses": 0, "reused_tokens": 0, "appended_tokens": 0, "evicted": 0}

    def take(self, ids: list[int]):
        """Remove and return (stored_ids, cache, last_logits) for the longest stored prompt that is an exact prefix of
        `ids` (and at least min_reuse tokens), else None. Taken entries are extended in place by the caller."""
        best, best_len = None, -1
        for key, (sids, cache, lg) in self.entries.items():
            n = len(sids)
            if self.min_reuse <= n <= len(ids) and n > best_len and ids[:n] == sids:
                best, best_len = key, n
        if best is None:
            self.stats["misses"] += 1
            return None
        self.stats["hits"] += 1
        self.stats["reused_tokens"] += best_len
        self.stats["appended_tokens"] += len(ids) - best_len
        return self.entries.pop(best)

    def put(self, ids: list[int], cache, last_logits) -> None:
        self.entries[self._next] = (list(ids), cache, last_logits)
        self._next += 1
        while len(self.entries) > 1 and self.nbytes() > self.max_bytes:
            self.entries.popitem(last=False)
            self.stats["evicted"] += 1

    def nbytes(self) -> int:
        return sum(c.nbytes() for _, c, _ in self.entries.values())


def grow(model, cache, need: int, slack: float = 1.25):
    """Return a cache with capacity >= need holding the same state (copies attention KV up to cache.len)."""
    if cache.max_len >= need:
        return cache
    new = model.new_cache(cache.k[next(iter(cache.k))].shape[0] if cache.k else 1, int(need * slack))
    n = cache.len
    for j in cache.k:
        new.k[j][:, :n] = cache.k[j][:, :n]
        new.v[j][:, :n] = cache.v[j][:, :n]
    for j in cache.conv:
        new.conv[j].copy_(cache.conv[j])
        new.S[j].copy_(cache.S[j])
    new.len = n
    return new


def extend(model, entry, ids: list[int], chunk: int = 8192, headroom: int = 512, to_tensor=None):
    """Prefill ids[len(stored):] natively on top of the stored cache. Returns (cache, last_logits)."""
    sids, cache, lg = entry
    rest = ids[len(sids):]
    if not rest:  # identical prompt: the stored logits are the answer
        return cache, lg
    cache = grow(model, cache, len(ids) + headroom)
    x = to_tensor(rest) if to_tensor is not None else rest
    lg = model.prefill(x, cache, chunk=chunk)
    return cache, lg


def session_gen(base_prefill, model, store: SessionStore, to_tensor, chunk: int = 8192):
    """Wrap a per-prompt prefill `base_prefill(ids, span) -> (cache, logits)` with the session store.
    Returns prefill(ids, span) -> (cache, logits, how) with how in {"session", "base"}; stores the prompt-end cache."""

    def prefill(ids: list[int], span=None):
        hit = store.take(ids)
        if hit is not None:
            c, lg = extend(model, hit, ids, chunk=chunk, to_tensor=to_tensor)
            how = "session"
        else:
            c, lg = base_prefill(ids, span)
            how = "base"
        store.put(ids, c, lg)
        return c, lg, how

    return prefill
