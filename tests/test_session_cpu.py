"""CPU tests for hybrid/session.py with a mock hybrid model (no fla kernels needed)."""
import torch

from xkv.session import SessionStore, grow, session_gen


class MockCache:
    def __init__(self, max_len, D=4):
        self.max_len, self.len = max_len, 0
        self.k = {0: torch.zeros(1, max_len, 1, D)}
        self.v = {0: torch.zeros(1, max_len, 1, D)}
        self.conv = {1: torch.zeros(1, 2, D)}
        self.S = {1: torch.zeros(1, 1, D, D)}

    def nbytes(self):
        return int(self.k[0].numel() * 4 * 2 * self.len / self.max_len) + 64


class MockModel:
    """State = running sum of token ids (stands in for the recurrent state); K stores the token id at its position."""

    def new_cache(self, B, max_len):
        return MockCache(max_len)

    def prefill(self, ids, cache, chunk=8192):
        ids = [int(t) for t in ids]
        for t in ids:
            cache.k[0][0, cache.len, 0, 0] = t
            cache.S[1][0, 0, 0, 0] += t
            cache.len += 1
        return torch.tensor([[float(cache.S[1][0, 0, 0, 0])]])


def full(model, ids):
    c = model.new_cache(1, len(ids) + 4)
    lg = model.prefill(ids, c)
    return c, lg


def test_session_extension_equals_full_prefill():
    m = MockModel()
    store = SessionStore(min_reuse=3)
    calls = []

    def base(ids, span):
        calls.append(len(ids))
        return full(m, ids)

    pre = session_gen(base, m, store, to_tensor=lambda x: x)
    turn1 = list(range(1, 9))
    c1, lg1, how1 = pre(turn1)
    turn2 = turn1 + [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]  # grows past capacity 8+4 -> grow()
    c2, lg2, how2 = pre(turn2)
    ref_c, ref_lg = full(m, turn2)
    assert (how1, how2) == ("base", "session") and calls == [8]
    assert torch.equal(lg2, ref_lg) and c2.len == len(turn2)
    assert torch.equal(c2.k[0][0, :c2.len], ref_c.k[0][0, :ref_c.len]) and torch.equal(c2.S[1], ref_c.S[1])
    assert store.stats["reused_tokens"] == 8 and store.stats["appended_tokens"] == 10


def test_non_prefix_falls_back_and_short_prefix_ignored():
    m = MockModel()
    store = SessionStore(min_reuse=5)
    calls = []

    def base(ids, span):
        calls.append(tuple(ids))
        return full(m, ids)

    pre = session_gen(base, m, store, to_tensor=lambda x: x)
    pre([1, 2, 3, 4, 5, 6])
    _, _, how = pre([1, 2, 9, 4, 5, 6, 7])  # diverges at position 2 -> no exact prefix -> base
    assert how == "base"
    _, _, how = pre([7, 8, 9])  # nothing stored is a prefix
    assert how == "base" and store.stats["hits"] == 0


def test_identical_prompt_returns_stored_logits_and_longest_prefix_wins():
    m = MockModel()
    store = SessionStore(min_reuse=2)
    pre = session_gen(lambda ids, s: full(m, ids), m, store, to_tensor=lambda x: x)
    _, lg_a, _ = pre([1, 2, 3])
    _, _, _ = pre([5, 6, 7, 8])
    _, lg_b, how = pre([1, 2, 3])
    assert how == "session" and torch.equal(lg_a, lg_b)
    pre([1, 2, 3, 4, 5])  # takes [1,2,3] (the longest stored prefix), stores [1..5]
    c, lg, how = pre([1, 2, 3, 4, 5, 6])
    assert how == "session" and store.stats["reused_tokens"] == 3 + 3 + 5


def test_lru_eviction_by_bytes():
    m = MockModel()
    store = SessionStore(min_reuse=1, max_bytes=200)
    pre = session_gen(lambda ids, s: full(m, ids), m, store, to_tensor=lambda x: x)
    for i in range(6):
        pre([100 + i] * 10)
    assert store.stats["evicted"] > 0 and store.nbytes() <= 200 or len(store.entries) == 1


def test_grow_keeps_state():
    m = MockModel()
    c, _ = full(m, [3, 4, 5])
    g = grow(m, c, 50)
    assert g.max_len >= 50 and g.len == 3 and torch.equal(g.k[0][0, :3], c.k[0][0, :3]) and torch.equal(g.S[1], c.S[1])
