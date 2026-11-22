"""Cross-model KV cache transfer: capture, ridge fit, map, and KL evaluation.

Model-agnostic for RoPE decoder LMs in HF transformers: keys are read from the
returned cache (post-RoPE) and un-rotated with the model's own cos/sin, so the
same code strips RoPE for Qwen3, Llama-3 (scaled RoPE), Mistral, Gemma.
"""
import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- RoPE

def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def rope_cos_sin(model, T, device):
    """[T, D] cos/sin, or [L, T, D] when layers use different RoPE (Gemma-3 local/global)."""
    pos = torch.arange(T, device=device)[None]
    dummy = torch.zeros(1, T, 1, device=device, dtype=torch.float32)
    cos, sin = model.model.rotary_emb(dummy, pos)
    cos, sin = cos[0].float(), sin[0].float()
    if hasattr(model.model, "rotary_emb_local"):
        cl, sl = model.model.rotary_emb_local(dummy, pos)
        cl, sl = cl[0].float(), sl[0].float()
        types = getattr(model.config, "layer_types", None) or \
            [getattr(l.self_attn, "attention_type", getattr(l, "attention_type", "full_attention")) for l in model.model.layers]
        loc = [t == "sliding_attention" for t in types]
        cos = torch.stack([cl if f else cos for f in loc])
        sin = torch.stack([sl if f else sin for f in loc])
    return cos, sin


def _cs(cos, sin, l, T):
    return (cos[l, :T], sin[l, :T]) if cos.dim() == 3 else (cos[:T], sin[:T])


def unrope(k, cos, sin):  # k [B,H,T,D] post-RoPE -> pre-RoPE
    """Exact inverse of rope() for any tables. HF folds YaRN's attention_scaling s into cos/sin (cos^2 + sin^2 = s^2),
    so cached keys are s*R(k); multiplying by the tables again gives s^2*k, so divide by cos^2 + sin^2.
    For plain RoPE (s = 1) this is the old formula."""
    n = cos * cos + sin * sin
    return (k * cos - rotate_half(k) * sin) / n


def rope(k, cos, sin):
    return k * cos + rotate_half(k) * sin


def cache_layers(pkv):
    if hasattr(pkv, "layers"):
        return [(l.keys, l.values) for l in pkv.layers]
    if hasattr(pkv, "key_cache"):
        return list(zip(pkv.key_cache, pkv.value_cache))
    return [(k, v) for k, v in pkv]


@torch.no_grad()
def capture_kv(model, ids, cos, sin, hidden=False):
    """Return pre-RoPE keys and values, each [L, B, H, T, D] (bf16), plus logits.

    hidden=True: third item is the per-layer residual stream [L, B, 1, T, d] instead of logits.
    """
    from transformers import DynamicCache
    # plain DynamicCache keeps every position (the default sliding-window cache drops keys for local layers)
    out = model(input_ids=ids, past_key_values=DynamicCache(), use_cache=True, logits_to_keep=1,
                output_hidden_states=hidden)
    ks, vs = [], []
    T = ids.shape[1]
    for l, (k, v) in enumerate(cache_layers(out.past_key_values)):
        c, s_ = _cs(cos, sin, l, T)
        ks.append(unrope(k.float(), c, s_).to(torch.bfloat16))
        vs.append(v)
    if hidden:
        h = torch.stack(out.hidden_states[1:])  # [L, B, T, d] outputs of each decoder layer
        return torch.stack(ks), torch.stack(vs), h.unsqueeze(2)
    return torch.stack(ks), torch.stack(vs), out.logits


@torch.no_grad()
def attention_received(model, ids, power=0.5, cap=10.0):
    """Per-token weight = mean attention it receives (all layers, heads, later queries), normalized to mean 1,
    compressed by `power` and capped (sinks would otherwise dominate). Returns [B, T]."""
    prev = model.config._attn_implementation
    model.set_attn_implementation("eager")
    try:
        out = model(input_ids=ids, output_attentions=True, logits_to_keep=1, use_cache=False)
    finally:
        model.set_attn_implementation(prev)
    rec = None
    for a in out.attentions:  # [B, H, Tq, Tk]
        r = a.float().mean(1).sum(1)  # [B, Tk]: total attention received
        rec = r if rec is None else rec + r
    T = ids.shape[1]
    nq = torch.arange(T, 0, -1, device=ids.device).float()  # queries that can see token t
    rec = rec / nq
    w = (rec / rec.mean(1, keepdim=True)).pow(power).clamp(max=cap)
    return w / w.mean(1, keepdim=True)


def flat_tokens(x):  # [L,B,H,T,D] -> [B*T, L, H*D]
    L, B, H, T, D = x.shape
    return x.permute(1, 3, 0, 2, 4).reshape(B * T, L, H * D)


# ---------------------------------------------------------------- ridge statistics

class GramStats:
    """Running centered second moments between all source and all target layers."""

    def __init__(self, Ls, ds, Lt, dt, device):
        self.Ls, self.ds, self.Lt, self.dt = Ls, ds, Lt, dt
        self.n = 0
        self.sx = torch.zeros(Ls * ds, device=device, dtype=torch.float64)
        self.sy = torch.zeros(Lt * dt, device=device, dtype=torch.float64)
        self.xx = torch.zeros(Ls * ds, Ls * ds, device=device, dtype=torch.float32)
        self.xy = torch.zeros(Ls * ds, Lt * dt, device=device, dtype=torch.float32)
        self.yy = torch.zeros(Lt * dt, device=device, dtype=torch.float64)

    def add(self, X, Y, w=None):  # X [N, Ls*ds], Y [N, Lt*dt] float32; w [N] optional sample weights
        if self.n == 0:  # shifted-data accumulation avoids cancellation on large key means
            self.shx, self.shy = X.mean(0), Y.mean(0)
        X = X - self.shx
        Y = Y - self.shy
        if w is not None:  # weighted ridge: sums of w*x, w*x x^T ...; n counts total weight
            r = w.sqrt()[:, None]
            self.n += w.sum().item()
            self.sx += (X * w[:, None]).sum(0).double()
            self.sy += (Y * w[:, None]).sum(0).double()
            Xr, Yr = X * r, Y * r
            self.xx.addmm_(Xr.T, Xr)
            self.xy.addmm_(Xr.T, Yr)
            self.yy += (Yr.double() ** 2).sum(0)
            return
        self.n += X.shape[0]
        self.sx += X.sum(0).double()
        self.sy += Y.sum(0).double()
        self.xx.addmm_(X.T, X)
        self.xy.addmm_(X.T, Y)
        self.yy += (Y.double() ** 2).sum(0)

    def finalize(self):
        n = self.n
        mx, my = self.sx / n, self.sy / n
        # center in place (float64 outer products cast back)
        self.xx.addr_(mx.float(), mx.float(), alpha=-n)  # in place: no Ls*ds x Ls*ds temporary
        self.xy.addr_(mx.float(), my.float(), alpha=-n)
        self.yyc = self.yy - n * my ** 2
        self.mx, self.my = (mx.float() + self.shx), (my.float() + self.shy)


def _solve_dtype(dev, n=0):
    if torch.device(dev).type != "cuda":  # CPU (unit tests): fp64 is cheap
        return torch.float64
    cc = torch.cuda.get_device_capability(dev)
    if n > 16384:  # big systems: fp64 copy would not fit next to the stats; fp32 Cholesky with ridge is fine
        return torch.float32
    return torch.float64 if cc in ((8, 0), (9, 0), (10, 0)) else torch.float32  # fast-fp64 parts only


def _lam(A, lam, lam_rel):
    return lam_rel * A.diagonal().mean().item() if lam_rel > 0 else lam


def _ridge(A, B, lam):
    """A is a private copy (callers gather it); regularize in place."""
    A.diagonal().add_(lam)
    L, info = torch.linalg.cholesky_ex(A)
    if info.item() != 0:
        return torch.linalg.lstsq(A, B).solution
    return torch.cholesky_solve(B, L)


def layer_r2(st, lam, H, lam_rel=0.0):
    """R2[t, h, s]: ridge R^2 of target layer t head h from single source layer s."""
    ds, dt = st.ds, st.dt
    D = dt // H
    R2 = torch.zeros(st.Lt, H, st.Ls, device=st.xx.device)
    syy = st.yyc.view(st.Lt, H, D).sum(-1)
    for s in range(st.Ls):
        sl = slice(s * ds, (s + 1) * ds)
        dt_ = _solve_dtype(st.xx.device)
        A = st.xx[sl, sl].to(dt_).clone()
        lm = _lam(A, lam, lam_rel)
        B = st.xy[sl].to(dt_)
        W = _ridge(A, B, lm)
        sse = st.yyc - (B * W).sum(0).double() - lm * (W * W).sum(0).double()
        R2[:, :, s] = (1 - sse.view(st.Lt, H, D).sum(-1) / syy).float()
        del A, B, W
    return R2


def fit_maps(st, R2, k, lam, H, lam_rel=0.0):
    """Per target layer: pick top-k source layers by head-averaged R^2 and solve ridge.

    Returns list over target layers of dict(sel=[source layers], W=[k*ds, dt], b=[dt], r2=[H]).
    """
    ds, dt = st.ds, st.dt
    D = dt // H
    maps = []
    dev = st.xx.device
    for t in range(st.Lt):
        score = R2[t].mean(0)
        sel = torch.topk(score, min(k, st.Ls)).indices.sort().values.tolist()
        idx = torch.cat([torch.arange(s * ds, (s + 1) * ds, device=dev) for s in sel])
        cols = torch.arange(t * dt, (t + 1) * dt, device=dev)
        dt_ = _solve_dtype(dev, len(idx))
        A = st.xx[idx[:, None], idx].to(dt_)  # one gather, no row-slab temporary
        lm = _lam(A, lam, lam_rel)
        B = st.xy[:, cols][idx].to(dt_)  # columns first: avoids a rows x all-target-columns copy
        W = _ridge(A, B, lm)
        sse = st.yyc[cols] - (B * W).sum(0).double() - lm * (W * W).sum(0).double()
        r2 = 1 - sse.view(H, D).sum(-1) / st.yyc[cols].view(H, D).sum(-1)
        b = st.my[cols] - st.mx[idx] @ W.float()
        vnat = (st.yyc[cols].view(H, D).sum(-1) / st.n).float()  # native E||k - mu||^2 per head
        maps.append(dict(sel=sel, W=W.float(), b=b, r2=r2.float(), mu=st.my[cols], H=H, D=D, vnat=vnat))
        del A, B
    return maps


def apply_maps(maps, Xs):
    """Xs [L_s, B, H, T, D] pre-RoPE source cache -> [L_t, B, H_t, T, D] mapped (float32)."""
    Ls, B, H, T, D = Xs.shape
    X = flat_tokens(Xs).float()  # [B*T, Ls, H*D]
    outs = []
    for m in maps:
        Y = X[:, m["sel"]].reshape(B * T, -1) @ m["W"] + m["b"]
        outs.append(Y.view(B, T, m["H"], m["D"]).permute(0, 2, 1, 3))
    return torch.stack(outs)


# ---------------------------------------------------------------- importance restore

def restore_scale(K, maps, mode, H, kw=None):
    """Rescale mapped pre-RoPE keys K [L,B,H,T,D] (float32).

    mode: none | var (centered, alpha=1/sqrt(R2_head)) | rms (match target k_norm RMS per token-head)
          | float string 'g1.2' (global centered gain)
    """
    if mode == "none":
        return K
    out = []
    for l, m in enumerate(maps):
        k = K[l]
        mu = m["mu"].view(1, H, 1, -1)
        if mode == "var":
            a = (1.0 / m["r2"].clamp(min=1e-3).sqrt()).view(1, H, 1, 1)
            k = mu + a * (k - mu)
        elif mode.startswith("g"):
            k = mu + float(mode[1:]) * (k - mu)
        elif mode == "ln":  # per-token centered norm set to the native per-head mean norm (no-k_norm models)
            c = k - mu
            tgt = m["vnat"].sqrt().view(1, H, 1, 1)
            k = mu + c * (tgt / c.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        elif mode in ("rmsfull", "rmsfullc"):  # k_norm over all heads (OLMo-2): weight [H*D]
            w = kw[l].view(1, H, 1, -1).float()
            u = k / w
            if mode == "rmsfullc":  # RMS estimated only on channels with non-tiny k_norm weight
                keep = (w.abs() >= 0.25 * w.abs().median()).float()
                ms = (u.pow(2) * keep).sum(dim=(1, 3), keepdim=True) / keep.sum().clamp(min=1)
            else:
                ms = u.pow(2).mean(dim=(1, 3), keepdim=True)
            k = k * torch.rsqrt(ms + 1e-6)  # rescale the mapped key; never divide noise by tiny w
        elif mode == "rms":
            w = kw[l].view(1, 1, 1, -1).float()  # target k_norm weight
            u = k / w
            k = w * u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + 1e-6)
        out.append(k)
    return torch.stack(out)


# ---------------------------------------------------------------- eval

def build_cache(K, V, cos, sin):
    from transformers import DynamicCache
    c = DynamicCache()
    T = K.shape[3]
    for l in range(K.shape[0]):
        cc, ss_ = _cs(cos, sin, l, T)
        c.update(rope(K[l].float(), cc, ss_).to(torch.bfloat16), V[l].to(torch.bfloat16), l)
    return c


def kl_rows(logp_ref, logits):
    lq = F.log_softmax(logits.float(), -1)
    return (logp_ref.exp() * (logp_ref - lq)).sum(-1)  # [B, M]


# ---------------------------------------------------------------- two-GPU placement

def _move_cache(c, dev):
    if c is None:
        return
    if hasattr(c, "layers"):
        for l in c.layers:
            if getattr(l, "keys", None) is not None and l.keys.device != torch.device(dev):
                l.keys, l.values = l.keys.to(dev), l.values.to(dev)
    elif hasattr(c, "key_cache"):
        c.key_cache = [k.to(dev) for k in c.key_cache]
        c.value_cache = [v.to(dev) for v in c.value_cache]


def place_on(model, dev_model, dev_io):
    """Keep `model` on dev_model while callers pass and receive tensors/caches on dev_io."""
    def pre(mod, args, kwargs):
        args = tuple(a.to(dev_model) if torch.is_tensor(a) else a for a in args)
        for k, v in list(kwargs.items()):
            if torch.is_tensor(v):
                kwargs[k] = v.to(dev_model)
        _move_cache(kwargs.get("past_key_values"), dev_model)
        return args, kwargs

    def post(mod, args, kwargs, out):
        out.logits = out.logits.to(dev_io)
        _move_cache(out.past_key_values, dev_io)
        return out

    model.register_forward_pre_hook(pre, with_kwargs=True)
    model.register_forward_hook(post, with_kwargs=True)
    return model


# ---------------------------------------------------------------- fp8 mapper quality simulation

def _fp8_round(x, dim):
    """Quantize to e4m3 with an absmax scale along `dim`, then dequantize (float32 out)."""
    amax = x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-12)
    scale = amax / 448.0
    return (x / scale).to(torch.float8_e4m3fn).float() * scale


def apply_maps_fp8sim(maps, Xs, center=True):
    """Same as apply_maps but X (per row, after centering) and W (per output column) rounded to e4m3."""
    Ls, B, H, T, D = Xs.shape
    X = flat_tokens(Xs).float()
    outs = []
    for m in maps:
        x = X[:, m["sel"]].reshape(B * T, -1)
        W = m.setdefault("_W8", _fp8_round(m["W"], 0))
        if center:
            mu = x.mean(0, keepdim=True)
            # as deployed (vllm_connector mm8): the mean term also goes through the e4m3 GEMM; Y is bf16 before the add
            Y = (_fp8_round(x - mu, 1) @ W).bfloat16().float() + (_fp8_round(mu, 1) @ W + m["b"]).bfloat16().float()
        else:
            Y = _fp8_round(x, 1) @ W + m["b"]
        outs.append(Y.view(B, T, m["H"], m["D"]).permute(0, 2, 1, 3))
    return torch.stack(outs)
