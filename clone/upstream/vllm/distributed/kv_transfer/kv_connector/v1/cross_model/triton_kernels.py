"""Fused mapped-key epilogue: k_norm RMS restore + RoPE in one pass over [T, Lt, H, D] (bf16, in place).

Replaces ~6 elementwise kernels per target layer (64 layers -> ~400 launches, several fp32 temporaries).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _restore_rope_kernel(Y, W, COS, SIN, T, LH, H: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr,
                         DO_RMS: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)  # one program per (t, l, h) row; int64: T*Lt*H*D passes 2^31 at ~32K tokens
    t = pid // LH
    lh = pid % LH
    l = lh // H
    offs = tl.arange(0, HALF)
    base = Y + pid * D
    x1 = tl.load(base + offs).to(tl.float32)
    x2 = tl.load(base + HALF + offs).to(tl.float32)
    if DO_RMS:
        w1 = tl.load(W + l * D + offs).to(tl.float32)
        w2 = tl.load(W + l * D + HALF + offs).to(tl.float32)
        u1 = x1 / w1
        u2 = x2 / w2
        ms = (tl.sum(u1 * u1, 0) + tl.sum(u2 * u2, 0)) / D
        r = 1.0 / tl.sqrt(ms + 1e-6)
        x1 = w1 * u1 * r
        x2 = w2 * u2 * r
    c1 = tl.load(COS + t * D + offs).to(tl.float32)
    c2 = tl.load(COS + t * D + HALF + offs).to(tl.float32)
    s1 = tl.load(SIN + t * D + offs).to(tl.float32)
    s2 = tl.load(SIN + t * D + HALF + offs).to(tl.float32)
    # rope with rotate_half: [x1, x2] * cos + [-x2, x1] * sin
    o1 = x1 * c1 - x2 * s1
    o2 = x2 * c2 + x1 * s2
    tl.store(base + offs, o1.to(Y.dtype.element_ty))
    tl.store(base + HALF + offs, o2.to(Y.dtype.element_ty))


@triton.jit
def _restore_rope_rows_kernel(Y, W, COS, SIN, NROWS, LH, H: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr,
                              DO_RMS: tl.constexpr, ROWS: tl.constexpr):
    """ROWS (t, l, h) rows of D values per program: the one-row kernel launched T*Lt*H tiny programs (21M at 32K on 72B)."""
    pid = tl.program_id(0).to(tl.int64)
    r = pid * ROWS + tl.arange(0, ROWS).to(tl.int64)
    m = (r < NROWS)[:, None]
    t = r // LH
    l = (r % LH) // H
    offs = tl.arange(0, HALF)[None, :]
    base = r[:, None] * D + offs
    x1 = tl.load(Y + base, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(Y + base + HALF, mask=m, other=0.0).to(tl.float32)
    if DO_RMS:
        wb = l[:, None] * D + offs
        w1 = tl.load(W + wb, mask=m, other=1.0).to(tl.float32)
        w2 = tl.load(W + wb + HALF, mask=m, other=1.0).to(tl.float32)
        u1 = x1 / w1
        u2 = x2 / w2
        ms = (tl.sum(u1 * u1, 1) + tl.sum(u2 * u2, 1)) / D
        rr = (1.0 / tl.sqrt(ms + 1e-6))[:, None]
        x1 = w1 * u1 * rr
        x2 = w2 * u2 * rr
    cb = t[:, None] * D + offs
    c1 = tl.load(COS + cb, mask=m, other=0.0).to(tl.float32)
    c2 = tl.load(COS + cb + HALF, mask=m, other=0.0).to(tl.float32)
    s1 = tl.load(SIN + cb, mask=m, other=0.0).to(tl.float32)
    s2 = tl.load(SIN + cb + HALF, mask=m, other=0.0).to(tl.float32)
    o1 = x1 * c1 - x2 * s1
    o2 = x2 * c2 + x1 * s2
    tl.store(Y + base, o1.to(Y.dtype.element_ty), mask=m)
    tl.store(Y + base + HALF, o2.to(Y.dtype.element_ty), mask=m)


def restore_rope_(Y, kw, cos, sin, rows=32):
    """Y [T, Lt, H, D] contiguous (modified in place); kw [Lt, D] or None; cos/sin [>=T, D]."""
    T, Lt, H, D = Y.shape
    assert Y.is_contiguous()
    n = T * Lt * H
    grid = (triton.cdiv(n, rows),)
    _restore_rope_rows_kernel[grid](Y, kw if kw is not None else Y, cos, sin, n, Lt * H, H=H, D=D, HALF=D // 2,
                                    DO_RMS=kw is not None, ROWS=rows, num_warps=4)
    return Y
