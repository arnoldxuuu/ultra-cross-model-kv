"""fp8 (e4m3) linear layers for the SOURCE model's prefill: per-output-channel weight scale, dynamic per-row activation
scale, torch._scaled_mm. CPU check: Qwen2.5-3B KV changes by 2.4% (K) / 5.0% (V), negligible against the mapper error."""
import torch

E4M3 = torch.float8_e4m3fn


class FP8Linear(torch.nn.Module):
    def __init__(self, lin: torch.nn.Linear, emulate: bool = False):
        super().__init__()
        W = lin.weight.data
        sw = (W.abs().amax(1, keepdim=True).float().clamp(min=1e-12) / 448.0)  # [out, 1]
        self.w8 = (W.float() / sw).to(E4M3).t()  # [in, out], column-major as _scaled_mm wants
        self.sw = sw.t().contiguous()            # [1, out]
        self.bias = None if lin.bias is None else lin.bias.data
        self.out_dtype = W.dtype
        self.emulate = emulate

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        m = x2.shape[0]
        pad = (-m) % 16  # _scaled_mm wants row counts that are multiples of 16
        if pad:
            x2 = torch.cat([x2, x2.new_zeros(pad, x2.shape[1])])
        sx = x2.abs().amax(1, keepdim=True).float().clamp(min=1e-12) / 448.0
        x8 = (x2.float() / sx).to(E4M3)
        if self.emulate:
            y = ((x8.float() * sx) @ (self.w8.float() * self.sw)).to(self.out_dtype)
            if self.bias is not None:
                y = y + self.bias
        else:
            y = torch._scaled_mm(x8, self.w8, scale_a=sx, scale_b=self.sw, out_dtype=self.out_dtype,
                                 bias=None if self.bias is None else self.bias.to(self.out_dtype))
        if pad:
            y = y[:m]
        return y.reshape(*shp[:-1], y.shape[-1])


def quantize_linears(model, skip=("lm_head",), emulate=False):
    """Replace every nn.Linear (except names containing `skip`) by FP8Linear in place. Returns the count."""
    n = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, torch.nn.Linear) and not any(s in full for s in skip):
                setattr(mod, cname, FP8Linear(child, emulate=emulate))
                n += 1
    return n
