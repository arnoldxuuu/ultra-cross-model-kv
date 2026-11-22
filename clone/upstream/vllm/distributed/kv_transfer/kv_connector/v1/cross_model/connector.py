"""vLLM v1 KV connector: cross-model KV transfer (small-model prefill -> mapped target KV).

Use:
    from vllm import LLM
    from vllm.config import KVTransferConfig
    llm = LLM(model="Qwen/Qwen3-32B", kv_transfer_config=KVTransferConfig(
        kv_connector="CrossModelConnector", kv_connector_module_path="kvx.vllm_connector", kv_role="kv_both",
        kv_connector_extra_config={"src": "Qwen/Qwen3-1.7B", "maps": "/vol/runs/q17-q32/maps_k16.pt",
                                   "tail": 64, "min_len": 1024, "restore": "rms"}))

Scheduler side: a prompt of length n >= min_len gets m = align_down(n - tail, block) tokens reported as an
external hit, so vLLM schedules only the last n - m tokens for native prefill.
Worker side: start_load_kv runs the source model on prompt[:m], maps its pre-RoPE KV with one dense GEMM per
K/V (kvx.ttft.dense_weights), applies the target k_norm RMS restore and target RoPE, and writes K/V into the
target's paged blocks at the request's slot mapping.
"""
import re
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger

logger = init_logger("vllm.kvx_connector")  # inside vllm.* so INFO is emitted
_LAYER_RE = re.compile(r"layers\.(\d+)\.")


@dataclass
class _Req:
    token_ids: list
    slot_mapping: torch.Tensor
    start: int = 0  # tokens [0, start) already hold native KV (prefix-cache hit): never overwrite shared blocks


@dataclass
class CrossModelMeta(KVConnectorMetadata):
    requests: list = field(default_factory=list)


def _tp_rank_size(override=None):
    """(rank, world size) of vLLM's tensor-parallel group; (0, 1) outside vLLM or before it is initialized."""
    if override is not None:
        return override
    try:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
        return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
    except Exception:
        return 0, 1


def write_kv_fa(kv, bi, off, K, V, block):
    """Write K, V [n, H, D] at slots (bi, off) into a FlashAttention-layout cache kv [blocks, H, block, 2D] (any strides).
    Fast path: whole blocks and a token-major physical layout ([blocks, block, H, 2D] contiguous, what vLLM allocates)
    -> one index_copy_ of whole blocks. Per-token advanced indexing took 1.25 s for 130K tokens x 64 layers."""
    n, H, D = K.shape
    phys = kv.permute(0, 2, 1, 3)  # [blocks, block, H, 2D]
    if n % block == 0 and n and phys.is_contiguous() and bool((off.view(-1, block) == torch.arange(block, device=off.device)).all()):
        blocks = bi.view(-1, block)[:, 0]
        src = torch.cat([K, V], -1).to(kv.dtype).view(n // block, block, H, 2 * D)
        phys.index_copy_(0, blocks, src)
        return
    kv[bi, :, off, :D] = K.to(kv.dtype)
    kv[bi, :, off, D:] = V.to(kv.dtype)


def target_cos_sin(cfg, T, device):
    """cos/sin [T, D] for a target config, matching HF's rotary embedding (incl. attention_scaling)."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    rs = getattr(cfg, "rope_scaling", None) or getattr(cfg, "rope_parameters", None) or {}
    rope_type = (rs.get("rope_type") or rs.get("type") or "default") if isinstance(rs, dict) else "default"
    if rope_type == "default":  # transformers >= 5 dropped "default" from ROPE_INIT_FUNCTIONS: plain RoPE
        rp = rs if isinstance(rs, dict) else {}
        theta = float(rp.get("rope_theta") or getattr(cfg, "rope_theta", None) or 10000.0)
        hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        dim = int(hd * float(rp.get("partial_rotary_factor") or getattr(cfg, "partial_rotary_factor", 1.0) or 1.0))
        inv_freq = 1.0 / theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.int64).float() / dim)
        scaling = 1.0
    else:
        inv_freq, scaling = ROPE_INIT_FUNCTIONS[rope_type](cfg, device)
    t = torch.arange(T, device=device, dtype=torch.float32)
    fr = torch.outer(t, inv_freq.float())
    emb = torch.cat([fr, fr], -1)
    return emb.cos() * scaling, emb.sin() * scaling


class CrossModelConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        cfg = self._kv_transfer_config
        self.block = vllm_config.cache_config.block_size
        self.tail = int(cfg.get_from_extra_config("tail", 64))
        self.min_len = int(cfg.get_from_extra_config("min_len", 1024))
        self.src_name = cfg.get_from_extra_config("src", None)
        self.maps_path = cfg.get_from_extra_config("maps", None)
        # optional calibration-mean K/V for the first A positions (attention sink), fit_eval sink_template.pt
        self.sink_path = cfg.get_from_extra_config("sink_template", None)
        self.restore = cfg.get_from_extra_config("restore", "rms")
        if self.restore not in ("rms", "none"):
            raise ValueError(f"kvx connector: restore={self.restore!r}; the serving path implements only 'rms' and 'none'")
        self.src_device = cfg.get_from_extra_config("src_device", None)
        self.fp8_source = bool(cfg.get_from_extra_config("fp8_source", False))
        self.tp_source = bool(cfg.get_from_extra_config("tp_source", False))  # shard the source prefill over TP ranks
        self.src_sdpa = str(cfg.get_from_extra_config("src_sdpa", "cudnn"))  # "cudnn" or "default"  # e4m3 linears in the source prefill
        self.chunk = int(cfg.get_from_extra_config("map_chunk_layers", 20))
        self.yarn = float(cfg.get_from_extra_config("yarn", 0.0))      # YaRN factor for BOTH models (e.g. 4 for 128K)
        self.orig_ctx = int(cfg.get_from_extra_config("orig_ctx", 32768))  # target layers per mapper GEMM chunk
        self.fp8 = bool(cfg.get_from_extra_config("fp8", False))  # e4m3 rowwise-scaled dense mapper GEMM  # e.g. "cuda:1": source + mapper off the target GPU
        self.tgt_name = vllm_config.model_config.model
        self.max_len = vllm_config.model_config.max_model_len
        self._need_load = {}
        self._local_hit = {}  # request id -> native prefix-cache hit (tokens) seen by get_num_new_matched_tokens
        self._worker_ready = False
        self.stats = dict(requests=0, mapped_tokens=0)

    # ------------------------------------------------------------ scheduler side
    def _n_mapped(self, n):
        if n < self.min_len:
            return 0
        return max(0, ((n - self.tail) // self.block) * self.block)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        m = self._n_mapped(len(request.prompt_token_ids or []))
        self._local_hit[request.request_id] = num_computed_tokens
        return max(0, m - num_computed_tokens), False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens > 0:
            self._need_load[request.request_id] = request

    def build_connector_meta(self, scheduler_output):
        meta = CrossModelMeta()
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id not in self._need_load:
                continue
            ids = list(new_req.prompt_token_ids or [])
            m = self._n_mapped(len(ids))
            blocks = torch.tensor(new_req.block_ids[0])
            slots = (torch.arange(self.block)[None, :] + blocks[:, None] * self.block).flatten()[:m]
            # write only past the LOCAL prefix-cache hit. new_req.num_computed_tokens already includes the tokens this
            # connector promised, so using it skipped the whole mapped prefix (72B served on empty KV, top1 12%)
            start = int(self._local_hit.pop(new_req.req_id, 0) or 0)  # the source still runs on ids[:m]
            meta.requests.append(_Req(token_ids=ids[:m], slot_mapping=slots, start=min(start, m)))
        self._need_load.clear()
        return meta

    def request_finished(self, request, block_ids):
        return False, None

    # ------------------------------------------------------------ worker side
    def _init_worker(self, kv_device):
        device = torch.device(self.src_device) if self.src_device else kv_device
        from transformers import AutoConfig, AutoModelForCausalLM
        from . import core, ttft
        self.core = core
        from .models import apply_yarn
        scfg = AutoConfig.from_pretrained(self.src_name)
        if getattr(self, "yarn", 0.0) > 0:
            apply_yarn(scfg, self.yarn, self.orig_ctx)
        if getattr(scfg, "text_config", None) is not None and "ConditionalGeneration" in (scfg.architectures or [""])[0]:
            # multimodal checkpoint (e.g. Ministral-3-3B, Mistral3ForConditionalGeneration): serve its text model only
            from transformers import AutoModelForImageTextToText
            from .models import TextLM
            if getattr(self, "yarn", 0.0) > 0:
                apply_yarn(scfg.text_config, self.yarn, self.orig_ctx)
            self.S = TextLM(AutoModelForImageTextToText.from_pretrained(
                self.src_name, config=scfg, dtype=torch.bfloat16, attn_implementation="sdpa")).to(device).eval()
        else:
            self.S = AutoModelForCausalLM.from_pretrained(self.src_name, config=scfg, dtype=torch.bfloat16,
                                                          attn_implementation="sdpa").to(device).eval()
        if getattr(self, "fp8_source", False):
            from .fp8 import quantize_linears
            nq = quantize_linears(self.S)
            logger.info("kvx connector: fp8 source, %d linear layers quantized", nq)
        self._src_group, self._src_world = None, 1
        if getattr(self, "tp_source", False):
            r_, w_ = _tp_rank_size(getattr(self, "tp_override", None))
            if w_ > 1:
                from vllm.distributed.parallel_state import get_tp_group
                from .tp_source import shard_source
                self._src_group, self._src_world = get_tp_group().device_group, w_
                shard_source(self.S, r_, w_, group=self._src_group)
                logger.info("kvx connector: source prefill sharded over %d TP ranks", w_)
        if getattr(self, "src_sdpa", "cudnn") == "vllm_fa":  # vLLM's own FA3/FA4 varlen kernel for the source prefill
            from .fa import use_vllm_fa
            use_vllm_fa(self.S)
            logger.info("kvx connector: source attention = vLLM flash-attn varlen")
        mp = torch.load(self.maps_path, map_location="cpu")
        for m in mp["k"] + mp["v"]:
            m["W16"], m["b16"] = m.pop("W").to(device, torch.bfloat16), m.pop("b").to(device, torch.bfloat16)
            for key in ("mu", "r2", "vnat"):
                m.pop(key, None)
        self.maps_k, self.maps_v = mp["k"], mp["v"]
        self.feat = mp.get("feat", "kv")  # "kvcat": each source layer feeds [K_pre_rope, V] to both maps
        self.ds = mp["k"][0]["W16"].shape[0] // len(mp["k"][0]["sel"])
        # dense mapper: unselected source blocks are zero; at k=24 of 28 source layers only 14% of FLOPs are wasted,
        # and one large GEMM per K/V runs near peak (grouped scatter-add was memory-bound: 1.2 s at 16K)
        Ls = self.S.config.num_hidden_layers
        self.Wk, self.bk = ttft.dense_weights(self.maps_k, Ls, self.ds, device)
        self.Wv, self.bv = ttft.dense_weights(self.maps_v, Ls, self.ds, device)
        for m in self.maps_k + self.maps_v:
            m.pop("W16", None)
        # tensor parallel: this rank owns KV heads [r*H/tp, (r+1)*H/tp) of every target layer -> keep only those columns
        H0, D0 = mp["k"][0]["H"], mp["k"][0]["D"]
        self.tp_rank, self.tp_size = _tp_rank_size(getattr(self, "tp_override", None))
        if self.tp_size > 1:
            assert H0 % self.tp_size == 0, (H0, self.tp_size)
            Hl = H0 // self.tp_size
            Lt_ = self.bk.numel() // (H0 * D0)
            cols = torch.cat([torch.arange(l * H0 * D0 + self.tp_rank * Hl * D0, l * H0 * D0 + (self.tp_rank + 1) * Hl * D0)
                              for l in range(Lt_)]).to(self.Wk.device)
            self.Wk, self.bk = self.Wk[:, cols].contiguous(), self.bk[cols].contiguous()
            self.Wv, self.bv = self.Wv[:, cols].contiguous(), self.bv[cols].contiguous()
            logger.info("kvx connector: TP rank %d/%d maps KV heads %d..%d", self.tp_rank, self.tp_size,
                        self.tp_rank * Hl, (self.tp_rank + 1) * Hl - 1)
        if self.fp8:
            def q(W):  # per output column scale, column-major e4m3 for torch._scaled_mm; chunked (no fp32 copy of W)
                sw = W.abs().amax(0, keepdim=True).float().clamp(min=1e-12) / 448.0
                out = torch.empty(W.shape[1], W.shape[0], dtype=torch.float8_e4m3fn, device=W.device)
                for c in range(0, W.shape[1], 4096):
                    out[c:c + 4096] = (W[:, c:c + 4096].float() / sw[:, c:c + 4096]).to(torch.float8_e4m3fn).t()
                return out.t(), sw
            self.Wk8, self.swk = q(self.Wk); del self.Wk
            self.Wv8, self.swv = q(self.Wv); del self.Wv
        torch.cuda.empty_cache()
        self.H, self.D = mp["k"][0]["H"] // self.tp_size, mp["k"][0]["D"]
        del mp
        tcfg = AutoConfig.from_pretrained(self.tgt_name)
        tcfg = getattr(tcfg, "text_config", None) or tcfg  # multimodal targets (Mistral Medium 3.5): text settings
        if getattr(self, "yarn", 0.0) > 0:
            apply_yarn(tcfg, self.yarn, self.orig_ctx)
        self.kw = None
        if self.restore == "rms":
            # target k_norm weights from the checkpoint (vLLM's module tree differs from HF's)
            import json, os
            from safetensors import safe_open

            def fetch(fname):  # vLLM may hand us a resolved local snapshot dir instead of a repo id
                if os.path.isdir(self.tgt_name):
                    p = os.path.join(self.tgt_name, fname)
                    return p if os.path.exists(p) else None
                from huggingface_hub import hf_hub_download
                try:
                    return hf_hub_download(self.tgt_name, fname)
                except Exception:
                    return None
            ix = fetch("model.safetensors.index.json")
            idx = json.load(open(ix))["weight_map"] if ix else None
            ws = []
            for l in range(tcfg.num_hidden_layers):
                name = f"model.layers.{l}.self_attn.k_norm.weight"
                with safe_open(fetch(idx[name] if idx else "model.safetensors"), "pt") as f:
                    ws.append(f.get_tensor(name))
            self.kw = torch.stack(ws).to(device, torch.bfloat16)
        Lmax = self.max_len
        self.cs, self.ss = core.rope_cos_sin(self.S, Lmax, device)
        # target RoPE from the config via transformers' own initializers: default, YaRN (Qwen3 at 128K),
        # llama3 scaling, ... (the target is loaded by vLLM, so its HF rotary module is not available here)
        self.ct, self.st = target_cos_sin(tcfg, Lmax, device)
        self.sinkK = self.sinkV = None
        if self.sink_path:
            st_ = torch.load(self.sink_path, map_location="cpu")  # K, V [Lt, H, A, D] (K pre-RoPE)
            A = st_["K"].shape[2]
            Kr = core.rope(st_["K"].float().to(device), self.ct[:A], self.st[:A])
            self.sinkK = Kr.permute(2, 0, 1, 3).to(torch.bfloat16).contiguous()       # [A, Lt, H, D] RoPE'd
            self.sinkV = st_["V"].float().to(device).permute(2, 0, 1, 3).to(torch.bfloat16).contiguous()
            if self.tp_size > 1:  # this rank's heads only
                h0 = self.tp_rank * self.H
                self.sinkK = self.sinkK[:, :, h0:h0 + self.H].contiguous()
                self.sinkV = self.sinkV[:, :, h0:h0 + self.H].contiguous()
        self._worker_ready = True

    @torch.no_grad()
    def _map_layers(self, ids):
        """Yield (layer, K [T,H,D] RoPE'd, V [T,H,D]) on the source device, one target layer at a time."""
        core = self.core
        device = self.cs.device
        x = torch.tensor([ids], device=device)
        # cuDNN SDPA first: 1.5x faster source prefill than torch's default pick (H100, 1.7B YaRN: 128K 7.35 -> 4.80 s)
        from torch.nn.attention import SDPBackend, sdpa_kernel
        order = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        if getattr(self, "src_sdpa", "cudnn") != "cudnn":
            order = order[1:]
        with sdpa_kernel(order, set_priority=True):
            out = self.S(input_ids=x, use_cache=True, logits_to_keep=1)
        T = len(ids)
        c_s, s_s = self.cs[:T].to(torch.bfloat16), self.ss[:T].to(torch.bfloat16)
        lay = core.cache_layers(out.past_key_values)
        if getattr(self, "_src_world", 1) > 1:  # each rank holds its source KV heads: gather all of them
            from .tp_source import gather_heads
            lay = [(gather_heads(k, self._src_world, self._src_group), gather_heads(v, self._src_world, self._src_group))
                   for k, v in lay]
        if self.feat == "kvcat":
            Xk = Xv = torch.cat([torch.cat([core.unrope(k, c_s, s_s)[0], v[0]], 0).transpose(0, 1).reshape(T, -1)
                                 for k, v in lay], 1)
        else:
            Xk = torch.cat([core.unrope(k, c_s, s_s)[0].transpose(0, 1).reshape(T, -1) for k, _ in lay], 1)
            Xv = torch.cat([v[0].transpose(0, 1).reshape(T, -1) for _, v in lay], 1)
        del out, lay
        ds, H, D = self.ds, self.H, self.D
        HD = H * D
        Lt = (self.bk.numel()) // HD
        if self.fp8:
            def mm8(X, W8, sw, b):
                mu = X.float().mean(0, keepdim=True)
                Xc = X.float() - mu
                sx = Xc.abs().amax(1, keepdim=True).clamp(min=1e-12) / 448.0
                X8 = (Xc / sx).to(torch.float8_e4m3fn)
                del Xc
                m16 = mu.expand(16, -1)  # mean term through the same fp8 GEMM (min 16 rows)
                sm = m16.abs().amax(1, keepdim=True).clamp(min=1e-12) / 448.0
                m8 = (m16 / sm).to(torch.float8_e4m3fn)
                def f(c0, c1):  # output columns [c0, c1) = target layers of one chunk
                    Y = torch._scaled_mm(X8, W8[:, c0:c1], scale_a=sx, scale_b=sw[:, c0:c1], out_dtype=torch.bfloat16)
                    muW = torch._scaled_mm(m8, W8[:, c0:c1], scale_a=sm, scale_b=sw[:, c0:c1], out_dtype=torch.float32)[:1]
                    return Y + (muW + b[c0:c1].float()).to(torch.bfloat16)
                return f
            fk = mm8(Xk, self.Wk8, self.swk, self.bk)
            fv = mm8(Xv, self.Wv8, self.swv, self.bv)
        else:
            fk = lambda c0, c1: torch.addmm(self.bk[c0:c1], Xk, self.Wk[:, c0:c1])
            fv = lambda c0, c1: torch.addmm(self.bv[c0:c1], Xv, self.Wv[:, c0:c1])
        from .triton_kernels import restore_rope_
        ct_, st_ = self.ct[:T].contiguous(), self.st[:T].contiguous()
        # target layers in chunks: peak mapped memory is 2*T*chunk*H*D bf16 instead of 2*T*Lt*H*D (10 GiB at 32K on 72B)
        for l0 in range(0, Lt, self.chunk):
            l1 = min(Lt, l0 + self.chunk)
            Yk4 = fk(l0 * HD, l1 * HD).view(T, l1 - l0, H, D)
            if not Yk4.is_contiguous():
                Yk4 = Yk4.contiguous()
            Yv4 = fv(l0 * HD, l1 * HD).view(T, l1 - l0, H, D)
            # fused k_norm RMS restore + RoPE over the chunk's layers in one Triton pass (in place)
            restore_rope_(Yk4, None if self.kw is None else self.kw[l0:l1].contiguous(), ct_, st_)
            if self.sinkK is not None:  # sink positions: fixed native template (input-invariant), not the mapped KV
                A = min(self.sinkK.shape[0], T)
                Yk4[:A] = self.sinkK[:A, l0:l1]
                Yv4[:A] = self.sinkV[:A, l0:l1]
            for l in range(l0, l1):
                yield l, Yk4[:, l - l0], Yv4[:, l - l0]
            del Yk4, Yv4

    def start_load_kv(self, forward_context, **kwargs: Any) -> None:
        meta = self._get_connector_metadata()
        if not isinstance(meta, CrossModelMeta) or not meta.requests:
            return
        layers = {}
        for name, layer in forward_context.no_compile_layers.items():
            kv = getattr(layer, "kv_cache", None)
            m = _LAYER_RE.search(name)
            if kv is None or m is None:
                continue
            kv = kv[0] if isinstance(kv, (list, tuple)) else kv
            if not layers:  # the paged write below assumes FlashAttention's [blocks, H, block, 2D] row split
                be = getattr(layer, "attn_backend", None)
                bname = be.get_name() if hasattr(be, "get_name") else type(getattr(layer, "impl", None)).__name__
                logger.info("kvx connector: backend=%s kv shape=%s stride=%s", bname, tuple(kv.shape), kv.stride())
                if "FLASHINFER" in str(bname).upper():
                    raise RuntimeError(f"kvx connector: attention backend {bname} reads each KV block as [2, H, block, D]; "
                                       "the connector writes FlashAttention's row split. Start vLLM with attention_backend='FLASH_ATTN'")
            layers[int(m.group(1))] = kv
        if not layers:
            raise RuntimeError("kvx connector: no attention layer with a kv_cache matched layers.<i>. in "
                               "forward_context.no_compile_layers; the scheduler already promised mapped tokens")
        device = next(iter(layers.values())).device
        if not self._worker_ready:
            self._init_worker(device)
        import time as _t
        for r in meta.requests:
            src_dev = self.cs.device
            torch.cuda.synchronize(src_dev); t0 = _t.perf_counter()
            s0 = r.start
            slots = r.slot_mapping[s0:].to(device)
            bi, off = slots // self.block, slots % self.block
            D = self.D
            t_map = 0.0; t_write = 0.0; t_first = None
            gen = self._map_layers(r.token_ids)
            while True:
                ta = _t.perf_counter()
                try:
                    l, K, V = next(gen)
                except StopIteration:
                    break
                tb = _t.perf_counter()
                if t_first is None:
                    t_first = tb - t0  # includes source prefill + first layer map
                else:
                    t_map += tb - ta
                kv = layers.get(l)
                if kv is None:
                    raise RuntimeError(f"kvx connector: target layer {l} has no KV cache (maps fit on a different depth?)")
                if s0:
                    K, V = K[s0:], V[s0:]
                K, V = K.to(kv.device, non_blocking=True), V.to(kv.device, non_blocking=True)
                if kv.dim() == 4 and kv.shape[-1] == 2 * D:
                    # vLLM 0.34 flash-attn: [num_blocks, H, block_size, 2*D], K in [..., :D], V in [..., D:]
                    write_kv_fa(kv, bi, off, K, V, self.block)
                elif kv.shape[0] == 2:
                    # older flash-attn layout [2, num_blocks, block_size, H, D]
                    kv[0][bi, off] = K.to(kv.dtype)
                    kv[1][bi, off] = V.to(kv.dtype)
                else:
                    raise RuntimeError(f"unknown KV cache layout {tuple(kv.shape)}")
                t_write += _t.perf_counter() - tb
            torch.cuda.synchronize(device); torch.cuda.synchronize(src_dev)
            logger.info("kvx connector: n=%d src+first_layer=%.3fs map_rest=%.3fs transfer+write=%.3fs total=%.3fs",
                        len(r.token_ids), t_first or 0.0, t_map, t_write, _t.perf_counter() - t0)
            self.stats["requests"] += 1
            self.stats["mapped_tokens"] += len(r.token_ids)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        return

    def wait_for_save(self):
        return
