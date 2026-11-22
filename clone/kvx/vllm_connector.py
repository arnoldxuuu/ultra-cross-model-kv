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
import os as _os
def _sync(dev):
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize(dev)


_BIAS_IN_GEMM = [True]  # torch._scaled_mm(bias=) with row-wise scales; flipped off at the first refusal


def _mm8_bias(A, W, sa, sw, bias):
    """e4m3 GEMM with the per-request bias (mean term + map bias) inside the GEMM epilogue (saves one pass over Y)."""
    if _BIAS_IN_GEMM[0]:
        try:
            return torch._scaled_mm(A, W, scale_a=sa, scale_b=sw, bias=bias, out_dtype=torch.bfloat16)
        except Exception:
            _BIAS_IN_GEMM[0] = False
    return torch._scaled_mm(A, W, scale_a=sa, scale_b=sw, out_dtype=torch.bfloat16) + bias


_SYNC = _os.environ.get("KVX_SYNC_COMPONENTS", "1") == "1"  # sync at component boundaries (exact split, ~1 ms cost)


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


def _tp_group(override=None):
    """torch.distributed process group of vLLM's TP ranks (device group, NCCL on GPU)."""
    if override is not None:
        return override
    from vllm.distributed.parallel_state import get_tp_group
    return get_tp_group().device_group


def kv_layout(kv, H, D, backend=""):
    """Classify a vLLM paged KV tensor. fa: [blocks, H, block, 2D] (FLASH_ATTN, K|V split per row);
    split5: [blocks, 2, block, H, D] (TRITON_ATTN, ROCM_AITER_FA); split4: [blocks, 2, block, H*D] (ROCM_ATTN);
    lead2: [2, blocks, block, H, D] (older flash-attn). The backend name settles the H == 2 ambiguity of fa vs split4."""
    b = str(backend).upper()
    sh = tuple(kv.shape)
    if kv.dim() == 4 and H == 2 and sh[1] == 2 and sh[3] == 2 * D and not b:
        raise RuntimeError(f"kvx connector: KV cache {sh} is fa or split4 when H == 2; pass the attention backend name")
    if kv.dim() == 4 and sh[1] == H and sh[3] == 2 * D and (H != 2 or "FLASH_ATTN" in b):
        return "fa"
    if kv.dim() == 5 and sh[1] == 2 and sh[3:] == (H, D):
        return "split5"
    if kv.dim() == 4 and sh[1] == 2 and sh[3] == H * D:
        return "split4"
    if kv.dim() == 5 and sh[0] == 2 and sh[3:] == (H, D):
        return "lead2"
    raise RuntimeError(f"kvx connector: unknown KV cache layout {sh} for backend {backend} (H={H}, D={D})")


def write_kv(kv, bi, off, K, V, block, layout):
    """Write K, V [n, H, D] at paged slots (bi, off) for any layout from kv_layout."""
    if layout == "fa":
        return write_kv_fa(kv, bi, off, K, V, block)
    n = K.shape[0]
    if layout == "split5":
        kv[bi, 0, off] = K.to(kv.dtype)
        kv[bi, 1, off] = V.to(kv.dtype)
    elif layout == "split4":
        kv[bi, 0, off] = K.reshape(n, -1).to(kv.dtype)
        kv[bi, 1, off] = V.reshape(n, -1).to(kv.dtype)
    elif layout == "lead2":
        kv[0][bi, off] = K.to(kv.dtype)
        kv[1][bi, off] = V.to(kv.dtype)
    else:
        raise ValueError(layout)


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
    cfg = getattr(cfg, "text_config", None) or cfg  # multimodal checkpoints (Mistral-Medium 3.5, Gemma): the text decoder
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
        self.src_sdpa = str(cfg.get_from_extra_config("src_sdpa", "cudnn"))  # "cudnn" or "default"  # e4m3 linears in the source prefill
        self.chunk = int(cfg.get_from_extra_config("map_chunk_layers", 20))
        self.yarn = float(cfg.get_from_extra_config("yarn", 0.0))      # YaRN factor for BOTH models (e.g. 4 for 128K)
        self.orig_ctx = int(cfg.get_from_extra_config("orig_ctx", 32768))  # target layers per mapper GEMM chunk
        self.fp8 = bool(cfg.get_from_extra_config("fp8", False))
        # TP>1: shard the source model over the TP ranks (1/n of its FLOPs each) instead of running it whole on each
        self.src_tp = bool(int(cfg.get_from_extra_config("src_tp", 1)))
        # token rows per mapper pass (0 = whole prompt); bounds connector memory at 256K-1M
        self.tok_chunk = int(cfg.get_from_extra_config("map_chunk_tokens", 65536))
        # source prefill in chunks of this many tokens (0 = one forward); needs src_sdpa=vllm_fa (no mask, causal
        # bottom-right). Bounds the source's activations at 512K-1M. src_fp8_attn: FA3 e4m3 attention in the source
        self.src_chunk = int(cfg.get_from_extra_config("src_chunk", 0))
        self.src_fp8_attn = bool(int(cfg.get_from_extra_config("src_fp8_attn", 0)))
        self.src_window = int(cfg.get_from_extra_config("src_window", 0))  # sliding window for the source (vllm_fa)
        self.src_full_last = int(cfg.get_from_extra_config("src_full_last", 0))  # with src_window + src_lean: last N layers full
        # "dense": one GEMM over all source layers (zero blocks for unselected); "sparse": per target layer, only its k
        # selected source layers (gather + GEMM): the fast mapper for small-k maps
        self.mapper = str(cfg.get_from_extra_config("mapper", "dense"))
        self.src_lean = int(cfg.get_from_extra_config("src_lean", 0))  # 1: lean source forward, 2: + torch.compile
        self.async_load = bool(int(cfg.get_from_extra_config("async_load", 0)))  # side stream + per-layer events
        self._ev = {}
        self.group = int(cfg.get_from_extra_config("group", 4))
        self.fused_build = bool(int(cfg.get_from_extra_config("fused_build", 1)))  # Triton build+e4m3 of the mapper input  # target layers per group (mapper="group")
        self.mem_fraction = float(cfg.get_from_extra_config("mem_fraction", 0.97))  # connector memory beyond vLLM's share  # e4m3 rowwise-scaled dense mapper GEMM  # e.g. "cuda:1": source + mapper off the target GPU
        self.tgt_name = vllm_config.model_config.model
        self.max_len = vllm_config.model_config.max_model_len
        self._need_load = {}
        self._m_req = {}  # request id -> mapped tokens decided in get_num_new_matched_tokens (per-request tail)
        self._local_hit = {}  # request id -> native prefix-cache hit (tokens) seen by get_num_new_matched_tokens
        self._worker_ready = False
        self._layout = None  # paged KV layout, classified at the first write (kv_layout)
        self._backend = ""
        self.stats = dict(requests=0, mapped_tokens=0)

    # ------------------------------------------------------------ scheduler side
    def _n_mapped(self, n, tail=None):
        if n < self.min_len:
            return 0
        t = self.tail if tail is None else max(1, int(tail))
        return max(0, ((n - t) // self.block) * self.block)

    @staticmethod
    def _req_tail(request):
        """Per-request native tail: SamplingParams(extra_args={"kv_transfer_params": {"kvx_tail": N}}), e.g. the
        question length so the document is mapped and the question read natively. None -> the engine-wide tail."""
        p = getattr(request, "kv_transfer_params", None) or {}
        return p.get("kvx_tail") if isinstance(p, dict) else None

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        m = self._n_mapped(len(request.prompt_token_ids or []), self._req_tail(request))
        self.__dict__.setdefault("_m_req", {})[request.request_id] = m
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
            m = self.__dict__.setdefault("_m_req", {}).pop(new_req.req_id, None)
            m = self._n_mapped(len(ids)) if m is None else m
            blocks = torch.tensor(new_req.block_ids[0])
            slots = (torch.arange(self.block)[None, :] + blocks[:, None] * self.block).flatten()[:m]
            # write only past the LOCAL prefix-cache hit. new_req.num_computed_tokens already includes the tokens this
            # connector promised, so using it skipped the whole mapped prefix (72B served on empty KV, top1 12%)
            start = int(self._local_hit.pop(new_req.req_id, 0) or 0)  # the source still runs on ids[:m]
            meta.requests.append(_Req(token_ids=ids[:m], slot_mapping=slots, start=min(start, m)))
        self._need_load.clear()
        return meta

    def request_finished(self, request, block_ids):
        self.__dict__.setdefault("_m_req", {}).pop(request.request_id, None)
        return False, None

    # ------------------------------------------------------------ worker side
    def _init_worker(self, kv_device):
        device = torch.device(self.src_device) if self.src_device else kv_device
        if device.type == "cuda":
            # vLLM caps the worker process at gpu_memory_utilization (torch per-process fraction) and fills that share
            # with weights + KV pool; the connector's buffers live in the rest of the GPU (32B TP2 128K on H100 OOMed
            # at "55.64 GiB allowed" with 24 GB physically free)
            frac = float(getattr(self, "mem_fraction", 0.97))
            if frac > 0:
                torch.cuda.set_per_process_memory_fraction(frac, device)
        from transformers import AutoConfig, AutoModelForCausalLM
        from . import core, ttft
        self.core = core
        from .models import apply_yarn
        scfg = AutoConfig.from_pretrained(self.src_name)
        if getattr(self, "yarn", 0.0) > 0:
            apply_yarn(scfg, self.yarn, self.orig_ctx)
        if getattr(scfg, "text_config", None) is not None and "ConditionalGeneration" in (scfg.architectures or [""])[0]:
            # multimodal checkpoint (Ministral 3, Gemma 3/4): the text decoder behind the TextLM view (vision unused)
            from transformers import AutoModelForImageTextToText
            from .models import TextLM
            if getattr(self, "yarn", 0.0) > 0:
                apply_yarn(scfg.text_config, self.yarn, self.orig_ctx)
            self.S = TextLM(AutoModelForImageTextToText.from_pretrained(
                self.src_name, config=scfg, dtype=torch.bfloat16, attn_implementation="sdpa")).to(device).eval()
        else:
            self.S = AutoModelForCausalLM.from_pretrained(self.src_name, config=scfg, dtype=torch.bfloat16,
                                                          attn_implementation="sdpa").to(device).eval()
        self.tp_rank, self.tp_size = _tp_rank_size(getattr(self, "tp_override", None))
        self.src_tp_on = False
        if self.tp_size > 1 and getattr(self, "src_tp", True):
            from .tp_source import shard_source
            shard_source(self.S, self.tp_rank, self.tp_size, _tp_group(getattr(self, "tp_group_override", None)))
            self.src_tp_on = True
            logger.info("kvx connector: source prefill sharded over TP=%d (rank %d)", self.tp_size, self.tp_rank)
        if getattr(self, "fp8_source", False):
            from .fp8 import quantize_linears
            nq = quantize_linears(self.S)
            logger.info("kvx connector: fp8 source, %d linear layers quantized", nq)
        if getattr(self, "src_sdpa", "cudnn") == "vllm_fa":  # vLLM's own FA3/FA4 varlen kernel for the source prefill
            from .fa import use_vllm_fa
            use_vllm_fa(self.S, fp8=getattr(self, "src_fp8_attn", False), window=getattr(self, "src_window", 0))
            logger.info("kvx connector: source attention = vLLM flash-attn varlen (fp8=%s)", getattr(self, "src_fp8_attn", False))
        elif getattr(self, "src_fp8_attn", False) or getattr(self, "src_chunk", 0) or getattr(self, "src_window", 0):
            raise ValueError("kvx connector: src_fp8_attn / src_chunk need src_sdpa=vllm_fa")
        self.lean = None
        if int(getattr(self, "src_lean", 0) or 0):
            from .lean_source import LeanSource
            self.lean = LeanSource(self.S, compile=int(self.src_lean) >= 2)
            logger.info("kvx connector: lean source forward (compile=%s)", int(self.src_lean) >= 2)
        mp = torch.load(self.maps_path, map_location="cpu")
        for m in mp["k"] + mp["v"]:
            # maps stay on the host until the dense weights are built and sliced to this TP rank's columns (the
            # full 26 x 88-layer Mistral-Medium mapper, 4.5 GB, OOMed next to vLLM's share when built on the GPU)
            m["W16"], m["b16"] = m.pop("W").to(torch.bfloat16), m.pop("b").to(torch.bfloat16)
            for key in ("mu", "r2", "vnat"):
                m.pop(key, None)
        self.maps_k, self.maps_v = mp["k"], mp["v"]
        self.xtok = None
        if mp.get("align") is not None:  # cross-tokenizer maps (research/xfit.py): both tokenizers, source BOS
            from transformers import AutoTokenizer
            self.xtok = (AutoTokenizer.from_pretrained(self.tgt_name), AutoTokenizer.from_pretrained(self.src_name))
            self.xbos_s = mp.get("bos_s")
            logger.info("kvx connector: cross-tokenizer maps (align=%s, source BOS %s)", mp.get("align"), self.xbos_s)
        self.feat = mp.get("feat", "kv")  # "kvcat": each source layer feeds [K_pre_rope, V] to both maps
        self.ds = mp["k"][0]["W16"].shape[0] // len(mp["k"][0]["sel"])
        # dense mapper: unselected source blocks are zero; at k=24 of 28 source layers only 14% of FLOPs are wasted,
        # and one large GEMM per K/V runs near peak (grouped scatter-add was memory-bound: 1.2 s at 16K)
        Ls = self.S.config.num_hidden_layers
        self.Wk, self.bk = ttft.dense_weights(self.maps_k, Ls, self.ds, "cpu")
        self.Wv, self.bv = ttft.dense_weights(self.maps_v, Ls, self.ds, "cpu")
        for m in self.maps_k + self.maps_v:
            m.pop("W16", None)
        # tensor parallel: this rank owns KV heads [r*H/tp, (r+1)*H/tp) of every target layer -> keep only those columns
        H0, D0 = mp["k"][0]["H"], mp["k"][0]["D"]
        mp_H0D0 = (H0, D0)
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
        self.Wk, self.bk, self.Wv, self.bv = (x.to(device) for x in (self.Wk, self.bk, self.Wv, self.bv))
        if self.fp8:
            def q(W):  # per output column scale, column-major e4m3 for torch._scaled_mm; chunked (no fp32 copy of W)
                sw = W.abs().amax(0, keepdim=True).float().clamp(min=1e-12) / 448.0
                out = torch.empty(W.shape[1], W.shape[0], dtype=torch.float8_e4m3fn, device=W.device)
                for c in range(0, W.shape[1], 4096):
                    out[c:c + 4096] = (W[:, c:c + 4096].float() / sw[:, c:c + 4096]).to(torch.float8_e4m3fn).t()
                return out.t(), sw
            self.Wk8, self.swk = q(self.Wk); del self.Wk
            self.Wv8, self.swv = q(self.Wv); del self.Wv
        self.sp = None
        if getattr(self, "mapper", "dense") == "group":
            # chunks of G target layers: one gather of the UNION of their selected source layers (~10 of 28 for k8 maps,
            # G=4) and one GEMM whose weight holds zero blocks where a layer did not select a source layer. FLOPs ~
            # |union|/Ls of dense, gathers amortized over G layers, output already in the chunk's [n, G, H, D] layout
            G = int(getattr(self, "group", 4) or 4)
            self.chunk = G
            HDl = (H0 // self.tp_size) * D0
            u8 = torch.uint8

            def per_group(maps, W, sw, b):
                out = []
                for g0 in range(0, len(maps), G):
                    ls = list(range(g0, min(len(maps), g0 + G)))
                    U = sorted(set().union(*[set(maps[l]["sel"]) for l in ls]))
                    sel = torch.tensor(U, device=b.device)
                    rows = torch.cat([torch.arange(s_ * self.ds, (s_ + 1) * self.ds, device=b.device) for s_ in U])
                    cols = torch.arange(g0 * HDl, (ls[-1] + 1) * HDl, device=b.device)
                    if self.fp8:
                        Wg = W.t().view(u8).index_select(0, cols).index_select(1, rows).contiguous()
                        out.append((sel, Wg.view(torch.float8_e4m3fn).t(), sw[:, cols].contiguous(), b[cols].contiguous()))
                    else:
                        out.append((sel, W.index_select(0, rows).index_select(1, cols).contiguous(), None, b[cols].contiguous()))
                return out
            if self.fp8:
                self.sp = dict(k=per_group(self.maps_k, self.Wk8, self.swk, self.bk), v=per_group(self.maps_v, self.Wv8, self.swv, self.bv), G=G)
                del self.Wk8, self.Wv8
            else:
                self.sp = dict(k=per_group(self.maps_k, self.Wk, None, self.bk), v=per_group(self.maps_v, self.Wv, None, self.bv), G=G)
                del self.Wk, self.Wv
            un = sum(len(x[0]) for x in self.sp["k"]) / len(self.sp["k"])
            logger.info("kvx connector: group mapper G=%d, mean union %.1f source layers per chunk (of %d)", G, un, Ls)
        elif getattr(self, "mapper", "dense") == "sparse":
            # per target layer: only its k selected source layers (FLOPs ~ k/Ls of dense; for small-k maps).
            # Built from the dense (TP-sliced, maybe e4m3) weights: rows of the selected source blocks, the layer's columns
            HDl = (H0 // self.tp_size) * D0
            u8 = torch.uint8

            def per_layer(maps, W, sw, b):
                out = []
                for l, m in enumerate(maps):
                    sel = torch.tensor(sorted(m["sel"]), device=b.device)
                    rows = torch.cat([torch.arange(s_ * self.ds, (s_ + 1) * self.ds, device=b.device) for s_ in sel.tolist()])
                    cols = torch.arange(l * HDl, (l + 1) * HDl, device=b.device)
                    if self.fp8:  # W is column-major e4m3 [R, C]: slice its contiguous transpose as bytes
                        Wl = W.t().view(u8).index_select(0, cols).index_select(1, rows).contiguous()
                        Wl = Wl.view(torch.float8_e4m3fn).t()
                        out.append((sel, Wl, sw[:, cols].contiguous(), b[cols].contiguous()))
                    else:
                        out.append((sel, W.index_select(0, rows).index_select(1, cols).contiguous(), None, b[cols].contiguous()))
                return out
            if self.fp8:
                self.sp = dict(k=per_layer(self.maps_k, self.Wk8, self.swk, self.bk), v=per_layer(self.maps_v, self.Wv8, self.swv, self.bv))
                del self.Wk8, self.Wv8
            else:
                self.sp = dict(k=per_layer(self.maps_k, self.Wk, None, self.bk), v=per_layer(self.maps_v, self.Wv, None, self.bv))
                del self.Wk, self.Wv
            kmean = sum(len(x[0]) for x in self.sp["k"]) / len(self.sp["k"])
            logger.info("kvx connector: sparse mapper, mean %.1f source layers per target layer (of %d)", kmean, Ls)
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
        # source tables: a cross-tokenizer source can need more positions than the target prompt has tokens
        self.cs, self.ss = core.rope_cos_sin(self.S, 2 * Lmax + 8 if getattr(self, "xtok", None) else Lmax, device)
        # target RoPE from the config via transformers' own initializers: default, YaRN (Qwen3 at 128K),
        # llama3 scaling, ... (the target is loaded by vLLM, so its HF rotary module is not available here)
        self.ct, self.st = target_cos_sin(tcfg, Lmax, device)
        self.sinkK = self.sinkV = None
        if self.sink_path:
            st_ = torch.load(self.sink_path, map_location="cpu")  # K, V [Lt, H, A, D] (K pre-RoPE)
            if "K" not in st_:  # kvx-pairs' per-layer format {layer: {"k": [A, H*D], "v": [A, H*D]}} (pre-RoPE K)
                H0_, D0_ = mp_H0D0
                ls_ = sorted(int(t) for t in st_)
                st_ = {X.upper(): torch.stack([st_[t][X].float().view(-1, H0_, D0_).permute(1, 0, 2) for t in ls_])
                       for X in ("k", "v")}
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
    def _map_chunks(self, ids):
        """Yield (token offset t0, layer, K [n,H,D] RoPE'd, V [n,H,D]) on the source device: token chunks of
        tok_chunk rows (all rows by default), one target layer at a time inside each."""
        core = self.core
        device = self.cs.device
        idx = None
        if getattr(self, "xtok", None) is not None:  # cross-tokenizer pair: the source runs on its own tokens
            s_ids, idx = self._xalign(ids)
            x = torch.tensor([s_ids], device=device)
            idx = idx.to(device)
        else:
            x = torch.tensor([ids], device=device)
        # cuDNN SDPA first: 1.5x faster source prefill than torch's default pick (H100, 1.7B YaRN: 128K 7.35 -> 4.80 s)
        from torch.nn.attention import SDPBackend, sdpa_kernel
        order = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        if getattr(self, "src_sdpa", "cudnn") != "cudnn" or torch.version.hip or getattr(self, "_cudnn_bad", False):
            order = order[1:]  # no cuDNN on ROCm, or it already failed in this session
        sc = int(getattr(self, "src_chunk", 0) or 0)
        lean = getattr(self, "lean", None)
        if lean is not None:  # plain-tensor forward: pre-RoPE K and V straight into per-layer buffers
            from .lean_source import fa_attn, sdpa_attn
            W, fp8a = getattr(self, "src_window", 0), getattr(self, "src_fp8_attn", False)
            nfull = int(getattr(self, "src_full_last", 0) or 0)  # last N source layers keep FULL attention (recall)
            Ls_src = len(lean.layers)
            if getattr(self, "src_sdpa", "cudnn") == "vllm_fa" and device.type == "cuda":
                mk = lambda w: fa_attn(window=w, fp8=fp8a)
            else:
                mk = lambda w: (lambda q, k, v, sc_: sdpa_attn(q, k, v, sc_, window=w))
            if W and nfull:
                fw, ff = mk(W), mk(0)
                attn = lambda li: ff if li >= Ls_src - nfull else fw
                attn.per_layer = True
            else:
                attn = mk(W)
            Ts = x.shape[1]
            lay_pre = lean.forward(x[0], self.cs[:Ts], self.ss[:Ts], attn, chunk=sc)

        def _prefill(order_):
            with sdpa_kernel(order_, set_priority=True):
                if lean is not None:
                    return None
                if sc and x.shape[1] > sc:
                    from transformers import DynamicCache
                    pkv, o_ = DynamicCache(), None
                    for a in range(0, x.shape[1], sc):
                        o_ = self.S(input_ids=x[:, a:a + sc], past_key_values=pkv, use_cache=True, logits_to_keep=1)
                    return o_
                return self.S(input_ids=x, use_cache=True, logits_to_keep=1)
        try:
            out = _prefill(order)
        except RuntimeError as e:  # sm_120 (RTX PRO 6000) + this cuDNN: fails at run time, e.g. SUBLIBRARY_LOADING_FAILED
            if order[0] != SDPBackend.CUDNN_ATTENTION or "cuDNN" not in str(e):
                raise
            self._cudnn_bad = True
            logger.warning("kvx connector: cuDNN SDPA failed (%s); source prefill falls back for this session", str(e)[:120])
            out = _prefill(order[1:])
        T = len(ids)
        c_s, s_s = self.cs[:T].to(torch.bfloat16), self.ss[:T].to(torch.bfloat16)
        pre = lean is not None  # lay holds PRE-RoPE keys: nothing to unrope downstream
        lay = lay_pre if pre else core.cache_layers(out.past_key_values)
        del out
        unr = (lambda k_, c_, s_: k_) if pre else core.unrope
        if idx is not None and pre:
            lay = [(k.index_select(2, idx), v.index_select(2, idx)) for k, v in lay]
        elif idx is not None:
            # target position j takes the aligned source token idx[j]: strip RoPE at its SOURCE position, re-rope at j
            # with the source tables, so the rest of the pipeline (which unropes row j at position j) is unchanged
            Ts = x.shape[1]
            cs_s, ss_s = self.cs[:Ts], self.ss[:Ts]
            cf, sf = self.cs[:T], self.ss[:T]
            lay = [(core.rope(core.unrope(k.float(), cs_s, ss_s).index_select(2, idx), cf, sf).to(k.dtype),
                    v.index_select(2, idx)) for k, v in lay]
        if device.type == "cuda" and getattr(self, "_sync_on", _SYNC):  # component timing: the source prefill ends here
            torch.cuda.synchronize(device)
        import time as _t
        self._t_src_done = _t.perf_counter()
        tp = self.S._kvx_tp if getattr(self, "src_tp_on", False) else None
        if tp is not None:  # this rank holds its source KV heads only: all-gather the heads the mapper reads
            from .tp_source import gather_heads
            ga = lambda x: gather_heads(x, tp["world"], tp["group"], tp["rep"], 0)  # [Hl, n, D] -> [H, n, D]
        else:
            ga = lambda x: x
        Ls_, k0 = len(lay), lay[0][0]
        Hs_, Ds_ = k0.shape[1] * (tp["world"] // tp["rep"] if tp else 1), k0.shape[-1]

        def build(t0, t1):
            """Mapper input rows [t0, t1): [n, Ls*Hs*D] (kvcat: [n, Ls*2*Hs*D]), one layer at a time, no big transients."""
            n = t1 - t0
            c_s, s_s = self.cs[t0:t1].to(torch.bfloat16), self.ss[t0:t1].to(torch.bfloat16)
            if self.feat == "kvcat":
                Xb = k0.new_empty(n, Ls_, 2, Hs_, Ds_)
                for l, (k, v) in enumerate(lay):
                    Xb[:, l, 0] = ga(unr(k[:, :, t0:t1], c_s, s_s)[0].contiguous()).transpose(0, 1)
                    Xb[:, l, 1] = ga(v[0, :, t0:t1].contiguous()).transpose(0, 1)
                X = Xb.view(n, -1)
                return X, X
            Xbk, Xbv = k0.new_empty(n, Ls_, Hs_, Ds_), k0.new_empty(n, Ls_, Hs_, Ds_)
            for l, (k, v) in enumerate(lay):
                Xbk[:, l] = ga(unr(k[:, :, t0:t1], c_s, s_s)[0].contiguous()).transpose(0, 1)
                Xbv[:, l] = ga(v[0, :, t0:t1].contiguous()).transpose(0, 1)
            return Xbk.view(n, -1), Xbv.view(n, -1)

        def segments(t0, t1):
            """Fused-build inputs for rows [t0, t1): ([HS, n, D] post-RoPE view, first column, unrope) per K/V segment."""
            HsD = Hs_ * Ds_
            if self.feat == "kvcat":
                seg = []
                for l, (k, v) in enumerate(lay):
                    seg.append((ga(k[0, :, t0:t1].contiguous()) if tp is not None else k[0, :, t0:t1], 2 * l * HsD, not pre))
                    seg.append((ga(v[0, :, t0:t1].contiguous()) if tp is not None else v[0, :, t0:t1], (2 * l + 1) * HsD, False))
                return seg, seg
            sk = [(ga(k[0, :, t0:t1].contiguous()) if tp is not None else k[0, :, t0:t1], l * HsD, not pre) for l, (k, _) in enumerate(lay)]
            sv = [(ga(v[0, :, t0:t1].contiguous()) if tp is not None else v[0, :, t0:t1], l * HsD, False) for l, (_, v) in enumerate(lay)]
            return sk, sv

        ds, H, D = self.ds, self.H, self.D
        HD = H * D
        Lt = (self.bk.numel()) // HD
        if self.fp8:
            def quant8(X, mu, rows=8192):  # centered rowwise e4m3 in row chunks: no fp32 copy of X
                X8 = torch.empty(X.shape, dtype=torch.float8_e4m3fn, device=X.device)
                sx = torch.empty(X.shape[0], 1, dtype=torch.float32, device=X.device)
                for r in range(0, X.shape[0], rows):
                    Xc = X[r:r + rows].float() - mu
                    sx[r:r + rows] = Xc.abs().amax(1, keepdim=True).clamp(min=1e-12) / 448.0
                    X8[r:r + rows] = (Xc / sx[r:r + rows]).to(torch.float8_e4m3fn)
                return X8, sx

            def mean8(mu):  # the centering vector through the same fp8 GEMM (min 16 rows)
                m16 = mu.expand(16, -1)
                sm = m16.abs().amax(1, keepdim=True).clamp(min=1e-12) / 448.0
                return (m16 / sm).to(torch.float8_e4m3fn), sm

            def mm8(X8, sx, m8, sm, W8, sw, b):
                def f(c0, c1):  # output columns [c0, c1) = target layers of one chunk
                    muW = torch._scaled_mm(m8, W8[:, c0:c1], scale_a=sm, scale_b=sw[:, c0:c1], out_dtype=torch.float32)[:1]
                    return _mm8_bias(X8, W8[:, c0:c1], sx, sw[:, c0:c1], (muW[0] + b[c0:c1].float()).to(torch.bfloat16))
                return f
        def sparse_f(lays, n, q=None, m8=None, sm=None, X=None):
            """Producer of target-layer column blocks [c0, c1) from per-layer (sel, W, sw, b): gather the selected
            source blocks of the (e4m3 or bf16) mapper input, one GEMM per target layer."""
            u8 = torch.uint8

            G = sp.get("G") if sp is not None else None

            def f(c0, c1):
                l0, l1 = c0 // HD, c1 // HD
                if G:  # group mapper: the chunk [l0, l1) is exactly one group entry
                    assert l0 % G == 0 and l1 - l0 <= G, (l0, l1, G)
                    groups = [(lays[l0 // G], slice(None))]
                    out = None
                else:
                    groups = [(lays[l], slice(j * HD, (j + 1) * HD)) for j, l in enumerate(range(l0, l1))]
                    out = torch.empty(n, (l1 - l0) * HD, dtype=torch.bfloat16, device=device)
                for (sel, W, sw, b), cs in groups:
                    # gather whole (token, source layer) rows: index_select on dim 0 of [n*Ls, ds] takes the fast
                    # row-copy path; dim 1 of [n, Ls, ds] ran indexSelectSmallIndex (132 ms at 32K, > the GEMM)
                    idx = (torch.arange(n, device=sel.device)[:, None] * Ls_ + sel[None, :]).reshape(-1)
                    if q is not None:
                        X8, sx = q
                        Xs = X8.view(u8).view(n * Ls_, -1).index_select(0, idx).view(n, -1).view(torch.float8_e4m3fn)
                        ms = m8.view(u8).view(16, Ls_, -1).index_select(1, sel).view(16, -1).view(torch.float8_e4m3fn)
                        mw = torch._scaled_mm(ms, W, scale_a=sm, scale_b=sw, out_dtype=torch.float32)[:1]
                        y = _mm8_bias(Xs, W, sx, sw, (mw[0] + b.float()).to(torch.bfloat16))
                    else:
                        Xs = X.view(n * Ls_, -1).index_select(0, idx).view(n, -1)
                        y = torch.addmm(b, Xs, W)
                    if out is None:
                        return y
                    out[:, cs] = y
                return out
            return f

        from .triton_kernels import restore_rope_
        sp = getattr(self, "sp", None)
        T = len(ids)
        # token chunks bound the mapper's memory at 256K-1M (input + outputs scale with the chunk, not the prompt);
        # every op is per token except the fp8 centering, and centering on ANY vector is exact, so the first chunk's
        # mean serves all chunks
        TC = int(getattr(self, "tok_chunk", 0) or 0) or T
        TC = max(self.block, (TC // self.block) * self.block) if TC < T else T
        mu_k = mu_v = None
        for t0 in range(0, T, TC):
            t1 = min(T, t0 + TC)
            n = t1 - t0
            fused = self.fp8 and getattr(self, "fused_build", True) and (device.type == "cuda" or getattr(self, "_fused_cpu", False))
            if fused:
                # one Triton pass pair (row amax, then e4m3) straight from the source cache: no bf16 mapper input,
                # no fp32 temporaries (the torch build + quant8 was ~2/3 of the map stage after the group mapper)
                from .triton_kernels import build_x8
                if mu_k is None:  # centering vector from the first <= 4096 rows (any vector is exact)
                    Xk0, Xv0 = build(t0, min(t1, t0 + 4096))
                    same = Xv0 is Xk0
                    mu_k = Xk0.mean(0, keepdim=True, dtype=torch.float32)
                    mu_v = mu_k if same else Xv0.mean(0, keepdim=True, dtype=torch.float32)
                    mk8, smk = mean8(mu_k)
                    mv8, smv = mean8(mu_v)
                    del Xk0, Xv0
                segk, segv = segments(t0, t1)
                cs_, ss_ = self.cs[t0:t1], self.ss[t0:t1]
                qk = build_x8(segk, cs_, ss_, mu_k.view(-1), n, f32=getattr(self, "_fused_cpu", False))
                qv = qk if same else build_x8(segv, cs_, ss_, mu_v.view(-1), n, f32=getattr(self, "_fused_cpu", False))
                del segk, segv
            else:
                Xk, Xv = build(t0, t1)
            if self.fp8 and not fused:
                same = Xv is Xk
                if mu_k is None:
                    mu_k = Xk.mean(0, keepdim=True, dtype=torch.float32)
                    mu_v = mu_k if same else Xv.mean(0, keepdim=True, dtype=torch.float32)
                    mk8, smk = mean8(mu_k)
                    mv8, smv = mean8(mu_v)
                # only the e4m3 copies stay alive during the GEMMs; K's bf16 input is freed before V is quantized
                qk = quant8(Xk, mu_k)
                if same:
                    qv = qk
                    del Xk, Xv
                else:
                    del Xk
                    qv = quant8(Xv, mu_v)
                    del Xv
            if self.fp8:
                if sp is not None:
                    fk = sparse_f(sp["k"], n, q=qk, m8=mk8, sm=smk)
                    fv = sparse_f(sp["v"], n, q=qv, m8=mv8, sm=smv)
                else:
                    fk = mm8(qk[0], qk[1], mk8, smk, self.Wk8, self.swk, self.bk)
                    fv = mm8(qv[0], qv[1], mv8, smv, self.Wv8, self.swv, self.bv)
            elif sp is not None:
                fk = sparse_f(sp["k"], n, X=Xk)
                fv = sparse_f(sp["v"], n, X=Xv)
                del Xk, Xv
            else:
                fk = lambda c0, c1, X=Xk: torch.addmm(self.bk[c0:c1], X, self.Wk[:, c0:c1])
                fv = lambda c0, c1, X=Xv: torch.addmm(self.bv[c0:c1], X, self.Wv[:, c0:c1])
                del Xk, Xv
            ct_, st_ = self.ct[t0:t1].contiguous(), self.st[t0:t1].contiguous()
            # target layers in chunks: peak mapped memory is 2*n*chunk*H*D bf16 instead of 2*n*Lt*H*D
            for l0 in range(0, Lt, self.chunk):
                l1 = min(Lt, l0 + self.chunk)
                Yk4 = fk(l0 * HD, l1 * HD).view(n, l1 - l0, H, D)
                if not Yk4.is_contiguous():
                    Yk4 = Yk4.contiguous()
                Yv4 = fv(l0 * HD, l1 * HD).view(n, l1 - l0, H, D)
                # fused k_norm RMS restore + RoPE over the chunk's layers in one Triton pass (in place)
                restore_rope_(Yk4, None if self.kw is None else self.kw[l0:l1].contiguous(), ct_, st_)
                if self.sinkK is not None and t0 == 0:  # sink positions: fixed native template, not the mapped KV
                    A = min(self.sinkK.shape[0], n)
                    Yk4[:A] = self.sinkK[:A, l0:l1]
                    Yv4[:A] = self.sinkV[:A, l0:l1]
                for l in range(l0, l1):
                    yield t0, l, Yk4[:, l - l0], Yv4[:, l - l0]
                del Yk4, Yv4
            del fk, fv
        del lay

    def _map_layers(self, ids):
        """(layer, K [T,H,D] RoPE'd, V [T,H,D]) per target layer, token chunks joined (tests / reference use)."""
        parts = {}
        for t0, l, K, V in self._map_chunks(ids):
            parts.setdefault(l, []).append((K.clone(), V.clone()))
        for l in sorted(parts):
            yield l, torch.cat([k for k, _ in parts[l]]), torch.cat([v for _, v in parts[l]])

    def _xalign(self, ids):
        """Target token ids -> (source ids, LongTensor[len(ids)] aligned source index), research/xalign contract:
        the source token whose character span holds the last character of each target token. Target offsets come from
        per-token decoding of the ids actually scheduled (no re-tokenization of the target side)."""
        from research.xalign import align
        tt, ts_ = self.xtok
        pieces = [tt.decode([i]) for i in ids]
        off, c = [], 0
        for p_ in pieces:
            off.append((c, c + len(p_)))
            c += len(p_)
        text = "".join(pieces)
        es = ts_(text, add_special_tokens=False, return_offsets_mapping=True)
        s_ids, s_off = list(es["input_ids"]), es["offset_mapping"]
        idx = align(s_off, off)
        if self.xbos_s is not None:
            s_ids = [self.xbos_s] + s_ids
            idx = idx + 1
        return s_ids, idx

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
                self._backend = str(bname)
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
        asyn = bool(getattr(self, "async_load", False)) and device.type == "cuda"
        self._sync_on = _SYNC and not asyn
        if asyn:
            # source + map + KV writes on a side stream; each target layer's attention waits only for its own layer
            # (wait_for_layer_load), so the target's tail prefill overlaps the remaining map groups and the CPU-side
            # launch latency of the connector is hidden behind GPU work
            if getattr(self, "_side", None) is None:
                self._side = torch.cuda.Stream(device=device)
            self._side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(self._side):
                self._load_requests(meta, layers, device, events=True)
        else:
            self._load_requests(meta, layers, device, events=False)

    def _load_requests(self, meta, layers, device, events=False):
        import time as _t
        for r in meta.requests:
            src_dev = self.cs.device
            self._sync_if(src_dev); t0 = _t.perf_counter()
            s0 = r.start
            slots_all = r.slot_mapping.to(device)
            D = self.D
            t_map = 0.0; t_write = 0.0; t_first = None
            gen = self._map_chunks(r.token_ids)
            cur = None
            while True:
                ta = _t.perf_counter()
                try:
                    c0, l, K, V = next(gen)
                except StopIteration:
                    break
                if cur != c0:  # slots of this token chunk past the local prefix hit s0
                    cur = c0
                    a0 = max(c0, s0)
                    slots = slots_all[a0:c0 + K.shape[0]]
                    bi, off = slots // self.block, slots % self.block
                    skip = a0 - c0
                self._sync_if(src_dev)
                tb = _t.perf_counter()
                if t_first is None:
                    t_first = tb - t0  # includes source prefill + first layer map
                else:
                    t_map += tb - ta
                kv = layers.get(l)
                if kv is None:
                    raise RuntimeError(f"kvx connector: target layer {l} has no KV cache (maps fit on a different depth?)")
                if skip:
                    K, V = K[skip:], V[skip:]
                if K.shape[0] == 0:
                    continue
                K, V = K.to(kv.device, non_blocking=True), V.to(kv.device, non_blocking=True)
                if self._layout is None:
                    self._layout = kv_layout(kv, K.shape[1], D, self._backend)
                    logger.info("kvx connector: KV layout %s (backend %s)", self._layout, self._backend)
                write_kv(kv, bi, off, K, V, self.block, self._layout)
                if events:
                    ev = torch.cuda.Event()
                    ev.record()
                    self._ev[l] = ev
                self._sync_if(device)
                t_write += _t.perf_counter() - tb
            self._sync_if(device); self._sync_if(src_dev)
            t_src = getattr(self, "_t_src_done", t0) - t0
            # one parseable line per request: source prefill / KV map / cache move (ms); first target token = TTFT - total
            logger.info("kvx components: n=%d src_ms=%.1f map_ms=%.1f move_ms=%.1f total_ms=%.1f tp=%d src_tp=%d",
                        len(r.token_ids), 1e3 * t_src, 1e3 * ((t_first or 0.0) - t_src + t_map), 1e3 * t_write,
                        1e3 * (_t.perf_counter() - t0), self.tp_size, int(getattr(self, "src_tp_on", False)))
            logger.info("kvx connector: n=%d src+first_layer=%.3fs map_rest=%.3fs transfer+write=%.3fs total=%.3fs",
                        len(r.token_ids), t_first or 0.0, t_map, t_write, _t.perf_counter() - t0)
            self.stats["requests"] += 1
            self.stats["mapped_tokens"] += len(r.token_ids)

    def _sync_if(self, dev):
        if getattr(self, "_sync_on", True):
            _sync(dev)

    def wait_for_layer_load(self, layer_name: str) -> None:
        ev = None
        m = _LAYER_RE.search(layer_name or "")
        if m is not None and getattr(self, "_ev", None):
            ev = self._ev.pop(int(m.group(1)), None)
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        return

    def wait_for_save(self):
        return
