# [KV Connector] CrossModelConnector: small-model prefill with mapped KV for faster TTFT

## What
A v1 KV connector that serves long prompts with a fraction of the target's prefill FLOPs. A small model of the same
family (same tokenizer) prefills the prompt; closed-form ridge maps turn its per-layer KV into the served model's KV
(RoPE removed before mapping and re-applied with the target's tables, YaRN included); the mapped KV is written into
the request's paged blocks and reported as an external prefix hit (`get_num_new_matched_tokens`). The target computes
only a short native tail and decodes as usual. No API change: enabled with `--kv-transfer-config`.

```bash
vllm serve Qwen/Qwen2.5-72B --tensor-parallel-size 2 --attention-backend FLASH_ATTN --kv-transfer-config '{
  "kv_connector": "CrossModelConnector", "kv_role": "kv_both",
  "kv_connector_extra_config": {"src": "Qwen/Qwen2.5-3B", "maps": "maps_k16.pt", "tail": 256, "fp8": true,
                                "restore": "none", "sink_template": "sink_template.pt", "tp_source": true}}'
```

## Results (one request at a time, native vLLM on the same GPUs, FLASH_ATTN both sides)
| pair | prompt | native vLLM | with connector | speedup | quality vs native |
|---|---|---|---|---|---|
| Qwen2.5-3B -> 72B, 1x B200 | 32K | 4.61 s | 0.570 s | 8.1x | first-token KL 0.076, top-1 81% |
| Qwen2.5-3B -> 72B, 2x B200 TP2, YaRN x4 | 128K | 17.1 s | 3.38 s | 5.05x | KL 0.056 |
| Qwen3-1.7B -> 32B, 1x B200, YaRN x4 | 128K | 21.0 s | 3.46 s | 6.1x | KL 0.047 |

## Design
- Scheduler side: the mapped prefix is whole blocks of `len - tail`; prefix-cache hits are respected (only blocks past
  the local hit are written; `NewRequestData.num_computed_tokens` already counts the connector's own tokens).
- Worker side: one dense GEMM per K/V (fp8 e4m3 rowwise via `torch._scaled_mm`) in target-layer chunks, a fused Triton
  k_norm-restore + RoPE epilogue (int64 offsets), whole-block `index_copy_` into the token-major FLASH_ATTN layout.
  FlashInfer reads blocks as `[2, H, N, D]` and is refused explicitly.
- TP: each rank maps only its KV heads; `tp_source` shards the source prefill over the TP group and all-gathers its KV.
- Source prefill: cuDNN SDPA first (1.5x over torch's default pick at 128K).
- Maps: produced offline (closed-form ridge on calibration text; ~minutes on one GPU). Format: list of per-target-layer
  dicts {sel, W, b, H, D}.

## Tests
CPU (no GPU): RoPE/YaRN exactness, paged write vs a reference loop (incl. prefix hits), scheduler call order, chunked
mapping == unchunked, TP=2 head slicing == TP=1, sharded source prefill == unsharded (2 gloo ranks), Triton epilogue
(interpreter), Mistral3 / YaRN round trip.

## Limitations
FLASH_ATTN layout only; attention-only targets (hybrid SSM/linear-attention state is not mapped); same-family pairs.
