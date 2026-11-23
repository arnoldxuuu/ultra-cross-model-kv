# OneTriangle clone: Qwen3 / Qwen2.5 / Gemma / Llama pairs (work in progress)

> **Status: TODO.** This is the earlier open clone of OneTriangle's cross-model KV transfer (code: the private
> `kvx` repo; not part of `xkv`). Its results are collected here so they are in one place. We are still working on
> beating all of OneTriangle's published benchmarks: the 128K speedups, exact recall at long context, and the
> Qwen2.5-3B -> 72B and Ministral -> Mistral Medium pairs are open. The main results of this repository are the
> Qwen3.5-4B -> Qwen3.8-27B ones in the [README](../README.md).

Commands in the Reproduce section refer to the `kvx` code base, not to this repository.

This repo rebuilds, from public material only, the method that OneTriangle (YC S26) sells as "the fastest, cheapest inference powered by KV cache transfer." A small model prefills the prompt. Its KV cache, with RoPE removed, is mapped into a large model's KV space and the target's RoPE is applied again. The large model then decodes from the mapped cache.

## Status against the target's claims (2026-11-19)

| OneTriangle claim | Our clone, measured |
|---|---|
| ~8.4x TTFT for Qwen3 1.7B -> 32B at 32K (7.34 s -> 0.87 s, H100, HF native baseline) | HF+FA2 on B200: 5.12 s -> 0.83 s (6.2x). Inside vLLM 0.34 on H200 against native vLLM on the same GPU: 5.16 s -> 1.02 s (**5.1x**); 3.6x at 4K |
| 9.6x for Qwen2.5 3B -> 72B at 127K | Against native vLLM, one request at a time. One B200, 32K: 4.61 s -> 0.57 s (**8.1x**). At 128K with YaRN, tensor parallel 2 on 2x B200: 17.1 s -> 3.38 s (**5.05x**). Every rank runs the full 3B prefill, and that prefill is the remaining bottleneck. Quality at 128K: first-token KL 0.056 |
| Long context (claims at 127K) | Qwen3 1.7B -> 32B at 128K with YaRN on one B200: 21.0 s -> 3.46 s (**6.1x**). Quality on real 128K book prompts: first-token KL 0.047. The maps were fit on 1K-token sequences |
| "First token computed natively" | It is required on the 24x pair. Plain ridge on Qwen2.5 3B -> 72B gives KL 4.09, worse than no context, because the 72B parks its attention sink on token 0. A fixed calibration-mean K/V template at token 0 gives 0.390 with no target compute; with a 256-token tail, 0.088 |
| 12.6x for Ministral 3B -> Mistral Medium 128B | Not testable: Mistral Medium weights are not public |
| 20% lower prefill cost, 40% lower TTFT | $ per 1M prefill tokens, bs1 on H200: $0.199 -> $0.039 at 32K (5.1x) |
| Lower KL than NVIDIA's linear mapper (no number published) | Qwen3 14B -> 32B: paper ridge k8 KL 0.112 -> ours 0.045 (-60%). Qwen3 1.7B -> 32B: 0.339 -> 0.139 with a 64-token native tail (-59%) |
| Quality retention | 1.7B -> 32B with a 64-token native tail: HellaSwag 99%, MMLU 5-shot 99%. 14B -> 32B: 100%. Gemma-2, Mistral, Llama, Qwen2.5 and an MoE target: 97-103% / 88-99% |
| "Restore relative importances" | Our reading is a family-specific key restore: RMS to the target's `k_norm` for Qwen3, a global gain of 1.1 for Qwen2.5 and Gemma-2, none for Llama. Plus native sink and tail tokens |
| No API change, vLLM integration | `kvx/vllm_connector.py`: a vLLM v1 KV connector. The prompt prefix is reported as an external cache hit, and mapped K/V are written into paged blocks |

## What the target claims

| Claim | Number | Source |
|---|---|---|
| Prefill cost / TTFT | 20% lower / 40% lower | YC company page |
| TTFT speedup | ~10x | onetriangle.ai |
| Ministral 3B -> Mistral Medium 128B at 32K | 12.6x (18.22 s -> 1.45 s) | onetriangle.ai |
| Qwen2.5 3B -> 72B at 127K | 9.6x | onetriangle.ai |
| Qwen3 1.7B -> 32B at 32K | ~8.4x | onetriangle.ai |
| Quality | "lower KL than NVIDIA's KV-transfer paper" (no number published) | YC page |
| Method | ridge (lam 0.01) on K and V, then a rank-16 residual distilled with KL; first token and last k tokens computed natively | onetriangle.ai/blog/cache-transfer |

Closest paper: Heo et al. (NVIDIA), arXiv 2608.03893. It uses a per-head ridge from the top-k source layers. For Qwen3 14B -> 32B at k=8 it keeps 97.6% of benchmark accuracy. The paper reports no KL.

## What is built

- `kvx/core.py` contains the mapper pipeline:
  - **RoPE strip:** keys are read from the cache and un-rotated with each model's own cos/sin, so it works for any RoPE model.
  - **Gram statistics:** one pass accumulates statistics over all source and target layers.
  - **Layer choice:** an R² table picks the top-k source layers for each target layer.
  - **Fit:** closed-form ridge.
  - **Importance restore:** rescaling of the mapped keys.
  - **Eval:** next-token KL between the mapped and native caches.
- `modal_fit_eval.py` fits and evaluates one model pair in a single Modal job.


## Across families (KL to native, 960-token mapped prefix, mean over 64 continuation tokens)

| Pair | Head layout | Small model alone | Best mapped | + 64 native tail | Accuracy retention with tail (HellaSwag / MMLU5) |
|---|---|---|---|---|---|
| Qwen3 1.7B -> 32B | 8x128 -> 8x128 | 0.752 | 0.219 (k24, rms, 4000 seqs) | 0.144 | 99% / 98% (mixed calibration: 101% / 99%) |
| Qwen3 14B -> 32B | 8x128 -> 8x128 | 0.258 | 0.079 (k16, rms, 2000 seqs) | 0.052 (0.045 with first 4 native) | 100% / 103% |
| Qwen3 1.7B -> 8B | 8x128 -> 8x128 | 0.455 | 0.133 (kvcat k16, 4000 seqs) | 0.087 | 101% / 96% (500 seqs) |
| Llama-3.2 1B -> 3.1 8B | 8x64 -> 8x128 | 0.322 | 0.147 (kvcat k16, 2000 seqs) | 0.099 | 99% / 91% |
| Llama-3.1 8B -> 70B | 8x128 -> 8x128 | 0.304 | 0.178 (k20) | 0.124 | 99% / MMLU OOM on 2xH200 |
| Qwen2.5 3B -> 7B | 2x128 -> 4x128 | 0.207 | 0.142 (kvcat k16, gain 1.1) | 0.125 | 102% / 90% (500 seqs) |
| Qwen2.5 1.5B -> 7B | 2x128 -> 4x128 | 0.276 | 0.239 (k16) | 0.177 | 103% / 88% |
| Ministral-8B -> Mistral-Small-24B | 8x128 -> 8x128 | 0.252 | 0.117 (k16, gain 1.1, 1000 seqs) | 0.081 | 100% / 98% |
| Phi-3-mini -> Phi-3-medium | 32x96 (MHA) -> 10x128 | 0.206 | 0.121 (k8, gain 1.1, λ_rel 1e-3) | 0.095 | - |
| OLMo-2 1B -> 7B | 16x128 (MHA) -> 32x128 (MHA) | 0.297 | 0.229 (k8, gain 1.1) | 0.155 | 101% / 85% |
| Qwen2.5 3B -> 72B (OneTriangle's 9.6x pair) | 2x128 -> 8x128 | 0.457 | 0.390 (k16 + calibration-mean sink template at token 0; plain ridge 4.09) | 0.086 (4 native sink tokens + 256 tail) | - |
| Gemma-3 1B -> 4B (multimodal checkpoint, text model) | 1x256 -> 4x256 | 0.308 | 0.206 (k16, gain 1.1) | 0.160 | 99% / 96% |
| Gemma-3 4B -> 12B | 4x256 -> 8x256 | 0.190 | 0.271 with ridge (values fail); **0.145 with an MLP value mapper** (-47%; KV-concat made it worse: 1.46) | 0.212 -> **0.103** (MLP) | 93% / 92% (ridge) |
| Gemma-3 270M -> 1B (local/global RoPE, sliding window) | 1x256 -> 1x256 | 0.423 | 0.222 (k16, gain 1.1) | 0.149 | 101% / n.a. (MMLU near chance) |
| Gemma-2 2B -> 9B | 4x256 -> 8x256 | 0.187 | 0.078 (k16, gain 1.1) | 0.054 | 100% / 97% |
| Qwen3 1.7B -> 30B-A3B (MoE target) | 8x128 -> 4x128 | 0.538 | 0.200 (k24, rms, 2000 seqs) | 0.137 | 98% / 99% |
| DeepSeek-R1-Distill-Qwen 1.5B -> 7B | 2x128 -> 4x128 | 1.057 | 0.553 (k16, gain 1.1) | 0.400 | not run |

The MoE target maps as well as the dense ones. Its speed case is weak, though: the 30B-A3B uses about 3.3B active parameters per token, so prefill costs about 6.6 GF/token plus attention. The 1.7B source plus the mapper costs about 5.4 GF/token. The TTFT ceiling is therefore about 1x at short context and about 2x at 32K, where attention dominates.

### Source-size frontier into Qwen3-32B (KL, 2000-4000 calibration sequences)

| Source | 32K TTFT ceiling | small alone | mapped (k16/24 + rms) | + tail 64 | + 4 sink + tail 256 (or 64) |
|---|---|---|---|---|---|
| 0.6B | 8.0x | 1.051 | 0.946 | 0.729 | 0.216 |
| 1.7B | 6.8x | 0.752 | 0.219 | 0.144 | 0.062 |
| 4B | 3.7x | 0.540 | 0.179 | 0.126 | 0.055 |
| 8B | 2.8x | 0.368 | 0.150 | 0.105 | 0.044 |
| 14B | 1.9x | 0.257 | 0.079 | 0.052 | 0.045 |

The 32K ceiling uses the compute roofline: (source prefill + dense mapper) against native 32B prefill. 1.7B is the sweet spot: it has a 6.8x ceiling and keeps 98-99% of benchmark accuracy with a 64-token native tail. 4B and 8B lose most of the speed for a modest gain in quality. 0.6B maps poorly unless a large native tail and native sink tokens are added.

Long context (Qwen3-1.7B -> 32B, 8K-token PG19 books, 4000-seq k24 maps): the 1.7B alone gives KL 1.036. Mixed web+book calibration gives 0.278, 0.210 with 64 native tail tokens, and 0.152 with 4 sink + 256 tail tokens; FineWeb-only calibration gave 0.479 mapped. The full stack (mix, λ_rel 1e-3, KV-concat inputs, k16) gives 0.274 / 0.206 / 0.148.

At **32K tokens** (8 PG19 test books, B200) the full stack does not degrade:

| Setting | KL |
|---|---|
| 1.7B alone | 1.018 |
| Mapped | 0.249 |
| + 64 native tail tokens | 0.188 |
| + 1024 native tail tokens (3% of the prompt) | 0.079 |
| + 4 sink and 1024 tail tokens | 0.076 |

At 8K, adding 64 native tail tokens gives 0.330, and 4 native sink tokens plus 256 native tail tokens give 0.210. The same settings at 1K give 0.219 / 0.144 / 0.062. Mapping quality halves from 1K to 8K, the main open gap. OneTriangle names exact recall at long context as its hardest case too.

### Calibration domain is the long-context lever (Qwen3-1.7B -> 8B, k24 + RMS)

| Calibration set | KL on 1K FineWeb-Edu | KL on 8K PG19 books |
|---|---|---|
| FineWeb-Edu, 4000 x 1K | **0.144** | 0.371 |
| PG19 train, 1000 x 4K | 0.307 (2000 x 1K) | 0.189 |
| Mix: 2000 FineWeb + 2000 PG19 (x 1K) | 0.158 | **0.186** |

Most of the apparent "long-context gap" came from domain mismatch: the maps were calibrated on web text and evaluated on books. One mixed calibration set halves the 8K book KL and costs about 10% on web text. The mapper should be calibrated on a mix that resembles serving traffic. OneTriangle's pitch to post-train on traffic points the same way.

### Recommended quality recipe (all levers stacked, Qwen3-1.7B -> 8B)

- 4000 calibration sequences, a mix of web and books;
- relative ridge λ 1e-3;
- K and V of each source layer as mapper input;
- top-k source layers;
- RMS key restore;
- 4 native sink tokens and a native tail.

This gives KL 0.142 (k24) mapped, 0.095 with a 64-token tail, and 0.049 with 4 sink + 256 tail tokens. The paper's ridge k=8 gives 0.244. Benchmark retention with the 64-token tail: HellaSwag 103%, MMLU 5-shot 98%.

What carries across families:

- **The native tail** is the largest single lever.
- **More calibration data** helps: 500 to 4000 sequences gives -10 to -25%.
- **Feeding both K and V of each source layer** helps most where keys are the bottleneck (Qwen2.5).
- **The key "importance restore" is family-specific:**
  - Qwen3 uses RMS restore to the target's `k_norm`.
  - Qwen2.5 and Gemma-2 use a global key gain of 1.1.
  - Llama uses none.

The paper's absolute ridge λ of 0.01 is fragile. On Phi-3, which has no qk-norm and large key magnitudes, the solve is ill-conditioned: R² > 1 and mapped KL 8.6. A relative λ (1e-3 times the mean Gram diagonal) fixes it, and is now the default.

**Attention sinks break the 24x pair, Qwen2.5-3B -> 72B.** Plain ridge gives KL 4.09, worse than no context at all (0.945).
Keys map fine and values fail, and recomputing the last 256 tokens natively does not help (3.72). Diagnosis: the 72B parks
its attention sink on token 0, the ridge mis-maps that token's value, and every later query reads it. The fix is one token.
Computing token 0 natively gives 0.391. A fixed calibration-mean K/V template at token 0, which needs no target compute at
serving time, gives 0.390. With 4 native sink tokens and a 256-token native tail it reaches 0.086. 
On the one failing pair, Gemma-3 4B -> 12B, ridge mapping does worse than running the 4B alone. Adding an MLP residual on the value maps (as the NVIDIA paper does for its failing pairs) halves KL, 0.275 -> 0.145, and beats the 4B. Held-out value R² did not change, 0.677 -> 0.672. R² is a poor proxy for output KL.

Other mapper and recompute variants were tried; none beat the recipe above.

## Reproduce

Commands run from this directory. Modal volume: `kvx`.

1. Fit maps and measure quality with the recommended recipe:

   ```bash
   KVX_GPU=B200 modal run modal_fit_eval.py::main --src Qwen/Qwen3-1.7B --tgt Qwen/Qwen3-32B \
     --ks 16 --modes rms --tails 64,256 --tail-k 16 --sinks 4 --n-calib 4000 --calib-src mix \
     --lam-rel 1e-3 --feat kvcat --bench-n 200 --bench-tail 64 --save-maps 1 --run-id my-run
   ```

   - Pairs with the same KV width also work with `--feat kv`. That halves the mapper cost.
   - Add `--eval-l 8192 --n-eval 16` (or 32768) to evaluate long contexts on PG19.

2. Measure serving TTFT and quality in vLLM 0.34, native first and then through the connector:

   ```bash
   KVX_GPU=H200 modal run modal_vllm.py --mode native --lens 4096,8192,16384,32768
   KVX_GPU=H200 modal run modal_vllm.py --mode mapped --maps /vol/runs/<run>/maps_k24.pt --fp8 1 --tag _mine
   python tools/vllm_quality.py out/vllm/native_Qwen3-32B.json out/vllm/mapped_Qwen3-32B_mine.json
   ```

3. Use the connector in your own vLLM:

   ```python
   KVTransferConfig(
       kv_connector="CrossModelConnector",
       kv_connector_module_path="kvx.vllm_connector",
       kv_role="kv_both",
       kv_connector_extra_config={
           "src": "Qwen/Qwen3-1.7B",
           "maps": "<maps.pt>",
           "tail": 64,
           "min_len": 1024,
           "restore": "rms",
           "fp8": True,
       },
   )
   ```

   Set `VLLM_ATTENTION_BACKEND=FLASH_ATTN`. The connector writes the packed `[blocks, H, block, 2D]` layout.

## Measured results

### OneTriangle's suites, Qwen3-1.7B -> Qwen3-8B (EC2 g7e, 1x RTX PRO 6000 96 GB, spot $2.11/h, run ~$4.93)

Maps: k24, 4000 mixed web+book calibration sequences, RMS key restore, fp8 dense mapper in vLLM 0.34 (FLASH_ATTN).
Harness: `clone/kvx/oteval.py` (xkv `genevals` loaders and scorers), OneTriangle's n. Settings matched to theirs:
long-document suites keep a 256-token native tail plus a 4-token sink template (OneTriangle computes the first token
and the last k tokens natively); ARC-C, HellaSwag and PIQA are scored by log-likelihood (lm-eval style), which is what
their natives match (their 8B ARC-C .555, ours .56).

| suite (their n) | mapped | native | OneTriangle generic 1.7B->8B (their native) | verdict |
|---|---:|---:|---:|---|
| QuALITY (500) | **.674** | .756 | .595 / .63, ladder .635 (.77 / .838) | ahead |
| BAMBOO PaperQA (84) | .798 | .845 | .77 / .845 (.87 / .893) | ahead of .77, behind .845 |
| ContractNLI (500) | .586 | .69 | .635 / .672 (.635 / .758) | behind |
| LongBench v2 (97, 32K window) | .278 | .371 | .33 / .299 (.445 / .454) | behind |
| ARC-C, log-likelihood (500) | .466 | .56 | .48, ladder .455 (.555 / .54) | ahead of ladder, behind generic |
| HellaSwag, log-likelihood (500) | .584 | .65 | .685 (.745) | behind |
| PIQA, log-likelihood (500) | .764 | .776 | .775 (.79) | behind, within noise |
| ZebraLogic, thinking off (200) | .215 | .25 | .185 (.79) | ahead (their native suggests reasoning on) |
| ZebraLogic, thinking on, 12K tokens (200) | .50 | .855 | ladder .555 (.81) | behind |
| MATH-500, thinking off / on (500) | .778 / .778 | .838 / .798 | ladder .82 / .925 (.85 / .94) | behind |

TTFT, one request at a time, same GPU and boot:

| prompt | HF native -> mapped | vLLM 0.34 native -> mapped | OneTriangle (HF) |
|---|---|---|---|
| 8K | 0.490 -> 0.270 s (1.81x) | 0.387 -> 0.234 s (1.66x) | 1.92x |
| 16K | 1.075 -> 0.606 s (1.77x) | 0.901 -> 0.570 s (1.58x) | 2.29x |
| 32K | 2.615 -> 1.401 s (1.87x) | 2.313 -> 1.341 s (1.72x) | 2.55x |

Reading: plain ridge maps beat OneTriangle's KL-distilled generic adapter on QuALITY and match it on BAMBOO, but lose
on ContractNLI, LongBench v2 and HellaSwag; on this pair the 1.7B prefill is a third of the 8B's native prefill at 32K,
which caps the speedup. Next levers: KL map distillation (xkv's works) and a sparse k16 mapper.

### OneTriangle's ladder, Qwen3-1.7B -> Qwen3-32B column (EC2 g7e, 1x RTX PRO 6000 96 GB, spot $2.28/h, ~$13)

Maps: k24 from 2000 mixed calibration sequences (two passes and maps offloaded to host between solves to fit the 32B
on one 96 GB GPU). vLLM 0.34 at gpu_memory_utilization 0.80: 0.88 left the connector too little, 0.75-0.78 too little
KV cache for one 33K-token request.

| suite (OneTriangle's n) | mapped | native | OneTriangle ladder 1.7B->32B (their native) | verdict |
|---|---:|---:|---:|---|
| QuALITY (500), 256-token native tail + sink template | **.674** | .83 | .645 (.81) | ahead |
| ARC-C, log-likelihood (500) | **.488** | .606 | .455 (.565) | ahead |
| MATH-500, thinking off (500) | .766 | | .83 (.865) | behind |
| LongBench v2 (97) | | .423 | .33 (.425) | mapped run OOMed here; the kvx-quality seat measured .392 on H200 (ahead) |
| MATH-500 / ZebraLogic, thinking on | | | .935 / .58 | not run |

### OneTriangle's ladder, Qwen3-1.7B -> Qwen3-14B and Qwen3-4B columns (EC2 g7e spot, $2.31/h, 1.15 h, $2.65)

Maps: k24 from 4000 mixed calibration sequences, fit on the box (about 10 min each). Same settings as the 32B column;
vLLM 0.34 at gpu_memory_utilization 0.85; reasoning arms not run.

| suite (OneTriangle's n) | 14B mapped | 14B native | ladder 1.7B->14B | 4B mapped | 4B native | ladder 1.7B->4B | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| QuALITY (500), 256-token native tail + sink template | **.686** | .788 | .635 | **.650** | .712 | .61 | ahead on both |
| MATH-500, thinking off (500) | .826 | | .83 | .762 | | .80 | behind (14B within noise) |
| ARC-C, log-likelihood (500) | .468 | .592 | .48 | .416 | .536 | .43 | behind |
| LongBench v2 (97) | | .402 | .415 | | .361 | .34 | mapped arm OOMed |

The 1.7B alone scores .438 on ARC-C, so the 4B mapping loses to its own source there. The LongBench v2 mapped arms
ran out of memory at 0.80/0.85 utilization even with 8-layer mapper chunks: with a small target, vLLM fills the share
with KV cache (65 GB for the 4B) and leaves the connector about 19 GB, while it needs about 25 GB at 32K tokens. The
8B ran at 0.70; the 32B fits at 0.80 because its weights take most of the share. Rerun them at about 0.60.

### DevEval (code), OneTriangle's ladder rows (EC2 g7e spot, 1x RTX PRO 6000, $2.31/h)

Harness: `clone/kvx/deveval.py` (generation) + `clone/kvx/deveval_eval.py` (pass@1 with seketeam/DevEval's semantics:
splice the body at `body_position`, run the item's tests with `setup.py pytest`, restore the file) +
`clone/box/deveval_setup.sh` (the authors' Python 3.9 environment on a throwaway CPU box; the tests run the repos'
code). Prompts: DevEval's local-infilling instruction prompts (file above and below + the signature and docstring to
complete; mean 4.5K tokens), chat template, greedy, 1024 new tokens, reasoning off. Items: 1497 of 1825 reference
solutions pass in that environment (pip must be in the venv: `setup.py pytest` fetches each repo's test
requirements); 500 sampled (seed 0) from those with prompts under ~28K tokens cover 78 repos (OneTriangle: 500 from 76).

| target | native | mapped, web+book maps, tail 256 | mapped, web+book+Python maps, tail 256 / 1024 | OneTriangle ladder mapped (native) |
|---|---:|---:|---:|---:|
| Qwen3-4B | .372 | .210 | .292 / **.324** | .30 (.355) |
| Qwen3-8B | .428 | | .354 / **.364** | .31 (.375) |
| Qwen3-14B | .490 | .254 | .344 / **.410** | .34 (.415) |
| Qwen3-32B | .518 | .258 | | .36 (.455) |

Maps fit only on web text and books kept half the target's pass@1 on code. `calib_src=mixcode` (a third each of
FineWeb-Edu, PG19 and Python files from codeparrot/codeparrot-clean, DevEval's 115 repos excluded by name) brings
retention to 84-87% at a 1024-token native tail (the code right above the function sits mid-prompt, before the file
below it). QuALITY with the same maps: 4B .626, 8B .684, 14B .684 (web+book maps: .650, .674, .686). Runs g7e-tp
($5.46) and g7e-mixcode ($3.04); the CPU scoring box is a c7i.4xlarge at $0.71/h.

### Calibration mix, native first tokens and reasoning budget (EC2 g7e spot, 1x RTX PRO 6000, $2.31/h)

`calib_src=mixall`: quarters of FineWeb-Edu, PG19, Python (codeparrot, DevEval repos excluded) and OpenWebMath, k24,
4000 sequences (32B: 2000, maps offloaded during the solve). Same eval settings as the ladder columns above.

| row (OneTriangle ladder mapped) | 4B | 8B | 14B | 32B |
|---|---:|---:|---:|---:|
| MATH-500, reasoning off: web+book maps | .762 | .778 | .826 | .766 |
| MATH-500, reasoning off: mixall maps (OT .80 / .82 / .83 / .83) | **.802** | .808 | **.844** | .822 |
| QuALITY: mixall maps (OT .61 / .635 / .635 / .645) | **.634** | **.678** | **.674** | **.678** |
| DevEval, reasoning off, tail 1024: mixall maps (OT .30 / .31 / .34 / .36) | | | **.400** | **.450** |
| DevEval, reasoning on, 12288 new tokens: mixall maps (OT .32 / .315 / .39 / .40) | .284 | **.342** | | |
| DevEval, reasoning on, 6144 new tokens: mixcode maps | .266 | .294 | **.416** | |

Reasoning on, many replies run out of tokens before closing their reasoning (4B: 229 of 500 at 6144, 201 at
12288; 8B: 186 -> 132); those count as failures. 32B MATH .822 and 8B .808 are inside the 95% interval of OneTriangle's
numbers. On the 1.7B->8B section the mixall maps give BAMBOO .810 (was .798), ContractNLI .586 (same), LongBench
v2 .258 (was .278).

lm-eval-style log-likelihood with the first k context tokens taken from the target itself (`loglik_impl(first=k)`,
exact because attention is causal; OneTriangle computes the first token natively too). ARC-C contexts are short
(31.7 tokens on average), so k changes the mapped share a lot:

| ARC-C loglik (OT ladder 4B .43, 14B .48) | k = 0 | k = 4 (79% mapped) | k = 16 (35% mapped) |
|---|---:|---:|---:|
| 4B, web+book maps | .416 | .432 | .472 |
| 14B, web+book maps | .468 | .458 | .516 |
| 4B / 14B / 8B, mixall maps | | .430 / .460 / .440 | |

8B HellaSwag / PIQA (OT generic .685 / .775): .564 / .768 at k = 1, .598 / .768 at k = 4 (native .650 / .776).
Runs g7e-mixcode ($3.04), g7e-devthink ($8.06), g7e-mixall ($8.75), g7e-mixall32 ($4.98); DevEval scoring on a
c7i.4xlarge ($9.60 for 13.45 h).

### Prefill throughput and $/1M prompt tokens at concurrency > 1 (EC2 g7e spot, 1x RTX PRO 6000, $2.31/h)

`clone/kvx/throughput.py`: closed batches of c requests of random tokens (no prefix reuse), one output token, native
vLLM 0.34 vs the connector on the same box and boot; median of 3 batches after a warm-up. Maps k24 (fp8 mapper),
64-token native tail + sink template. $/1M = box $/h / (prompt tokens/s x 3600) x 1e6.

| pair | prompt | c | native tok/s | mapped tok/s | throughput | $/1M prompt tokens native -> mapped |
|---|---|---:|---:|---:|---:|---|
| Qwen3-1.7B -> 14B | 8K | 1 / 4 / 16 | 12,146 / 12,155 / 12,042 | 33,020 / 35,904 / 37,164 | 2.7x / 3.0x / **3.1x** | $0.053 -> $0.017 (c=16) |
| Qwen3-1.7B -> 14B | 16K | 1 / 4 / 8 | 10,566 / 10,588 / 10,556 | 30,218 / 31,329 / 31,657 | 2.9x / 3.0x / 3.0x | $0.061 -> $0.020 (c=8) |
| Qwen3-1.7B -> 14B | 32K | 1 | 8,560 | 26,028 | 3.0x | $0.075 -> $0.025 |
| Qwen3-1.7B -> 32B | 8K | 1 / 2 / 4 | 5,106 / 5,096 / 5,066 | 25,867 / 26,490 / 28,955 | 5.1x / 5.2x / **5.7x** | $0.127 -> $0.022 (c=4) |

Native prefill is compute-bound at c=1 already (tokens/s flat in c); the mapped path gains a little with c as the
source prefill and mapper GEMMs fill the GPU. The 32B leaves ~8 GB of KV cache next to its weights on one 96 GB GPU, so
c stops at 4 for 8K prompts. Run id g7e-tp ($5.46 for the whole box, which also ran LongBench and DevEval arms).

### Qwen2.5-3B -> Qwen2.5-72B in vLLM (OneTriangle's 9.6x pair; Modal B200, FLASH_ATTN on both sides, one request at a time)

| Prompt | Native vLLM | Mapped (fp8 k16 maps, sink template, 256-token native tail) | Speedup |
|---|---|---|---|
| 4K | 0.461 s | 0.099 s | 4.7x |
| 8K | 0.941 s | 0.156 s | 6.0x |
| 16K | 2.02 s | 0.278 s | 7.3x |
| 32K | 4.61 s | 0.570 s | 8.1x |

Quality against native vLLM on 16 FineWeb-Edu prefixes: approximate first-token KL 0.076, top-1 agreement 81%,
and greedy decoding matches for the first 7.4 tokens on average. The HF KL on 64 docs is 0.088. OneTriangle reports
9.6x at 127K. Our compute roofline at 32K is 13.4x. The source prefill runs with cuDNN SDPA (1.5x faster than torch's default pick) and the KV write copies whole blocks.
Before those two changes, 32K was 0.860 s (5.4x).

Serving bugs found on the way: vLLM 0.34 ignores `VLLM_ATTENTION_BACKEND` (pass `attention_backend`). FlashInfer reads
each KV block as `[2, H, N, D]`, so the connector refuses it. And `NewRequestData.num_computed_tokens` already counts the
connector's own tokens, so it is not the prefix-cache hit.

### Qwen2.5-3B -> Qwen2.5-72B at 128K in vLLM (YaRN x4, tensor parallel 2 on 2x B200, FLASH_ATTN, one request at a time)

| Prompt | Native vLLM TP=2 | Mapped (each rank maps its own KV heads; sink template, 1024-token native tail) | Speedup |
|---|---|---|---|
| 32K | 2.54 s | 0.542 s | 4.7x |
| 64K | 6.24 s | 1.25 s | 5.0x |
| 130,816 | 17.1 s | 3.38 s | 5.05x |

Quality at 128K on 4 real PG19 prompts: approximate first-token KL 0.056 and top-1 agreement 3/4. OneTriangle reports
9.6x at 127K on this pair. Tensor parallelism halves the native time, but every rank still runs the whole 3B source
prefill (3.1 s at 128K), so that prefill is now the bottleneck. Splitting the source across ranks is the next lever.

### Qwen3-1.7B -> Qwen3-32B at 128K in vLLM (YaRN x4 on both models; B200, FLASH_ATTN, one request at a time)

| Prompt | Native vLLM | Mapped (fp8 k24 mixed-calibration maps, 1024-token native tail) | Speedup |
|---|---|---|---|
| 32K | 2.46 s | 0.643 s | 3.8x |
| 64K | 6.82 s | 1.40 s | 4.9x |
| 130,816 | 21.0 s | 3.51 s | 6.0x |

Quality at 128K, on 4 real PG19 prompts of 130,816 tokens: approximate first-token KL 0.048 and top-1 agreement 3/4.
The maps were fit on 1K-token sequences without YaRN. At 128K the connector takes 3.1 s: about half is the 1.7B source
prefill and half is the dense k24 mapper.

### Qwen3-1.7B -> Qwen3-32B (OneTriangle's headline Qwen pair; Modal B200, run cost ~$2.10)

| Variant | KL mean | KL first | HellaSwag (200) | MMLU 5-shot (200) |
|---|---|---|---|---|
| Native 32B | 0 | 0 | 0.695 | 0.805 |
| Run only the 1.7B | 0.752 | 0.871 | 0.50 | 0.635 |
| Ridge k=8 | 0.339 | 0.689 | | |
| Ridge k=8 + RMS restore | 0.272 | 0.588 | | |
| k=16 + RMS restore | 0.276 | 0.560 | 0.61 | 0.65 |
| k=16 + RMS + last 64 tokens native | **0.183** | **0.180** | **0.675 (97%)** | **0.77 (96%)** |
| k=16 + RMS + last 256 tokens native | 0.118 | 0.110 | | |
| **4000-seq calibration**, k=24 + RMS | 0.219 | 0.512 | 0.605 | 0.61 |
| 4000-seq, k=24 + RMS + last 64 native | **0.144** | **0.160** | **0.685 (99%)** | **0.79 (98%)** |
| 4000-seq, k=24 + RMS + first 4 + last 256 native | 0.062 | 0.062 | | |

Measured TTFT (Modal H200, HF forwards, batch 1, dense k=16 mapper, last 64 tokens native):

| Prompt | Native 32B | 1.7B prefill | + mapper | Mapped TTFT | Speedup |
|---|---|---|---|---|---|
| 4K | 0.568 s | 0.047 s | 0.100 s | 0.157 s | 3.6x |
| 8K | 1.223 s | 0.099 s | 0.208 s | 0.288 s | 4.2x |
| 16K | 2.817 s | 0.237 s | 0.453 s | 0.579 s | 4.9x |
| 32K | 7.281 s | 0.641 s | 1.083 s | 1.310 s | **5.6x** |

The table above uses HF SDPA. Re-measured on Modal B200 with flash-attention-2 for both models and a block-accumulate mapper (addmm over source-layer views: no gather copies, FLOPs proportional to k):

| Prompt | Native 32B (HF+FA2) | Mapped TTFT | Speedup | Native vLLM 0.34 (B200) | Speedup vs vLLM |
|---|---|---|---|---|---|
| 4K | 0.345 s | 0.135 s | 2.5x | 0.222 s | 1.6x |
| 8K | 0.761 s | 0.202 s | 3.8x | 0.470 s | 2.3x |
| 16K | 1.833 s | 0.372 s | 4.9x | 1.025 s | 2.8x |
| 32K | 5.120 s | **0.829 s** | 6.2x | 2.484 s | 3.0x |

OneTriangle reports 7.34 s -> 0.87 s (~8.4x) on an H100 at 32K for this pair, against an HF native baseline. Our mapped time matches theirs: 0.83 s. A native baseline matters a lot. vLLM's native prefill is 2x faster than HF+FA2, and it is the fair baseline for a serving claim.

### Best serving result: vLLM 0.34 on an H200, fp8 dense mapper, Triton epilogue, k=24 maps from 4000 calibration sequences

| Prompt | Native vLLM | With connector | Speedup | $/1M prefill tok native -> mapped |
|---|---|---|---|---|
| 4K | 0.433 s | 0.120 s | 3.6x | $0.133 -> $0.037 |
| 8K | 0.940 s | 0.214 s | 4.4x | $0.145 -> $0.033 |
| 16K | 2.108 s | 0.432 s | 4.9x | $0.162 -> $0.033 |
| 32K | 5.159 s | **1.017 s** | **5.1x** | $0.199 -> $0.039 |

First-token quality against native vLLM on 16 FineWeb-Edu prompts: approximate KL 0.119, top-1 agreement 94%.

With a 1024-token native tail (quality-first), H200 TTFT is 0.296 s at 8K, 0.545 s at 16K and 1.188 s at 32K: 3.2x / 3.9x / 4.3x against native vLLM. At 32K the tail costs +0.16 s and cuts KL on books from 0.188 to 0.079.

The recommended default uses mixed web+book calibration maps. It serves at the same speed (32K: 1.029 s, 5.0x) with KL 0.125 and top-1 94% on the same prompts, and it is 42% better on 8K books.

### Inside vLLM 0.34 on an H200 at up to 32K (Modal; 1.7B source and dense mapper on the same GPU; k=16 maps from 500-sequence calibration)

| Prompt | Native vLLM | With connector | Speedup |
|---|---|---|---|
| 4K | 0.433 s | 0.122 s | 3.5x |
| 8K | 0.940 s | 0.230 s | 4.1x |
| 16K | 2.108 s | 0.471 s | 4.5x |
| 32K | 5.159 s | **1.095 s** | **4.7x** |

First-token quality on 16 prompts: approximate KL 0.133, top-1 agreement 87.5%. These maps use 500 calibration sequences; the 4000-sequence maps measured better in HF evaluation.

Prefill cost per 1M prompt tokens, at Modal's H200 price of $4.54/h, batch 1, from the TTFTs above:

| Prompt | Native vLLM | With connector | Cheaper by |
|---|---|---|---|
| 4K | $0.133 | $0.038 | 3.5x |
| 8K | $0.145 | $0.035 | 4.1x |
| 16K | $0.162 | $0.036 | 4.5x |
| 32K | $0.199 | $0.042 | 4.7x |

OneTriangle reports 8.4x at 32K for this pair. Their baseline is native HF forwards on an H100. Ours is native vLLM on the same H200, which is the harder bar.


### Against prefill/decode disaggregation (H200 numbers above, roofline transfer)

With native P/D disaggregation the target still prefills every prompt token, on a prefill GPU. It then ships 256 KB/token of 32B KV to the decode GPU. At 32K that is 8.4 GB: about 19 ms over NVLink (450 GB/s) or about 170 ms over 400 Gb/s InfiniBand. So disaggregation does not lower TTFT: 5.16 s + transfer. It only frees the decode GPU from prefill work. Cross-model transfer cuts the prefill itself: 1.02 s at 32K.

The two compose. The source prefill and mapper can run on a cheap prefill GPU and ship the 1.7B's KV, which is 112 KB/token, 2.3x less than the 32B's. The mapper then runs on the decode GPU. Our first connector version ran the source on a second GPU and copied mapped KV over PCIe; that copy cost 0.23 s at 16K. Shipping source KV instead, and mapping on the receiver, is the better split.

### Inside vLLM 0.34: `CrossModelConnector` (Vast 2x RTX PRO 6000, 1.7B + block mapper on GPU1, Qwen3-32B on GPU0)

Final configuration: the 1.7B, a dense one-GEMM mapper and Qwen3-32B all on one RTX PRO 6000. The connector writes mapped K/V straight into vLLM's paged blocks.

| Prompt | Native vLLM | With connector | Speedup |
|---|---|---|---|
| 4K | 0.804 s | **0.223 s** | **3.6x** |
| 8K | 1.607 s | **0.381 s** | **4.2x** |
| 16K | 3.683 s | **0.761 s** | **4.8x** |

The first-token quality check against native vLLM gives approximate KL 0.111 and 94% top-1 agreement.

Adding a fused Triton epilogue (k_norm RMS restore and RoPE in one pass) on top of the fp8 mapper gives 0.197 s / 0.339 s / 0.744 s at 4K / 8K / 16K. At that point the connector is GEMM-bound: the dense K+V map takes about 0.24 s at 16K and the 1.7B prefill about 0.17 s.

With an fp8 (e4m3, row-wise scaled) dense mapper, centered per request, TTFT is 0.204 s at 4K (3.9x), 0.350 s at 8K (4.6x) and 0.762 s at 16K (4.8x). Quality is unchanged: approximate KL 0.116, top-1 94%. The fp8 GEMM alone is 1.61x faster than bf16 on this card. At 16K, though, the per-layer RMS restore and RoPE passes, which are memory-bound, now limit the mapper. At 16K, the connector spends about 0.17 s on the source prefill and about 0.45 s on the dense mapper. vLLM spends about 0.08 s on the 64-token tail.

The first version ran the source on GPU1 with a per-layer mapper and copied over PCIe:

| Prompt | Native vLLM | With connector | Speedup |
|---|---|---|---|
| 4K | 0.804 s | 0.482 s | 1.67x |
| 8K | 1.607 s | 0.896 s | 1.79x |
| 16K | 3.683 s | 1.798 s | 2.05x |

Quality inside vLLM, on 16 FineWeb-Edu prompts of 961 tokens, against native vLLM:

- first-token approximate KL over the top-20 log-probabilities: 0.115;
- top-1 agreement: 94%.

The connector reports the mapped prefix to vLLM as an external KV hit. vLLM therefore prefills only the last 64 tokens natively, and the connector writes the mapped K/V into the paged blocks. Remaining costs in that first version:

- the 1.7B prefill runs as an HF model;
- the mapper uses a Python loop over layers;
- every layer crosses PCIe from GPU1.

At 32K the compute roofline caps the speedup at 10.9x. Two parts of that path ran well above their floors:

- The mapper takes 0.44 s against a floor of about 0.25 s. The dense GEMM does 1.75x the needed FLOPs, and the RoPE handling adds overhead.
- The 64-token tail forward takes 0.23 s against a floor of about 20 ms.

### Qwen3-14B -> Qwen3-32B (NVIDIA paper's main pair; Vast H200 NVL, $3.27/h, run cost ~$0.70)

| Variant | KL mean | KL first | HellaSwag (200) | MMLU 5-shot (200) |
|---|---|---|---|---|
| Native 32B | 0 | 0 | 0.695 | 0.805 |
| Run only the 14B | 0.258 | 0.275 | 0.665 | 0.81 |
| Ridge k=8 (paper config) | 0.112 | 0.182 | | |
| Ridge k=8 + RMS key restore | 0.094 | 0.174 | | |
| Ridge k=16 + RMS restore | 0.096 | 0.214 | 0.69 | 0.81 |
| k=16 + RMS + last 64 tokens native | **0.063** | **0.069** | 0.70 | 0.85 |
| k=16 + RMS + last 256 tokens native | 0.037 | 0.041 | | |
| **2000-seq calibration**, k=16 + RMS | 0.079 | 0.142 | 0.67 | 0.815 |
| 2000-seq, k=16 + RMS + last 64 native | 0.052 | 0.052 | 0.69 (100%) | 0.835 |
| 2000-seq, k=16 + RMS + first 4 + last 64 native | **0.045** | **0.048** | | |

This matches the paper's retention: it reports 97.6% on average, and we measure 99% on HellaSwag and 100% on MMLU. MMLU on 200 items can't separate this pair, since the 14B alone scores as high as the 32B. KL gives the finer signal, and two additions cut it:

- RMS key restore cuts KL by 16%.
- Recomputing the last 64 prompt tokens natively cuts it by another third.

KL is KL(native || mapped) of the target's next-token distribution. The source model's cache maps a 960-token FineWeb-Edu prefix, and the target processes the 64-token continuation natively.

### Qwen3-1.7B -> Qwen3-8B (L40S, 500 x 1024 calibration, 64 eval sequences)

| Variant | KL mean | KL first token | Top-1 agree | Cost |
|---|---|---|---|---|
| Target sees no context | 1.049 | 6.66 | 0.622 | |
| Run only the 1.7B | 0.455 | 0.653 | 0.727 | |
| Ridge k=8 (NVIDIA paper) | 0.244 | 0.599 | 0.806 | $0.27 per run |
| Ridge k=8 + RMS key restore | **0.191** | 0.531 | 0.822 | same run |
| Exact K + mapped V (oracle) | 0.107 | 0.108 | 0.874 | |
| Mapped K + exact V (oracle) | 0.155 | 0.483 | 0.835 | |

- **RMS key restore:** puts back the target's per-token, per-head key RMS, which is the scale its `k_norm` would impose. This is our reading of "restore relative importances", and it cuts KL by 22%.
- **Values limit quality on this pair:** the exact-K + mapped-V and mapped-K + exact-V oracle rows show it. OneTriangle's blog reports the reverse for its pairs.
- **HellaSwag retention:** 92%.
- **MMLU 5-shot:** below the 1.7B alone when the target recomputes only the last token. The last-N native recompute run is next.
