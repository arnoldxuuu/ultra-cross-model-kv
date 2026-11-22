#!/bin/bash
# Brief step 4 on one 2x B200 box: Qwen2.5-3B -> Qwen2.5-72B.
#  1. refit maps with the full recipe (mixed web+book calibration, 4000 seqs, k16, sink template) -- the old 72B maps
#     were FineWeb-only, 500 seqs, and live on Jerry's Modal volume anyway
#  2. TTFT at 32K / 64K / ~128K (YaRN x4), TP=2: vLLM native, vLLM mapped with the sharded source prefill
#     (kvx/tp_source.py), and HF native (the matched basis, decision D1); 128K quality on 4 PG19 prompts
#  3. the 8 routed cells (QuALITY, ContractNLI, BAMBOO, LongBench v2 via transfer; ARC-C, HellaSwag, PIQA, ZebraLogic
#     native route under 1,024 tokens) -- via kvx/oteval.py
set -uo pipefail
cd /work/clone
export PYTHONPATH=/work/clone:/work/xkv/src HF_HOME=/vol/hf HF_HUB_ENABLE_HF_TRANSFER=1
SRC=Qwen/Qwen2.5-3B; TGT=Qwen/Qwen2.5-72B; RID=q25-3b-72b-mix; LOG=/vol/runs/step4.log
echo "== fit $(date -u +%H:%M)" >> $LOG
[ -f /vol/runs/$RID/maps_k16.pt ] || /venv-fit/bin/python run_local.py modal_fit_eval:fit_eval_impl --json \
  "{\"src\":\"$SRC\",\"tgt\":\"$TGT\",\"n_calib\":4000,\"ks\":\"16\",\"modes\":\"none\",\"tails\":\"256\",\"sinks\":\"4\",\"calib_src\":\"mix\",\"save_maps\":1,\"run_id\":\"$RID\",\"n_eval\":64}" >> $LOG 2>&1
MAPS=/vol/runs/$RID/maps_k16.pt; SINK=/vol/runs/$RID/sink_template.pt
COMMON="\"src\":\"$SRC\",\"tgt\":\"$TGT\",\"lens\":\"32768,65536,130816\",\"maxlen\":131072,\"yarn\":4,\"orig_ctx\":32768,\"quality_long\":130816,\"tp\":2"
echo "== vllm mapped (src_tp) $(date -u +%H:%M)" >> $LOG
/venv-serve/bin/python run_local.py modal_vllm:serve_impl --json "{\"mode\":\"mapped\",$COMMON,\"maps\":\"$MAPS\",\"tail\":1024,\"fp8\":1,\"restore\":\"none\",\"sink\":\"$SINK\",\"util\":0.62,\"src_tp\":1,\"tag\":\"-2xb200-128k-mix-tpsrc\"}" >> $LOG 2>&1
echo "== vllm native $(date -u +%H:%M)" >> $LOG
/venv-serve/bin/python run_local.py modal_vllm:serve_impl --json "{\"mode\":\"native\",$COMMON,\"maps\":\"\",\"tail\":0,\"util\":0.85,\"tag\":\"-2xb200-128k\"}" >> $LOG 2>&1
echo "== HF native + HF mapped TTFT (matched basis), 32K (2 GPUs, place_on) $(date -u +%H:%M)" >> $LOG
/venv-fit/bin/python run_local.py modal_ttft:ttft --json "{\"src\":\"$SRC\",\"tgt\":\"$TGT\",\"maps_run\":\"$RID\",\"k\":16,\"lens\":\"32768\",\"tail\":256,\"dense\":1,\"attn\":\"sdpa\"}" >> $LOG 2>&1
echo "== routed cells (requests < 1,024 tokens native via min_len; longer ones transferred) $(date -u +%H:%M)" >> $LOG
/venv-serve/bin/python run_local.py kvx.oteval:oteval_impl --json "{\"tgt\":\"$TGT\",\"mode\":\"mapped\",\"src\":\"$SRC\",\"maps\":\"$MAPS\",\"sink\":\"$SINK\",\"restore\":\"none\",\"tail\":256,\"min_len\":1024,\"tp\":2,\"src_tp\":1,\"util\":0.80,\"suites\":\"ARC-Challenge,HellaSwag,PIQA,ZebraLogic,QuALITY,ContractNLI,BAMBOO PaperQA,LongBench v2\",\"limits\":\"{\\\"ARC-Challenge\\\":100,\\\"HellaSwag\\\":100,\\\"PIQA\\\":100,\\\"ZebraLogic\\\":100,\\\"QuALITY\\\":100,\\\"ContractNLI\\\":100,\\\"LongBench v2\\\":100}\",\"tag\":\"routed-72b\"}" >> $LOG 2>&1
echo "== step4 done $(date -u +%H:%M)" >> $LOG
