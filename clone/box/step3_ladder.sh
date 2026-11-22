#!/bin/bash
# Brief step 3 on one box (2x RTX PRO 6000 96 GB or 1x H200): Qwen3-1.7B -> 4B / 8B / 14B / 32B ladder.
#  1. refit maps with the recipe (mixed web+book calibration, 4000 seqs, k24, rms restore, sink template)
#  2. OneTriangle's suites, native and mapped, through vLLM (kvx/oteval.py), at their n
#  3. TTFT on three bases at 8/16/32K in the same boot: HF native, vLLM native, vLLM mapped (decision D1)
# Writes everything under /vol/runs; pull with box/pull.sh. Usage on the box: bash box/step3_ladder.sh [targets]
set -uo pipefail
cd /work/clone
export PYTHONPATH=/work/clone:/work/xkv/src HF_HOME=/vol/hf HF_HUB_ENABLE_HF_TRANSFER=1 TRITON_INTERPRET=0
TGTS="${1:-8B 32B 4B 14B}"     # cells per $: 8B and 32B first (also the 1.7B->8B tab and the 32B TTFT curve)
SRC=Qwen/Qwen3-1.7B
SUITES="ARC-Challenge,MATH-500,ZebraLogic,QuALITY,LongBench v2"
LOG=/vol/runs/step3.log
for T in $TGTS; do
  TGT=Qwen/Qwen3-$T; RID=q17-q${T%B}-mix
  echo "== $T fit $(date -u +%H:%M)" >> $LOG
  [ -f /vol/runs/$RID/maps_k24.pt ] || /venv-fit/bin/python run_local.py modal_fit_eval:fit_eval_impl --json \
    "{\"src\":\"$SRC\",\"tgt\":\"$TGT\",\"n_calib\":4000,\"ks\":\"24\",\"modes\":\"rms\",\"tails\":\"64\",\"sinks\":\"4\",\"calib_src\":\"mix\",\"save_maps\":1,\"run_id\":\"$RID\",\"n_eval\":64}" >> $LOG 2>&1
  MAPS=/vol/runs/$RID/maps_k24.pt
  for MODE in native mapped; do
    for TH in 0 1; do
      S="$SUITES"; [ $TH = 1 ] && S="MATH-500"   # reasoning-on cells: MATH-500 (DevEval deferred: repo envs)
      echo "== $T oteval $MODE think=$TH $(date -u +%H:%M)" >> $LOG
      /venv-serve/bin/python run_local.py kvx.oteval:oteval_impl --json \
        "{\"tgt\":\"$TGT\",\"mode\":\"$MODE\",\"src\":\"$SRC\",\"maps\":\"$MAPS\",\"suites\":\"$S\",\"thinking\":$TH,\"tag\":\"ladder-$T-$MODE-think$TH\",\"limits\":\"{\\\"LongBench v2\\\": 97}\"}" >> $LOG 2>&1
    done
  done
  echo "== $T ttft $(date -u +%H:%M)" >> $LOG
  for MODE in native mapped; do
    /venv-serve/bin/python run_local.py modal_vllm:serve_impl --json \
      "{\"mode\":\"$MODE\",\"src\":\"$SRC\",\"tgt\":\"$TGT\",\"maps\":\"$MAPS\",\"lens\":\"8192,16384,32768\",\"tail\":64,\"fp8\":1,\"restore\":\"rms\",\"util\":0.85,\"tag\":\"-ladder-$T\"}" >> $LOG 2>&1
  done
  /venv-fit/bin/python run_local.py modal_ttft:ttft --json \
    "{\"src\":\"$SRC\",\"tgt\":\"$TGT\",\"maps_run\":\"$RID\",\"k\":24,\"lens\":\"8192,16384,32768\",\"tail\":64,\"dense\":1,\"attn\":\"sdpa\"}" >> $LOG 2>&1
done
echo "== step3 done $(date -u +%H:%M)" >> $LOG
