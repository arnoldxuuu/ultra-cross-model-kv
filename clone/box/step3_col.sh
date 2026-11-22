#!/bin/bash
# One ladder column (Qwen3-1.7B -> TGT) on one g7e (96 GB), quality only (TTFT curves belong to the kvx-speed seat).
# Matched settings from box 1: long suites mapped with a 256-token native tail + sink template; MATH/ZebraLogic with
# reasoning on and off as OneTriangle's ladder; ARC-C by log-likelihood (their native matches lm-eval scoring).
# Usage: bash box/step3_col.sh 32B   (fit uses one_pass 0 for 32B: K and V stats in two passes, ~11 GB less)
set -uo pipefail
T="$1"; cd /work/clone
export PYTHONPATH=/work/clone:/work/xkv/src HF_HOME=/vol/hf HF_HUB_ENABLE_HF_TRANSFER=1
SRC=Qwen/Qwen3-1.7B; TGT=Qwen/Qwen3-$T; RID=q17-q${T%B}-mix; LOG=/vol/runs/col_$T.log; MAPS=/vol/runs/$RID/maps_k24.pt
SINK=/vol/runs/$RID/sink_template.pt; OP=1; OP0=0; NC=4000; [ "$T" = 32B ] && { OP=0; OP0=1; NC=2000; }
J() { python3 -c "import json,sys; print(json.dumps(json.loads(sys.argv[1])))" "$1"; }
echo "== fit $(date -u +%H:%M)" >> $LOG
[ -f $MAPS ] || /venv-fit/bin/python run_local.py modal_fit_eval:fit_eval_impl --json "$(J "{\"src\":\"$SRC\",\"tgt\":\"$TGT\",\"n_calib\":$NC,\"ks\":\"24\",\"modes\":\"rms\",\"tails\":\"64\",\"sinks\":\"4\",\"calib_src\":\"mix\",\"save_maps\":1,\"run_id\":\"$RID\",\"n_eval\":32,\"one_pass_\":$OP,\"offload_maps\":$OP0}")" >> $LOG 2>&1
UT=0.85; [ "$T" = 32B ] && UT=0.80   # 0.88 starves the connector (OOM); 0.78 leaves 7.1 GiB KV, 8.25 needed at 33K
# long suites (32K prompts): the connector needs ~25 GB outside vLLM's share. Small targets let vLLM fill the share
# with KV cache (65 GB for 4B), so 0.80/0.85 OOMed on 4B and 14B; 0.60 leaves ~38 GB. 32B stays at 0.80 (weights).
UL=0.60; [ "$T" = 32B ] && UL=0.80
run() { local tag; tag=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['tag'])" "$(J "$2")")
  [ -f /vol/runs/oteval/$tag.json ] && { echo "== $1 skipped (have $tag)" >> $LOG; return; }
  echo "== $1 $(date -u +%H:%M)" >> $LOG; /venv-serve/bin/python run_local.py kvx.oteval:oteval_impl --json "$(J "$2")" >> $LOG 2>&1; }
for MODE in native mapped; do
  run "long $MODE" "{\"tgt\":\"$TGT\",\"mode\":\"$MODE\",\"src\":\"$SRC\",\"maps\":\"$MAPS\",\"sink\":\"$SINK\",\"suites\":\"QuALITY,LongBench v2\",\"limits\":\"{\\\"QuALITY\\\":500,\\\"LongBench v2\\\":97}\",\"tail\":256,\"min_len\":512,\"util\":$UL,\"chunk\":8,\"tag\":\"col$T-$MODE-long\"}"
  [ "$MODE" = native ] && continue   # verdicts compare mapped vs theirs; native arms only for the long suites
  run "math off $MODE" "{\"tgt\":\"$TGT\",\"mode\":\"$MODE\",\"src\":\"$SRC\",\"maps\":\"$MAPS\",\"suites\":\"MATH-500\",\"limits\":\"{\\\"MATH-500\\\":500}\",\"tail\":16,\"min_len\":32,\"util\":$UT,\"tag\":\"col$T-$MODE-math\"}"
done
if [ ! -f /vol/runs/oteval/col$T-loglik.json ]; then
echo "== loglik $(date -u +%H:%M)" >> $LOG
/venv-fit/bin/python run_local.py kvx.oteval:loglik_impl --json "$(J "{\"src\":\"$SRC\",\"tgt\":\"$TGT\",\"maps\":\"$MAPS\",\"suites\":\"ARC-Challenge\",\"n\":500,\"tail\":1,\"tag\":\"col$T-loglik\"}")" >> $LOG 2>&1
fi
MODE=mapped
[ "${NOTHINK:-0}" = 1 ] || \
  run "think $MODE" "{\"tgt\":\"$TGT\",\"mode\":\"$MODE\",\"src\":\"$SRC\",\"maps\":\"$MAPS\",\"suites\":\"MATH-500,ZebraLogic\",\"limits\":\"{\\\"MATH-500\\\":500,\\\"ZebraLogic\\\":200}\",\"max_new\":\"{\\\"MATH-500\\\":12288,\\\"ZebraLogic\\\":12288}\",\"thinking\":1,\"tail\":16,\"min_len\":32,\"util\":$UT,\"tag\":\"col$T-$MODE-think\"}"
echo "COL-DONE $(date -u +%H:%M)" >> $LOG
