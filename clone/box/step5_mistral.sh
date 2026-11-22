#!/bin/bash
# Brief step 5 on one 2x B200 box: Ministral-3-3B -> Mistral-Medium-3.5-128B (FP8 weights, 8 KV x 128 on both sides).
# Both sides always run YaRN (source factor 16 / orig 16384, target factor 64 / orig 4096, attention_scaling 1.416):
# needs core.unrope's exact inverse for scaled tables (f2a193a). Fit on transformers 5 (venv-fit5), target on cuda:1.
#  1. fit maps (mixed calibration, 2000 seqs first: the 128B forward is the cost), sink template
#  2. TTFT 8/16/32/64K: vLLM native and mapped (TP=2, src_tp), HF native at 32K (matched basis)
#  3. the 6 cells: ARC-C, MATH-500, ZebraLogic, QuALITY, LongBench v2 (+ DevEval deferred: repo envs)
set -uo pipefail
cd /work/clone
export PYTHONPATH=/work/clone:/work/xkv/src HF_HOME=/vol/hf HF_HUB_ENABLE_HF_TRANSFER=1
SRC=mistralai/Ministral-3-3B-Instruct-2512-BF16; TGT=mistralai/Mistral-Medium-3.5-128B; RID=min3b-med128-mix
LOG=/vol/runs/step5.log
# both repos also carry a Mistral-native consolidated copy (Medium: +134 GB): fetch the HF-format files only
for R in $SRC $TGT; do
  /venv-fit5/bin/python -c "from huggingface_hub import snapshot_download as s; s('$R', allow_patterns=['model-*.safetensors','*.json','*.jinja','tekken.json'])" >> $LOG 2>&1
done
echo "== fit $(date -u +%H:%M)" >> $LOG
[ -f /vol/runs/$RID/maps_k16.pt ] || /venv-fit5/bin/python run_local.py modal_fit_eval:fit_eval_impl --json \
  "{\"src\":\"$SRC\",\"tgt\":\"$TGT\",\"n_calib\":2000,\"ks\":\"16\",\"modes\":\"none\",\"tails\":\"256\",\"sinks\":\"4\",\"calib_src\":\"mix\",\"save_maps\":1,\"run_id\":\"$RID\",\"n_eval\":32}" >> $LOG 2>&1
MAPS=/vol/runs/$RID/maps_k16.pt; SINK=/vol/runs/$RID/sink_template.pt
for MODE in mapped native; do
  echo "== vllm $MODE $(date -u +%H:%M)" >> $LOG
  /venv-serve/bin/python run_local.py modal_vllm:serve_impl --json "{\"mode\":\"$MODE\",\"src\":\"$SRC\",\"tgt\":\"$TGT\",\"maps\":\"$MAPS\",\"lens\":\"8192,16384,32768,65536\",\"maxlen\":66048,\"tail\":256,\"fp8\":1,\"restore\":\"none\",\"sink\":\"$SINK\",\"util\":0.80,\"tp\":2,\"src_tp\":1,\"tag\":\"-mistral-2xb200\"}" >> $LOG 2>&1
done
echo "== suites $(date -u +%H:%M)" >> $LOG
for MODE in mapped native; do
  /venv-serve/bin/python run_local.py kvx.oteval:oteval_impl --json "{\"tgt\":\"$TGT\",\"mode\":\"$MODE\",\"src\":\"$SRC\",\"maps\":\"$MAPS\",\"sink\":\"$SINK\",\"restore\":\"none\",\"tail\":16,\"min_len\":32,\"tp\":2,\"src_tp\":1,\"util\":0.80,\"max_len\":33792,\"suites\":\"ARC-Challenge,MATH-500,ZebraLogic,QuALITY,LongBench v2\",\"limits\":\"{\\\"ARC-Challenge\\\":100,\\\"MATH-500\\\":100,\\\"ZebraLogic\\\":100,\\\"QuALITY\\\":100,\\\"LongBench v2\\\":97}\",\"tag\":\"mistral-$MODE\"}" >> $LOG 2>&1
done
echo "== step5 done $(date -u +%H:%M)" >> $LOG
