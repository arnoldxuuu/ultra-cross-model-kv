#!/bin/bash
# Box 6: does code in the calibration mix fix DevEval retention (52-56% with web+book maps)? Per target (14B, 4B, 8B):
# fit k24 maps on mixcode (4000 seqs: web, books, Python; DevEval repos excluded), then DevEval mapped (tail 256 and
# TAIL2) and QuALITY mapped (no regression vs the mix maps: 14B .686, 4B .650). 8B also gets DevEval native.
cd /work/clone
export PYTHONPATH=/work/clone:/work/xkv/src HF_HOME=/vol/hf HF_HUB_ENABLE_HF_TRANSFER=1 KVX_DEVEVAL_REPOS=/vol/deveval/repos.txt
LOG=/vol/runs/box6.log; DL=$(( $(date +%s) + ${BUDGET_S:-8400} )); SRC=Qwen/Qwen3-1.7B; TAIL2=${TAIL2:-1024}
left() { echo $(( DL - $(date +%s) )); }
gpu_clean() { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done; sleep 5; }
step() {  # step NAME VENV MODULE:FUNC JSON
  local s; s=$(left); [ "$s" -lt 300 ] && { echo "== $1 SKIPPED (deadline) $(date -u +%H:%M)" >> $LOG; return; }
  echo "== $1 $(date -u +%H:%M) (budget ${s}s)" >> $LOG
  timeout -k 30 "$s" /$2/bin/python run_local.py "$3" --json "$4" >> /vol/runs/box6_$1.log 2>&1
  echo "== $1 rc=$? $(date -u +%H:%M)" >> $LOG; gpu_clean; }
for T in 14B 4B 8B; do
  R=q17-q${T%B}-mixcode; M=/vol/runs/$R
  [ -f $M/maps_k24.pt ] || step fit$T venv-fit modal_fit_eval:fit_eval_impl "{\"src\":\"$SRC\",\"tgt\":\"Qwen/Qwen3-$T\",\"n_calib\":4000,\"ks\":\"24\",\"modes\":\"rms\",\"tails\":\"64\",\"sinks\":\"4\",\"calib_src\":\"mixcode\",\"save_maps\":1,\"run_id\":\"$R\",\"n_eval\":32,\"one_pass_\":1}"
  [ -f $M/maps_k24.pt ] || { echo "== $T no maps, next" >> $LOG; continue; }
  for TL in 256 $TAIL2; do
    step dev$T-mapped-mixcode-t$TL venv-serve kvx.deveval:deveval_gen_impl "{\"tgt\":\"Qwen/Qwen3-$T\",\"mode\":\"mapped\",\"src\":\"$SRC\",\"maps\":\"$M/maps_k24.pt\",\"sink\":\"$M/sink_template.pt\",\"tail\":$TL,\"util\":0.60,\"chunk\":8,\"tag\":\"dev$T-mapped-mixcode-t$TL\"}"
  done
  [ $T = 8B ] && step dev8B-native venv-serve kvx.deveval:deveval_gen_impl "{\"tgt\":\"Qwen/Qwen3-8B\",\"mode\":\"native\",\"util\":0.60,\"tag\":\"dev8B-native\"}"
  step qual$T-mixcode venv-serve kvx.oteval:oteval_impl "{\"tgt\":\"Qwen/Qwen3-$T\",\"mode\":\"mapped\",\"src\":\"$SRC\",\"maps\":\"$M/maps_k24.pt\",\"sink\":\"$M/sink_template.pt\",\"suites\":\"QuALITY\",\"limits\":\"{\\\"QuALITY\\\":500}\",\"tail\":256,\"min_len\":512,\"util\":0.60,\"chunk\":8,\"tag\":\"col$T-mapped-mixcode-qual\"}"
done
echo "BOX6-DONE $(date -u +%H:%M)" >> $LOG
