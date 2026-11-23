#!/bin/bash
# Box 8: does math in the calibration mix close the MATH-500 (reasoning off) rows? Per target 14B/8B/4B: fit k24 maps
# on mixall (quarters: web, books, Python, OpenWebMath), MATH-500 mapped (tail 16, as before), QuALITY mapped (check),
# and for 14B DevEval mapped t1024 (check the code gain holds).
cd /work/clone
export PYTHONPATH=/work/clone:/work/xkv/src HF_HOME=/vol/hf HF_HUB_ENABLE_HF_TRANSFER=1 KVX_DEVEVAL_REPOS=/vol/deveval/repos.txt
LOG=/vol/runs/box8.log; DL=$(( $(date +%s) + ${BUDGET_S:-14400} )); SRC=Qwen/Qwen3-1.7B
left() { echo $(( DL - $(date +%s) )); }
gpu_clean() { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done; sleep 5; }
step() {
  local s; s=$(left); [ "$s" -lt 300 ] && { echo "== $1 SKIPPED (deadline) $(date -u +%H:%M)" >> $LOG; return; }
  echo "== $1 $(date -u +%H:%M) (budget ${s}s)" >> $LOG
  timeout -k 30 "$s" /$2/bin/python run_local.py "$3" --json "$4" >> /vol/runs/box8_$1.log 2>&1
  echo "== $1 rc=$? $(date -u +%H:%M)" >> $LOG; gpu_clean; }
for T in 14B 8B 4B; do
  R=q17-q${T%B}-mixall; M=/vol/runs/$R
  [ -f $M/maps_k24.pt ] || step fit$T venv-fit modal_fit_eval:fit_eval_impl "{\"src\":\"$SRC\",\"tgt\":\"Qwen/Qwen3-$T\",\"n_calib\":4000,\"ks\":\"24\",\"modes\":\"rms\",\"tails\":\"64\",\"sinks\":\"4\",\"calib_src\":\"mixall\",\"save_maps\":1,\"run_id\":\"$R\",\"n_eval\":32,\"one_pass_\":1}"
  [ -f $M/maps_k24.pt ] || { echo "== $T no maps" >> $LOG; continue; }
  step math$T venv-serve kvx.oteval:oteval_impl "{\"tgt\":\"Qwen/Qwen3-$T\",\"mode\":\"mapped\",\"src\":\"$SRC\",\"maps\":\"$M/maps_k24.pt\",\"suites\":\"MATH-500\",\"limits\":\"{\\\"MATH-500\\\":500}\",\"tail\":16,\"min_len\":32,\"util\":0.85,\"tag\":\"col$T-mapped-mixall-math\"}"
  step qual$T venv-serve kvx.oteval:oteval_impl "{\"tgt\":\"Qwen/Qwen3-$T\",\"mode\":\"mapped\",\"src\":\"$SRC\",\"maps\":\"$M/maps_k24.pt\",\"sink\":\"$M/sink_template.pt\",\"suites\":\"QuALITY\",\"limits\":\"{\\\"QuALITY\\\":500}\",\"tail\":256,\"min_len\":512,\"util\":0.60,\"chunk\":8,\"tag\":\"col$T-mapped-mixall-qual\"}"
  [ $T = 14B ] && step dev14B-mixall venv-serve kvx.deveval:deveval_gen_impl "{\"tgt\":\"Qwen/Qwen3-14B\",\"mode\":\"mapped\",\"src\":\"$SRC\",\"maps\":\"$M/maps_k24.pt\",\"sink\":\"$M/sink_template.pt\",\"tail\":1024,\"util\":0.60,\"chunk\":8,\"tag\":\"dev14B-mapped-mixall-t1024\"}"
done
# reasoning-on DevEval with room to finish: at 6144 new tokens 186/500 8B replies never closed their reasoning (.294)
for T in 8B 4B; do
  M=/vol/runs/q17-q${T%B}-mixall; [ -f $M/maps_k24.pt ] || continue
  step dev$T-think12k venv-serve kvx.deveval:deveval_gen_impl "{\"tgt\":\"Qwen/Qwen3-$T\",\"mode\":\"mapped\",\"src\":\"$SRC\",\"maps\":\"$M/maps_k24.pt\",\"sink\":\"$M/sink_template.pt\",\"tail\":1024,\"util\":0.60,\"chunk\":8,\"thinking\":1,\"max_new\":12288,\"max_len\":40960,\"tag\":\"dev$T-mapped-mixall-t1024-think12k\"}"
done
# 32B DevEval reasoning off with the mixcode maps fit on box 7 (uploaded): max_len 33792 (40960 needs 10 GiB of KV,
# the 32B leaves 9.1 GiB on one 96 GB GPU)
M=/vol/runs/q17-q32-mixcode
[ -f $M/maps_k24.pt ] && step dev32B-mixcode venv-serve kvx.deveval:deveval_gen_impl "{\"tgt\":\"Qwen/Qwen3-32B\",\"mode\":\"mapped\",\"src\":\"$SRC\",\"maps\":\"$M/maps_k24.pt\",\"sink\":\"$M/sink_template.pt\",\"tail\":1024,\"util\":0.80,\"chunk\":8,\"tag\":\"dev32B-mapped-mixcode-t1024\"}"
echo "BOX8-DONE $(date -u +%H:%M)" >> $LOG
