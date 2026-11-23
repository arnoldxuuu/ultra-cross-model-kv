#!/bin/bash
# Box 9: mixall maps (web, books, Python, OpenWebMath) on the remaining rows. 32B: fit (2000 seqs, maps offloaded),
# MATH-500 off (web+book maps .766 vs OT .83), QuALITY check. 8B (maps from box 8, uploaded): the 1.7B->8B section's
# long suites that were behind with web+book maps (ContractNLI .586, LongBench .278, BAMBOO .798).
cd /work/clone
export PYTHONPATH=/work/clone:/work/xkv/src HF_HOME=/vol/hf HF_HUB_ENABLE_HF_TRANSFER=1 KVX_DEVEVAL_REPOS=/vol/deveval/repos.txt
LOG=/vol/runs/box9.log; DL=$(( $(date +%s) + ${BUDGET_S:-10800} )); SRC=Qwen/Qwen3-1.7B
left() { echo $(( DL - $(date +%s) )); }
gpu_clean() { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done; sleep 5; }
step() {
  local s; s=$(left); [ "$s" -lt 300 ] && { echo "== $1 SKIPPED (deadline) $(date -u +%H:%M)" >> $LOG; return; }
  echo "== $1 $(date -u +%H:%M) (budget ${s}s)" >> $LOG
  timeout -k 30 "$s" /$2/bin/python run_local.py "$3" --json "$4" >> /vol/runs/box9_$1.log 2>&1
  echo "== $1 rc=$? $(date -u +%H:%M)" >> $LOG; gpu_clean; }
M8=/vol/runs/q17-q8-mixall
if [ -f $M8/maps_k24.pt ]; then
  step long8B venv-serve kvx.oteval:oteval_impl "{\"tgt\":\"Qwen/Qwen3-8B\",\"mode\":\"mapped\",\"src\":\"$SRC\",\"maps\":\"$M8/maps_k24.pt\",\"sink\":\"$M8/sink_template.pt\",\"suites\":\"ContractNLI,BAMBOO PaperQA,LongBench v2\",\"limits\":\"{\\\"ContractNLI\\\":500,\\\"LongBench v2\\\":97}\",\"tail\":256,\"min_len\":512,\"util\":0.60,\"chunk\":8,\"tag\":\"col8B-mapped-mixall-long\"}"
fi
# lm-eval-style loglik with the first 1 / 4 context tokens native (mixall maps; 14B/4B maps uploaded from home)
llstep() { local s; s=$(left); [ "$s" -lt 240 ] && { echo "== $1 SKIPPED (deadline) $(date -u +%H:%M)" >> $LOG; return; }
  echo "== $1 $(date -u +%H:%M) (budget ${s}s)" >> $LOG
  timeout -k 30 "$s" /venv-fit/bin/python run_local.py kvx.oteval:loglik_impl --json "$2" >> /vol/runs/box9_$1.log 2>&1
  echo "== $1 rc=$? $(date -u +%H:%M)" >> $LOG; gpu_clean; }
for F in 1 4; do
  for T in 14B 4B; do
    MM=/vol/runs/q17-q${T%B}-mixall/maps_k24.pt; [ -f $MM ] || continue
    llstep ll$T-f$F "{\"src\":\"$SRC\",\"tgt\":\"Qwen/Qwen3-$T\",\"maps\":\"$MM\",\"suites\":\"ARC-Challenge\",\"n\":500,\"tail\":1,\"first\":$F,\"methods\":\"mapped\",\"tag\":\"col$T-mixall-loglik-f$F\"}"
  done
  [ -f $M8/maps_k24.pt ] && llstep ll8B-f$F "{\"src\":\"$SRC\",\"tgt\":\"Qwen/Qwen3-8B\",\"maps\":\"$M8/maps_k24.pt\",\"suites\":\"ARC-Challenge,HellaSwag,PIQA\",\"n\":500,\"tail\":1,\"first\":$F,\"tag\":\"col8B-mixall-loglik-f$F\"}"
done
R=q17-q32-mixall; M=/vol/runs/$R
[ -f $M/maps_k24.pt ] || step fit32B venv-fit modal_fit_eval:fit_eval_impl "{\"src\":\"$SRC\",\"tgt\":\"Qwen/Qwen3-32B\",\"n_calib\":2000,\"ks\":\"24\",\"modes\":\"rms\",\"tails\":\"64\",\"sinks\":\"4\",\"calib_src\":\"mixall\",\"save_maps\":1,\"run_id\":\"$R\",\"n_eval\":32,\"one_pass_\":0,\"offload_maps\":1}"
if [ -f $M/maps_k24.pt ]; then
  step math32B venv-serve kvx.oteval:oteval_impl "{\"tgt\":\"Qwen/Qwen3-32B\",\"mode\":\"mapped\",\"src\":\"$SRC\",\"maps\":\"$M/maps_k24.pt\",\"suites\":\"MATH-500\",\"limits\":\"{\\\"MATH-500\\\":500}\",\"tail\":16,\"min_len\":32,\"util\":0.80,\"chunk\":8,\"tag\":\"col32B-mapped-mixall-math\"}"
  step dev32B venv-serve kvx.deveval:deveval_gen_impl "{\"tgt\":\"Qwen/Qwen3-32B\",\"mode\":\"mapped\",\"src\":\"$SRC\",\"maps\":\"$M/maps_k24.pt\",\"sink\":\"$M/sink_template.pt\",\"tail\":1024,\"util\":0.80,\"chunk\":8,\"tag\":\"dev32B-mapped-mixall-t1024\"}"
  step qual32B venv-serve kvx.oteval:oteval_impl "{\"tgt\":\"Qwen/Qwen3-32B\",\"mode\":\"mapped\",\"src\":\"$SRC\",\"maps\":\"$M/maps_k24.pt\",\"sink\":\"$M/sink_template.pt\",\"suites\":\"QuALITY\",\"limits\":\"{\\\"QuALITY\\\":500}\",\"tail\":256,\"min_len\":512,\"util\":0.80,\"chunk\":8,\"tag\":\"col32B-mapped-mixall-qual\"}"
fi
echo "BOX9-DONE $(date -u +%H:%M)" >> $LOG
