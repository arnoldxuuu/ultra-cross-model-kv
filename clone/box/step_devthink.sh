#!/bin/bash
# Box 7: DevEval with web+book+Python (mixcode) maps. Reasoning on (OneTriangle's code-7 rows) for 14B/8B/4B at the
# 1024-token native tail, then 32B (2000-seq fit with maps offloaded) reasoning off. One deadline bounds every step.
cd /work/clone
export PYTHONPATH=/work/clone:/work/xkv/src HF_HOME=/vol/hf HF_HUB_ENABLE_HF_TRANSFER=1 KVX_DEVEVAL_REPOS=/vol/deveval/repos.txt
LOG=/vol/runs/box7.log; DL=$(( $(date +%s) + ${BUDGET_S:-16800} )); SRC=Qwen/Qwen3-1.7B
left() { echo $(( DL - $(date +%s) )); }
gpu_clean() { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done; sleep 5; }
step() {  # step NAME VENV MODULE:FUNC JSON
  local s; s=$(left); [ "$s" -lt 300 ] && { echo "== $1 SKIPPED (deadline) $(date -u +%H:%M)" >> $LOG; return; }
  echo "== $1 $(date -u +%H:%M) (budget ${s}s)" >> $LOG
  timeout -k 30 "$s" /$2/bin/python run_local.py "$3" --json "$4" >> /vol/runs/box7_$1.log 2>&1
  echo "== $1 rc=$? $(date -u +%H:%M)" >> $LOG; gpu_clean; }
fit() {  # fit T NCALIB EXTRA_JSON
  local R=q17-q${1%B}-mixcode
  [ -f /vol/runs/$R/maps_k24.pt ] || step fit$1 venv-fit modal_fit_eval:fit_eval_impl "{\"src\":\"$SRC\",\"tgt\":\"Qwen/Qwen3-$1\",\"n_calib\":$2,\"ks\":\"24\",\"modes\":\"rms\",\"tails\":\"64\",\"sinks\":\"4\",\"calib_src\":\"mixcode\",\"save_maps\":1,\"run_id\":\"$R\",\"n_eval\":32$3}"
  touch /vol/runs/$R/FIT-DONE 2>/dev/null; }
dev() {  # dev T THINK UTIL MAXNEW
  local M=/vol/runs/q17-q${1%B}-mixcode
  [ -f $M/maps_k24.pt ] || { echo "== $1 no maps" >> $LOG; return; }
  step dev$1-mapped-mixcode-t1024-think$2 venv-serve kvx.deveval:deveval_gen_impl "{\"tgt\":\"Qwen/Qwen3-$1\",\"mode\":\"mapped\",\"src\":\"$SRC\",\"maps\":\"$M/maps_k24.pt\",\"sink\":\"$M/sink_template.pt\",\"tail\":1024,\"util\":$3,\"chunk\":8,\"thinking\":$2,\"max_new\":$4,\"max_len\":40960,\"tag\":\"dev$1-mapped-mixcode-t1024-think$2\"}"; }
for T in 14B 8B 4B; do fit $T 4000 ",\"one_pass_\":1"; dev $T 1 0.60 6144; done
fit 32B 2000 ",\"one_pass_\":0,\"offload_maps\":1"
dev 32B 0 0.80 1024
echo "BOX7-DONE $(date -u +%H:%M)" >> $LOG
