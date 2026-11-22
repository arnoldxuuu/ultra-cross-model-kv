#!/bin/bash
# Box 5 (one g7e, 3 h backstop): (d) prefill throughput and $/1M prompt tokens at c > 1, native vs mapped, 1.7B->14B
# and 1.7B->32B; then the 14B/4B LongBench v2 mapped reruns at util 0.60 (0.80/0.85 OOMed); then the 4B reasoning-on
# arms (MATH-500, ZebraLogic, 12K tokens) until the deadline. Every step is bounded so results land before the backstop.
cd /work/clone
export PYTHONPATH=/work/clone:/work/xkv/src HF_HOME=/vol/hf HF_HUB_ENABLE_HF_TRANSFER=1
LOG=/vol/runs/box5.log; DL=$(( $(date +%s) + ${BUDGET_S:-8400} ))   # all work ends 2 h 20 min after start (backstop 3 h from launch)
SRC=Qwen/Qwen3-1.7B
left() { echo $(( DL - $(date +%s) )); }
gpu_clean() {  # vLLM EngineCore children outlive a killed parent: free the GPU before the next step
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done; sleep 5; }
step() {  # step NAME MODULE:FUNC JSON
  local s; s=$(left); [ "$s" -lt 300 ] && { echo "== $1 SKIPPED (deadline) $(date -u +%H:%M)" >> $LOG; return; }
  echo "== $1 $(date -u +%H:%M) (budget ${s}s)" >> $LOG
  timeout -k 30 "$s" /venv-serve/bin/python run_local.py "$2" --json "$3" >> /vol/runs/box5_$1.log 2>&1
  echo "== $1 rc=$? $(date -u +%H:%M)" >> $LOG; gpu_clean; }
M14=/vol/runs/q17-q14-mix; M32=/vol/runs/q17-q32-mix; M4=/vol/runs/q17-q4-mix
G14="8192:1,4,16;16384:1,4,8;32768:1,4"; G32="8192:1,2,4"
for MODE in native mapped; do
  step tp14-$MODE kvx.throughput:throughput_impl "{\"tgt\":\"Qwen/Qwen3-14B\",\"mode\":\"$MODE\",\"src\":\"$SRC\",\"maps\":\"$M14/maps_k24.pt\",\"sink\":\"$M14/sink_template.pt\",\"grid\":\"$G14\",\"util\":0.70,\"tag\":\"tp14-$MODE\"}"
done
for MODE in native mapped; do
  step tp32-$MODE kvx.throughput:throughput_impl "{\"tgt\":\"Qwen/Qwen3-32B\",\"mode\":\"$MODE\",\"src\":\"$SRC\",\"maps\":\"$M32/maps_k24.pt\",\"sink\":\"$M32/sink_template.pt\",\"grid\":\"$G32\",\"util\":0.80,\"tag\":\"tp32-$MODE\"}"
done
for T in 4B 14B; do
  M=/vol/runs/q17-q${T%B}-mix
  step lb$T kvx.oteval:oteval_impl "{\"tgt\":\"Qwen/Qwen3-$T\",\"mode\":\"mapped\",\"src\":\"$SRC\",\"maps\":\"$M/maps_k24.pt\",\"sink\":\"$M/sink_template.pt\",\"suites\":\"LongBench v2\",\"limits\":\"{\\\"LongBench v2\\\":97}\",\"tail\":256,\"min_len\":512,\"util\":0.60,\"chunk\":8,\"tag\":\"col$T-mapped-lb2\"}"
done
for S in MATH-500 ZebraLogic; do
  N=500; [ $S = ZebraLogic ] && N=200
  step think4B-${S%%-*} kvx.oteval:oteval_impl "{\"tgt\":\"Qwen/Qwen3-4B\",\"mode\":\"mapped\",\"src\":\"$SRC\",\"maps\":\"$M4/maps_k24.pt\",\"suites\":\"$S\",\"limits\":\"{\\\"$S\\\":$N}\",\"max_new\":\"{\\\"$S\\\":12288}\",\"thinking\":1,\"tail\":16,\"min_len\":32,\"util\":0.85,\"tag\":\"col4B-mapped-think-${S%%-*}\"}"
done
echo "BOX5-DONE $(date -u +%H:%M)" >> $LOG
