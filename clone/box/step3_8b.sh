#!/bin/bash
# EC2 g7e (1x RTX PRO 6000 96 GB), < 3 h: Qwen3-1.7B -> 8B. Covers OneTriangle's 1.7B->8B generic-adapter tab
# (HellaSwag, PIQA, ARC-C, ZebraLogic, QuALITY, ContractNLI, BAMBOO, LongBench v2) and the ladder's 8B column
# (ARC-C, MATH-500 off/on, ZebraLogic, QuALITY, LongBench v2; DevEval deferred: per-repo envs), plus the 1.7B->8B
# TTFT curve on three bases (vLLM native, vLLM mapped, HF native+mapped).
set -uo pipefail
cd /work/clone
export PYTHONPATH=/work/clone:/work/xkv/src HF_HOME=/vol/hf HF_HUB_ENABLE_HF_TRANSFER=1
SRC=Qwen/Qwen3-1.7B; TGT=Qwen/Qwen3-8B; RID=q17-q8-mix; LOG=/vol/runs/step3_8b.log; MAPS=/vol/runs/$RID/maps_k24.pt
echo "== fit $(date -u +%H:%M)" >> $LOG
[ -f $MAPS ] || /venv-fit/bin/python run_local.py modal_fit_eval:fit_eval_impl --json \
  "{\"src\":\"$SRC\",\"tgt\":\"$TGT\",\"n_calib\":4000,\"ks\":\"24\",\"modes\":\"rms\",\"tails\":\"64\",\"sinks\":\"4\",\"calib_src\":\"mix\",\"save_maps\":1,\"run_id\":\"$RID\",\"n_eval\":64}" >> $LOG 2>&1
S1="ARC-Challenge,MATH-500,ZebraLogic,QuALITY,LongBench v2,ContractNLI,BAMBOO PaperQA,HellaSwag,PIQA"
LIM='{\"ARC-Challenge\":500,\"MATH-500\":500,\"ZebraLogic\":200,\"QuALITY\":500,\"LongBench v2\":97,\"ContractNLI\":500,\"BAMBOO PaperQA\":84,\"HellaSwag\":500,\"PIQA\":500}'
for MODE in mapped native; do
  echo "== suites $MODE $(date -u +%H:%M)" >> $LOG
  /venv-serve/bin/python run_local.py kvx.oteval:oteval_impl --json "{\"tgt\":\"$TGT\",\"mode\":\"$MODE\",\"src\":\"$SRC\",\"maps\":\"$MAPS\",\"suites\":\"$S1\",\"limits\":\"$LIM\",\"tail\":16,\"min_len\":32,\"util\":0.85,\"tag\":\"q8-$MODE\"}" >> $LOG 2>&1
  echo "== MATH thinking $MODE $(date -u +%H:%M)" >> $LOG
  /venv-serve/bin/python run_local.py kvx.oteval:oteval_impl --json "{\"tgt\":\"$TGT\",\"mode\":\"$MODE\",\"src\":\"$SRC\",\"maps\":\"$MAPS\",\"suites\":\"MATH-500\",\"limits\":\"{\\\"MATH-500\\\":500}\",\"max_new\":\"{\\\"MATH-500\\\":8192}\",\"thinking\":1,\"tail\":16,\"min_len\":32,\"util\":0.85,\"tag\":\"q8-$MODE-think\"}" >> $LOG 2>&1
done
echo "== ttft $(date -u +%H:%M)" >> $LOG
for MODE in native mapped; do
  /venv-serve/bin/python run_local.py modal_vllm:serve_impl --json "{\"mode\":\"$MODE\",\"src\":\"$SRC\",\"tgt\":\"$TGT\",\"maps\":\"$MAPS\",\"lens\":\"8192,16384,32768\",\"tail\":64,\"fp8\":1,\"restore\":\"rms\",\"util\":0.85,\"tag\":\"-q8-g7e\"}" >> $LOG 2>&1
done
/venv-fit/bin/python run_local.py modal_ttft:ttft --json "{\"src\":\"$SRC\",\"tgt\":\"$TGT\",\"maps_run\":\"$RID\",\"k\":24,\"lens\":\"8192,16384,32768\",\"tail\":64,\"dense\":1,\"attn\":\"sdpa\"}" >> $LOG 2>&1
echo "== step3_8b done $(date -u +%H:%M)" >> $LOG
