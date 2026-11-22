#!/bin/bash
# kvx on a rented CUDA box (Vast / EC2), no Modal. Run from rsi-brain:
#   box/push.sh <ssh-target>            # copies this repo + Magni's src as a tarball (the repo is private)
# then on the box:
#   bash /work/clone/box/bootstrap.sh fit|serve|both
# Layout = the Modal volume: /vol/runs (results), /vol/data (token caches), /vol/hf (HF cache). Weights come from the
# Hub (kvx.models.model_path falls back to repo ids when /hf/hub is absent).
set -euo pipefail
WHAT="${1:-both}"
mkdir -p /vol/runs /vol/data /vol/hf
export HF_HOME=/vol/hf HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
if [ "$WHAT" = fit ] || [ "$WHAT" = both ]; then
  if [ ! -x /venv-fit/bin/python ]; then
    python3 -m venv /venv-fit
    /venv-fit/bin/pip install -q "torch==2.8.0" "transformers==4.56.2" accelerate "datasets==3.6.0" numpy hf_transfer \
      "huggingface-hub<1.0" "math-verify[antlr4_13_2]>=0.7" "rouge-score>=0.1.2" "nltk>=3.8" pytest triton
  fi
fi
if [ "$WHAT" = fit5 ]; then  # transformers 5.x: needed for Ministral-3 / Mistral Medium 3.5 ("ministral3" text model)
  if [ ! -x /venv-fit5/bin/python ]; then
    python3 -m venv /venv-fit5
    /venv-fit5/bin/pip install -q "torch==2.9.0" "transformers>=5.20,<6" accelerate "datasets<4.1" numpy hf_transfer \
      "math-verify[antlr4_13_2]>=0.7" pytest triton
  fi
fi
if [ "$WHAT" = serve ] || [ "$WHAT" = both ]; then
  if [ ! -x /venv-serve/bin/python ]; then
    python3 -m venv /venv-serve
    /venv-serve/bin/pip install -q "vllm==0.34.0" hf_transfer "datasets<4.1" "math-verify[antlr4_13_2]>=0.7" \
      "rouge-score>=0.1.2" "nltk>=3.8" pytest
  fi
  command -v nvcc >/dev/null || echo "WARNING: no nvcc on PATH; vLLM's Triton/flashinfer JIT needs a CUDA devel image"
fi
cd /work/clone
export PYTHONPATH=/work/clone:/work/xkv/src
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -x /venv-fit/bin/python ] && /venv-fit/bin/python -m pytest -q tests/test_core_cpu.py -x 2>&1 | tail -1 || true
echo "ready: /venv-fit/bin/python run_local.py MODULE:FUNC --json '{...}'   (fit, HF TTFT, bench)"
echo "       /venv-serve/bin/python run_local.py modal_vllm:serve_impl --json '{...}'   (vLLM serving, oteval)"
