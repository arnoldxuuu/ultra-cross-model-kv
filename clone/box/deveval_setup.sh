#!/bin/bash
# DevEval test environment on a throwaway CPU box (the authors' env: Python 3.9, pytest 7.4 + pytest-runner, their
# requirement.txt), plus data.jsonl and Source_Code/ under /work/deveval. Then the ground-truth pass list.
#   bash box/deveval_setup.sh && /venv-dev/bin/python -m kvx.deveval_eval --root /work/deveval --out /work/deveval/gt.jsonl
set -euo pipefail
W=/work/deveval; mkdir -p $W; cd $W
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH=$HOME/.local/bin:$PATH
uv python install 3.9
[ -x /venv-dev/bin/python ] || uv venv --allow-existing -p 3.9 /venv-dev
[ -d repo ] || git clone -q --depth 1 https://github.com/seketeam/DevEval.git repo
uv pip install -p /venv-dev/bin/python -q "pytest==7.4.0" "pytest-runner==6.0.0" "func_timeout==4.3.5" tqdm \
  "setuptools==68.0.0" pip wheel -r repo/requirement.txt   # setup.py pytest fetches repo deps via pip
[ -f data.jsonl ] || tar -xzf repo/data.tar.gz -C $W
[ -f data.jsonl ] || find $W -maxdepth 3 -name data.jsonl -exec mv {} $W/ \;
if [ ! -d Source_Code ]; then
  curl -sL -o Source_Code.tar.gz https://huggingface.co/datasets/LJ0815/DevEval/resolve/main/Source_Code.tar.gz
  tar -xzf Source_Code.tar.gz -C $W && rm Source_Code.tar.gz
fi
ls $W; wc -l $W/data.jsonl; ls $W/Source_Code | head
