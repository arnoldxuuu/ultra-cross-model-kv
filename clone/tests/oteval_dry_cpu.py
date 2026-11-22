"""CPU dry run of kvx/oteval.py: every OneTriangle suite loads and scores with a fake always-"A" generator (no GPU).
Run: venv with transformers 4.56.2 + datasets<4.1 + math-verify; needs Magni src at kvx/ext/ultra/src."""
import sys, json
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "..", "src"), os.path.join(HERE, "..")]
from transformers import AutoTokenizer
from kvx.oteval import run_suites
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
ans = tok.encode("A", add_special_tokens=False)
gen = lambda prompts, max_new: [ans for _ in prompts]   # always answers "A"
r = run_suites(gen, tok, ["ARC-Challenge", "QuALITY", "ZebraLogic", "MATH-500", "LongBench v2", "ContractNLI", "BAMBOO PaperQA", "HellaSwag", "PIQA"],
               limits={k: 8 for k in ["ARC-Challenge", "QuALITY", "ZebraLogic", "MATH-500", "LongBench v2", "ContractNLI", "BAMBOO PaperQA", "HellaSwag", "PIQA"]})
print({k: ({kk: v[kk] for kk in v if kk in ("score", "acc", "accuracy", "n", "error")} if isinstance(v, dict) else v) for k, v in r.items()})
