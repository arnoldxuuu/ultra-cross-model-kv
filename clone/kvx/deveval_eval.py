"""DevEval pass@1 runner (seketeam/DevEval semantics: splice the completion over body_position, run the item's tests
with `python setup.py pytest --addopts <test>` in the repo, restore the file). Repos run in parallel, the items of one
repo one after another (they edit the repo in place). Run on a throwaway box: the tests execute the repos' code.

  python -m kvx.deveval_eval --root /dev/DevEval --out gt.jsonl                       # ground truth (which items pass)
  python -m kvx.deveval_eval --root /dev/DevEval --completions c.jsonl --out res.jsonl
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor


def adjust_indent(code, n):
    return textwrap.indent(textwrap.dedent(code), " " * n)


def run_tests(root, item, timeout):
    """'Pass' / 'Error' / 'TimeOut', exactly as pass_k.execution_tests (every test must pass)."""
    cwd = os.path.join(root, "Source_Code", item["project_path"])
    for test in sorted(set(item["tests"])):
        try:
            p = subprocess.run([sys.executable, "setup.py", "pytest", "--addopts", test], cwd=cwd, timeout=timeout,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.TimeoutExpired:
            return "TimeOut"
        if p.returncode != 0:
            return "Error"
    return "Pass"


def eval_one(root, item, completion, timeout):
    if completion is None:  # ground truth: the repo as shipped
        return run_tests(root, item, timeout)
    if completion.strip() in ("", "pass"):
        return "Error"
    path = os.path.join(root, "Source_Code", item["completion_path"])
    tmp = path + ".kvx_orig"
    shutil.copyfile(path, tmp)
    try:
        lines = open(path).readlines()
        s, e = item["body_position"][0] - 1, item["body_position"][1]
        body = adjust_indent(completion, item["indent"])
        open(path, "w").write("".join(lines[:s] + ["\n", body, "\n"] + lines[e:]))
        return run_tests(root, item, timeout)
    finally:
        shutil.move(tmp, path)


def eval_repo(args):
    root, items, comps, timeout = args
    out = []
    for it in items:
        c = None if comps is None else comps.get(it["namespace"])
        r = "Missing" if (comps is not None and c is None) else eval_one(root, it, c, timeout)
        out.append(dict(namespace=it["namespace"], project_path=it["project_path"], result=r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)              # holds data.jsonl and Source_Code/
    ap.add_argument("--out", required=True)
    ap.add_argument("--completions", default="")          # jsonl: namespace, completion (function body)
    ap.add_argument("--only", default="")                 # jsonl/txt of namespaces to evaluate
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--timeout", type=int, default=60)
    a = ap.parse_args()
    data = [json.loads(l) for l in open(os.path.join(a.root, "data.jsonl"))]
    comps = None
    if a.completions:
        comps = {}
        for l in open(a.completions):
            d = json.loads(l)
            comps[d["namespace"]] = d["completion"]
    if a.only:
        keep = {json.loads(l)["namespace"] if l.lstrip().startswith("{") else l.strip() for l in open(a.only) if l.strip()}
        data = [d for d in data if d["namespace"] in keep]
    elif comps is not None:
        data = [d for d in data if d["namespace"] in comps]
    by = defaultdict(list)
    for d in data:
        by[d["project_path"]].append(d)
    jobs = sorted(by.values(), key=len, reverse=True)
    res = []
    with ProcessPoolExecutor(a.workers) as ex, open(a.out, "w") as f:
        for out in ex.map(eval_repo, [(a.root, j, comps, a.timeout) for j in jobs]):
            for r in out:
                f.write(json.dumps(r) + "\n")
            f.flush()
            res += out
            print(f"{len(res)}/{len(data)} pass={sum(r['result'] == 'Pass' for r in res)}", flush=True)
    n = len(res)
    print(json.dumps(dict(n=n, pass_at_1=(sum(r["result"] == "Pass" for r in res) / n) if n else None)))


if __name__ == "__main__":
    main()
