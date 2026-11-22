"""Score the finish line: targets.json (OneTriangle's 143 published metrics) vs ours.json (our best per row).

Verdicts: B beaten (ours > theirs on the same direction; flag kept for noise/harness), NY not yet (measured, not
beaten, or a claim with no number), NM never measured, NA named reason it cannot be measured, F finding.
ours.json rows: {"id": ..., "ours": float|null, "verdict": optional override, "flag": "noise|harness|...",
"src": "where the number comes from", "reason": "why it cannot be measured"}.
Usage: python targets/score.py [--md] -> totals, and with --md a markdown table of every row.
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, "targets.json")))
O = {r["id"]: r for r in json.load(open(os.path.join(HERE, "ours.json")))}


def verdict(theirs, o):
    if o is None:
        return "NM"
    if o.get("verdict"):
        return o["verdict"]
    if o.get("reason") and o.get("ours") is None:
        return "NA"
    if o.get("ours") is None or theirs is None:
        return "NM"
    return "B" if o["ours"] > theirs else "NY"


rows = []
for c in T["cells"]:
    o = O.get(c["id"])
    rows.append((c["id"], c["theirs"], o, verdict(c["theirs"], o)))
for c in T["ttft_curves"] + T["ttft_homepage_only"] + [{**x, "id": f"claim|{x['id']}"} for x in T["claims"]]:
    o = O.get(c["id"])
    rows.append((c["id"], c.get("speedup") or c.get("speedup_range") or c.get("claim"), o, verdict(None, o)))
tot = Counter(v for _, _, _, v in rows)
print(f"{len(rows)} metrics: " + ", ".join(f"{k} {tot[k]}" for k in ("B", "NY", "NM", "NA", "F")))
if "--md" in sys.argv:
    print("\n| metric | theirs | ours | verdict | flag / source |\n|---|---|---|---|---|")
    for i, th, o, v in rows:
        o = o or {}
        print(f"| {i} | {th} | {o.get('ours', '')} | {v} | {o.get('flag', '')} {o.get('src', '')} {o.get('reason', '')} |")
