"""Build the scoreboard of every OneTriangle published number vs ours.

python tools/scoreboard.py --ot ~/kvx-data/ot --out ~/kvx-data/scoreboard.json --readme README.md

Sources (downloaded copies, parsed as data):
  home.html       9 pairs: TTFT tables (native vs OneTriangle s) + peak-speedup headline; a chart point is "measured" when
                  it lies on the chart's solid line, "estimated" on the dashed one.
  bundle/latest-benchmark-bundle.json   Qwen3.5-4B -> Qwen3.8-27B mapped benchmark rows (+ native, source arms).
  metrics_extracted.json                every other number (benchmarks page data, blog notes), from the extraction pass.
Ours: ours.json, keyed by row id: {value, rig, run_id, baseline_vllm, note}. Status rules (directive): same pair (or the
closest open-weight pair, labelled), same context/benchmark, same N or more. TTFT: lower is better; quality: higher.
"""
import argparse, html, json, os, re


# Explicit directions for extracted metric names (first match wins); "context" = a CI bound, p-value, sample count or setup
# number: kept on the board for reference, not a target.
DIRECTION_OVERRIDES = [
    (r"95% ci|ci low|ci high|p-value|standard error|setup, not a score|task examples in calibration|paired samples|"
     r"correlation r\b|same than|range (low|high)|value-minus-key harm|vk diff|value minus key", "context"),
    (r"gap vs native|gap closed|gap recovered|decode tok/s|latency reduction|kl32 relative reduction|kl32 reduction|"
     r"nll reduction|decode faster|accuracy (gain|lift)|speedup", True),
    (r"perplexity|windows worsened|worse than|loop rate", False),
]


def parse_home(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    rows = []
    # one section per pair: a chart (polylines) then a "View latency values" table
    for m in re.finditer(r"<table[^>]*>.*?</table>", t, re.S):
        tab = m.group(0)
        if "Time to first token" not in tab:
            continue
        head = t[max(0, m.start() - 20000):m.start()]
        pairs = re.findall(r">\s*([A-Za-z][^<>]{2,60}?)\s*(?:→|&rarr;)\s*([A-Za-z][^<>]{2,60}?)\s*<", head)
        src, tgt = (html.unescape(pairs[-1][0]).strip(), html.unescape(pairs[-1][1]).strip()) if pairs else ("?", "?")
        htxt = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", head)))
        peak = re.findall(r"([0-9.]+)\s*×\s*peak speedup", htxt)
        peak_ctx = re.findall(r"([0-9.]+[KM])\s*context\s*·\s*(measured|estimated)", htxt)
        solid_max = 0.0
        for pl in re.finditer(r"<polyline points=\"([^\"]+)\"([^>]*)>", head[-6000:]):
            if "dasharray" in pl.group(2):
                continue
            xs = [float(p.split(",")[0]) for p in pl.group(1).split()]
            solid_max = max(solid_max, max(xs))
        cap = re.search(r"(Measured[^<]{0,400}|Illustrative[^<]{0,400}|Projected[^<]{0,400})", html.unescape(t[m.end():m.end() + 3000]))
        caption = cap.group(1).strip() if cap else ""
        rig = re.search(r"on ([0-9]+×H100|H100s)", caption)
        illustrative = caption.startswith(("Illustrative", "Projected"))
        slug = re.sub(r"[^a-z0-9]+", "-", f"{src}-{tgt}".lower()).strip("-")
        for ctx, nat, ot in re.findall(r"<th[^>]*>([0-9]+K)</th><td>([0-9.]+) s</td><td>([0-9.]+) s</td>", tab):
            k = float(ctx[:-1])
            measured = (not illustrative) and k <= solid_max + 1e-6
            base = dict(page="https://onetriangle.ai/", pair=f"{src} -> {tgt}", context=ctx,
                        their_rig=(rig.group(1) if rig else "H100") + ", HF forwards", measured=measured, caption=caption[:300])
            rows.append(dict(base, id=f"home-{slug}-{ctx}-ttft", metric="TTFT s (mapped)", their_value=float(ot),
                             direction="lower", their_baseline_native_s=float(nat)))
            rows.append(dict(base, id=f"home-{slug}-{ctx}-speedup", metric="speedup x vs their native",
                             their_value=round(float(nat) / float(ot), 3), direction="higher"))
        if peak:
            rows.append(dict(id=f"home-{slug}-peak", page="https://onetriangle.ai/", pair=f"{src} -> {tgt}",
                             context=peak_ctx[-1][0] if peak_ctx else "?", metric="peak speedup x (headline)",
                             their_value=float(peak[-1]), direction="higher",
                             measured=bool(peak_ctx and peak_ctx[-1][1] == "measured"),
                             their_rig=(rig.group(1) if rig else "H100") + ", HF forwards", caption=caption[:300]))
    return rows


def parse_bundle(path):
    b = json.load(open(path))
    pair = f"{b['pair']['source']} -> {b['pair']['mapped_receiver']}"
    rows = []
    for sec in ("api_and_coding", "agent_benchmarks"):
        for r in b.get(sec, []):
            slug = re.sub(r"[^a-z0-9]+", "-", r["benchmark"].lower()).strip("-")
            rows.append(dict(id=f"bundle-{slug}-mapped", page="https://onetriangle.ai/benchmarks/qwen-27/2026-11-13/",
                             pair=pair, metric="score (mapped)", context=r["benchmark"], their_value=r.get("mapped"),
                             n=r.get("mapped_n"), direction="higher", measured=True, their_rig="their endpoint",
                             their_native_target=r.get("receiver"), their_native_source=r.get("source", r.get("source_proxy")),
                             note=r.get("note", "")))
    return rows


def best_seat_entry(row, cands):
    """Several seats measured the same row: keep the better value (row direction), preferring entries whose N meets
    theirs; blocked/hold entries only when no seat has a plain measurement. First seat wins ties (old behaviour)."""
    if len(cands) == 1 or row is None or row.get("direction") not in ("higher", "lower"):
        return cands[0]
    plain = [c for c in cands if c.get("value") is not None and not c.get("blocked") and c.get("quality_ok") is not False]
    if not plain:
        return cands[0]
    n_ok = [c for c in plain if not (row.get("n") and c.get("n") and c["n"] < row["n"])] or plain
    sign = 1 if row["direction"] == "higher" else -1
    return max(n_ok, key=lambda c: (sign * c["value"], -n_ok.index(c)))


def status(row, ours):
    if row.get("direction") == "context":
        return "context"
    o = ours.get(row["id"])
    if not o:
        return row.get("status_default", "open")
    if o.get("blocked"):
        return f"blocked({o['blocked']})"
    if o.get("quality_ok") is False:  # timing measured, but the same config failed its quality check: not a win
        return "hold(quality)"
    g = o.get("gpu_measured")
    if g and not o.get("rig_exempt") and "H100" in str(row.get("their_rig", "")) and ("H100" not in g or "H200" in g or "MIXED" in g.upper()):
        return "hold(rig)"  # measured on another GPU class (e.g. Modal upgraded an H100 request to H200): rerun on H100
    if o.get("value") is None:
        return "open"
    if o.get("forecast"):  # extrapolated from measured points, not a measurement: never counts as beaten
        return "behind(forecast)" if (o["value"] < row["their_value"]) == (row.get("direction", "higher") == "higher") else "open(forecast)"
    if row.get("n") and o.get("n") and o["n"] < row["n"]:
        return "behind(N)"
    v, t = o["value"], row["their_value"]
    if t is None:
        return "open"
    better = v < t if row["direction"] == "lower" else v > t
    # their published value is rounded: a difference inside half its last printed digit is a tie, not a loss
    sv = repr(float(t)).rstrip("0")
    dec = len(sv.split(".")[1]) if "." in sv else 0
    tie = abs(v - t) <= 0.5 * 10 ** (-dec) + 1e-12 or (o.get("ci") and abs(v - t) <= o["ci"])
    return "tied" if tie else ("beaten" if better else "behind")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ot", default=os.path.expanduser("~/kvx-data/ot"))
    ap.add_argument("--ours", default=os.path.expanduser("~/kvx-data/ours.json"))
    ap.add_argument("--out", default=os.path.expanduser("~/kvx-data/scoreboard.json"))
    ap.add_argument("--readme", default="")
    a = ap.parse_args()
    rows = parse_home(f"{a.ot}/home.html") + parse_bundle(f"{a.ot}/bundle/latest-benchmark-bundle.json")
    ext = f"{a.ot}/metrics_extracted.json"
    if os.path.exists(ext):
        seen = {r["id"] for r in rows}
        for r in json.load(open(ext)):
            arm = (r.get("arm") or "mapped")
            if not (arm.startswith("mapped") or arm.startswith("diagnostic")) or arm == "placeholder":
                continue  # their own transfer results are targets; baselines and rival methods are context
            rid = "x-" + r["id"]
            if rid in seen:
                continue
            seen.add(rid)
            mt = (str(r.get("metric", "")) + " " + str(r.get("unit", ""))).lower()
            hib = not re.search(r"ttft|latency|time|loop|\bkl\b|kl32|nll|error|gap|cost|lower better|\(ms\)|\bms\b|\bs\b|seconds", mt)
            for pat, dirn in DIRECTION_OVERRIDES:
                if re.search(pat, mt):
                    hib = dirn
                    break
            rows.append(dict(id=rid, page=r.get("page"), pair=r.get("pair"), metric=f"{r.get('metric')} [{arm}]",
                             context=r.get("benchmark_or_context"), their_value=r.get("their_value"), n=r.get("n"),
                             direction="context" if hib == "context" else ("higher" if hib else "lower"),
                             measured=r.get("measured"), their_rig=r.get("rig"),
                             note=r.get("quote", "")[:200]))
    for r in rows:  # one spelling per pair across sources: "Qwen3 1.7B -> Qwen3 8B" == "Qwen3-1.7B → Qwen3-8B"
        p = str(r.get("pair") or "?").replace("→", "->")
        p = re.sub(r"\s*->\s*", " -> ", p)
        r["pair"] = " -> ".join(re.sub(r"[\s_]+", "-", side.strip()) for side in p.split(" -> "))
    ours = json.load(open(a.ours)) if os.path.exists(a.ours) else {}
    # each seat reports its rows in <data>/seats/<seat>/ours.json (same keys); the lead's own file wins on conflicts,
    # and between seats the better measurement wins (by the row's direction, among entries with N >= theirs)
    import glob
    cands = {}
    for f in sorted(glob.glob(os.path.join(os.path.dirname(a.ours), "seats", "*", "ours.json"))):
        for k, v in json.load(open(f)).items():
            cands.setdefault(k, []).append(dict(v, seat=f.split(os.sep)[-2]))
    byid = {r["id"]: r for r in rows}
    for k, cs in cands.items():
        if k not in ours:
            ours[k] = best_seat_entry(byid.get(k), cs)
    for r in rows:
        r["ours"] = ours.get(r["id"])
        r["status"] = status(r, ours)
    for r in rows:
        r["tier"] = tier(r)
    t1 = [r for r in rows if r["tier"] == 1]
    t1c = {}
    for r in t1:
        k = r["status"].split("(")[0]
        t1c[k] = t1c.get(k, 0) + 1
    counts = {}
    for r in rows:
        counts[r["status"].split("(")[0]] = counts.get(r["status"].split("(")[0], 0) + 1
    out = dict(total=len(rows), counts=counts, tier1=dict(total=len(t1), counts=t1c), rows=rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(dict(total=len(rows), counts=counts, tier1=out["tier1"])))
    if a.readme:
        write_readme(a.readme, out)


OWNERS = {}  # explicit owner overrides by row id


def owner_default(r):
    if r["id"].startswith("bundle-"):
        return "quality"
    if r["id"].endswith("-peak") or str(r.get("metric", "")).startswith("TTFT"):
        return "speed"
    return "-"


def tier(r):
    """The user, 2026-11-22: only Tier 1 matters. 1 = the 45 headline rows (9 peak speedups, their measured end-to-end TTFT
    points, the 15 bundle rows); 3 = untracked (components, derived speedups, context, diagnostics, hand ratings); 2 = rest."""
    m = str(r.get("metric") or "")
    if r["id"].endswith("-peak") or r["id"].startswith("bundle-"):
        return 1
    if m.startswith("TTFT s") and r.get("measured") and r.get("status") != "context":
        return 1
    if (r.get("status") == "context" or "component" in m.lower() or r["id"].endswith("-speedup") or "diagnostic" in m
            or "hand" in m.lower() or "rating" in m.lower()):
        return 3
    return 2


def category(r):
    m = (r.get("metric") or "").lower()
    if r["id"].startswith("home-"):
        return "TTFT table"
    if "component" in m:
        return "TTFT component"
    if "ttft" in m or "speedup" in m or "latency" in m or "throughput" in m or "cost" in m:
        return "speed/cost"
    if "diagnostic" in m:
        return "diagnostic"
    return "quality"


def write_readme(path, sb):
    rows = sb["rows"]
    full = os.path.join(os.path.dirname(os.path.abspath(path)), "docs", "SCOREBOARD.md")
    os.makedirs(os.path.dirname(full), exist_ok=True)
    lines = ["# Scoreboard: every OneTriangle published number vs ours", "",
             f"{sb['total']} rows; generated by `tools/scoreboard.py`. 'measured' is THEIR label (est. = their projection).", "",
             "| id | pair | metric | context / benchmark | theirs | their measured? | ours | status |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        o = r.get("ours") or {}
        cell = lambda x: str(x).replace("|", "/").replace("\n", " ")[:90]
        lines.append(f"| {cell(r['id'])} | {cell(r['pair'])} | {cell(r['metric'])} | {cell(r.get('context'))} | {r['their_value']} | "
                     f"{'yes' if r.get('measured') else 'est.' if r.get('measured') is False else '?'} | {o.get('value', '')} | {r['status']} |")
    open(full, "w").write("\n".join(lines) + "\n")
    agg = {}
    for r in rows:
        k = (r["pair"] or "?", category(r))
        a = agg.setdefault(k, {"rows": 0, "beaten": 0, "tied": 0, "behind": 0, "blocked": 0, "open": 0})
        a["rows"] += 1
        a[r["status"].split("(")[0]] = a.get(r["status"].split("(")[0], 0) + 1
    c = sb["counts"]
    t1 = sb.get("tier1", {"total": 0, "counts": {}})
    t1c = t1["counts"]
    tier1_lines = [f"**OneTriangle Tier 1: {t1c.get('beaten', 0)} / {t1['total']} beaten** (tied {t1c.get('tied', 0)}, behind "
                   f"{t1c.get('behind', 0)}, on quality hold {t1c.get('hold', 0)}, blocked {t1c.get('blocked', 0)}, open {t1c.get('open', 0)}): the 9 homepage peak "
                   "speedups, their measured end-to-end TTFT points, and the 15 rows of their benchmark bundle. Only Tier 1 is the goal.", "",
                   "| id | pair | metric | context / benchmark | theirs | ours | owner | status |", "|---|---|---|---|---|---|---|---|"]
    for r in [x for x in rows if x.get("tier") == 1]:
        o = r.get("ours") or {}
        tier1_lines.append(f"| {r['id']} | {r['pair']} | {r['metric']} | {r.get('context')} | {r['their_value']} | "
                           f"{o.get('value', '')} | {OWNERS.get(r['id'], owner_default(r))} | {r['status'].split('(')[0]} |")
    out = tier1_lines + ["", "All rows (Tier 2 and untracked included):", "", f"**Beaten {c.get('beaten', 0)} of {sb['total']}** published OneTriangle numbers (tied {c.get('tied', 0)}, behind "
           f"{c.get('behind', 0)}, blocked {c.get('blocked', 0)}, open {c.get('open', 0)}). Full row table: "
           "[docs/SCOREBOARD.md](docs/SCOREBOARD.md). Rules: same pair, context/benchmark and N or more; TTFT on their GPU class "
           "(H100) and speedups against native vLLM/SGLang.", "",
           "| pair | category | rows | beaten | tied | behind | blocked | open |", "|---|---|---|---|---|---|---|---|"]
    for (pair, cat), a in sorted(agg.items(), key=lambda kv: (-kv[1]["rows"], kv[0])):
        out.append(f"| {pair} | {cat} | {a['rows']} | {a['beaten']} | {a['tied']} | {a['behind']} | {a['blocked']} | {a['open']} |")
    block = "\n".join(out)
    s = open(path).read()
    a, b = "<!-- SCOREBOARD:BEGIN -->", "<!-- SCOREBOARD:END -->"
    if a in s:
        s = s[:s.index(a) + len(a)] + "\n" + block + "\n" + s[s.index(b):]
    else:
        s = s.replace("## What the target claims", f"## Scoreboard: every OneTriangle number vs ours\n\n{a}\n{block}\n{b}\n\n## What the target claims", 1)
    open(path, "w").write(s)


if __name__ == "__main__":
    main()
