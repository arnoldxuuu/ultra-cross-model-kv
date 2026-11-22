"""Run any Modal entry function on a rented box (Vast / EC2) without Modal.

  python run_local.py MODULE:FUNC --json '{"arg": value, ...}'      e.g. modal_vllm:serve_impl, modal_ttft:ttft
  python run_local.py --src ... --tgt ...                           (old form: modal_fit_eval.fit_eval_impl flags)

The `modal` package is not needed: the modules fall back to a no-op stand-in whose decorators return the function.
Results go where the function writes them (/vol/runs/...); the return value is printed as JSON and saved to
/vol/runs/local/<module>-<func>-<UTC>.json. Weights: kvx.models.model_path falls back to Hub ids when /hf/hub is absent.
"""
import argparse, importlib, inspect, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("/vol/runs/local", exist_ok=True)

if len(sys.argv) > 1 and ":" in sys.argv[1] and not sys.argv[1].startswith("-"):
    mod_name, fn_name = sys.argv[1].split(":", 1)
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--json", default="{}")
    a = ap.parse_args()
    fn = getattr(importlib.import_module(mod_name), fn_name)
    fn = getattr(fn, "local", None) or getattr(fn, "__wrapped__", None) or fn
    kwargs = json.loads(a.json)
else:  # old form
    import modal_fit_eval as m
    mod_name, fn_name, fn = "modal_fit_eval", "fit_eval_impl", m.fit_eval_impl
    sig = inspect.signature(fn)
    ap = argparse.ArgumentParser()
    for name, p in sig.parameters.items():
        if p.default is inspect.Parameter.empty:
            ap.add_argument("--" + name.replace("_", "-"), dest=name, type=str, required=True)
        else:
            ap.add_argument("--" + name.replace("_", "-"), dest=name, type=type(p.default), default=p.default)
    kwargs = vars(ap.parse_args())
t0 = time.time()
r = fn(**kwargs)
out = r if isinstance(r, dict) else {"result": r}
out["_local"] = {"target": f"{mod_name}:{fn_name}", "args": kwargs, "wall_s": round(time.time() - t0, 1)}
p = f"/vol/runs/local/{mod_name}-{fn_name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
json.dump(out, open(p, "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str)[:20000])
print("saved", p)
