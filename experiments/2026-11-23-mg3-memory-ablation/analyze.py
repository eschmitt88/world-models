"""Aggregate raw.jsonl into metrics.json and summary.json (same protocol and
metric as experiments/2026-11-23-mg2-memory-horizon/analyze.py, split by
memory on/off). Videos are stored at 640x352, the LPIPS resolution basis."""
import json, itertools
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, yaml, imageio.v2 as iio, lpips

HERE = Path(__file__).parent; R = HERE / "results"
cfg = yaml.safe_load((HERE / "config.yaml").read_text())
rows = [json.loads(l) for l in (R / "raw.jsonl").read_text().splitlines()]
rng = np.random.default_rng(0)
def ci(x):
    x = np.asarray(x); bs = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

bins = [0, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
curve, ret = {}, {}
for mem in (True, False):
    pts = defaultdict(list)
    for r in rows:
        if r["memory"] != mem: continue
        for g, v in zip(r["gap"], r["lpips_mirror"]):
            pts[max(b for b in bins if b <= g)].append(v)
    curve[str(mem)] = [dict(gap=g, n=len(v), mean=ci(v)[0], lo=ci(v)[1], hi=ci(v)[2]) for g, v in sorted(pts.items())]
    ret[str(mem)] = []
    for T in cfg["T"]:
        v = [r["lpips_return_vs_lead"] for r in rows if r["memory"] == mem and r["T"] == T]
        if v: m, lo, hi = ci(v); ret[str(mem)].append(dict(T=T, round_trip=2 * T, n=len(v), mean=m, lo=lo, hi=hi))
per_image = defaultdict(dict)
for r in rows:
    per_image[f'{r["image"]}_mem{int(r["memory"])}'].setdefault(r["T"], []).append(r["lpips_return_vs_lead"])
per_image = {k: {T: float(np.mean(v)) for T, v in d.items()} for k, d in per_image.items()}

dev = "cuda" if torch.cuda.is_available() else "cpu"
lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
def T_(f):
    t = torch.from_numpy(np.ascontiguousarray(f)).to(dev).permute(2, 0, 1)[None].float() / 127.5 - 1
    return torch.nn.functional.interpolate(t, scale_factor=0.5, mode="area")
def frame(p, i):
    for j, f in enumerate(iio.get_reader(str(p))):
        if j == i: return f
V = R / "videos"; refs = {}
Tm = max(cfg["T"])
with torch.no_grad():
    f0 = {n: frame(V / f"{n}_T{cfg['T'][0]}_s0_mem1.mp4", 0) for n in cfg["images"] if (V / f"{n}_T{cfg['T'][0]}_s0_mem1.mp4").exists()}
    refs["unrelated_scenes"] = ci([float(lp(T_(f0[a]), T_(f0[b]))) for a, b in itertools.combinations(f0, 2)]) if len(f0) > 1 else None
    for mem in (1, 0):
        sp = []
        for n in cfg["images"]:
            fs = [frame(V / f"{n}_T{Tm}_s{s}_mem{mem}.mp4", cfg["lead"] + 2 * Tm) for s in cfg["seeds"] if (V / f"{n}_T{Tm}_s{s}_mem{mem}.mp4").exists()]
            sp += [float(lp(T_(a), T_(b))) for a, b in itertools.combinations(fs, 2)]
        refs[f"seed_spread_at_return_Tmax_mem{mem}"] = ci(sp) if sp else None
refs["Tmax"] = Tm
(R / "summary.json").write_text(json.dumps(dict(curve=curve, return_=ret, per_image=per_image, refs=refs, n_rows=len(rows)), indent=1))
metrics = {"n_rollouts": len(rows)}
for mem in ("True", "False"):
    for x in ret[mem]:
        metrics[f"return_lpips_mem{'on' if mem == 'True' else 'off'}_T{x['T']}"] = x["mean"]
for k, v in refs.items():
    if isinstance(v, (list, tuple)): metrics[f"ref_{k}"] = v[0]
(HERE / "metrics.json").write_text(json.dumps(metrics, indent=1))
print(json.dumps(metrics, indent=1))
