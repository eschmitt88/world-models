"""Aggregate raw.jsonl into metrics.json, summary.json and figures.

Memory curve: LPIPS between mirror frames (t_peak-k, t_peak+k) vs gap 2k,
pooled over images/seeds/T. Reference levels: same-view consecutive frames
(floor), different seeds' return frames at the longest T (sample spread), and
frame 0 of different images (unrelated scenes)."""
import json, itertools
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, yaml, imageio.v2 as iio

HERE = Path(__file__).parent; R = HERE / "results"
cfg = yaml.safe_load((HERE / "config.yaml").read_text())
rows = [json.loads(l) for l in (R / "raw.jsonl").read_text().splitlines()]
rng = np.random.default_rng(0)

def ci(x):
    x = np.asarray(x); bs = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

# memory curve: bin by gap
bins = [0, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
curve = {}
for cond in ("outback", "still"):
    pts = defaultdict(list)
    for r in rows:
        if r["cond"] != cond: continue
        for g, v in zip(r["gap"], r["lpips_mirror"]):
            b = max(i for i in range(len(bins)) if bins[i] <= g)
            pts[bins[b]].append(v)
    curve[cond] = [dict(gap=g, n=len(v), mean=ci(v)[0], lo=ci(v)[1], hi=ci(v)[2]) for g, v in sorted(pts.items())]

ret = {}
for cond in ("outback", "still"):
    ret[cond] = []
    for T in cfg["T"]:
        v = [r["lpips_return_vs_lead"] for r in rows if r["cond"] == cond and r["T"] == T]
        if v: m, lo, hi = ci(v); ret[cond].append(dict(T=T, round_trip=2 * T, n=len(v), mean=m, lo=lo, hi=hi))
per_image = defaultdict(dict)
for r in rows:
    if r["cond"] == "outback":
        per_image[r["image"]].setdefault(r["T"], []).append(r["lpips_return_vs_lead"])
per_image = {k: {T: float(np.mean(v)) for T, v in d.items()} for k, d in per_image.items()}

# reference levels from videos
import lpips
lp = lpips.LPIPS(net="alex", verbose=False).cuda().eval() if torch.cuda.is_available() else lpips.LPIPS(net="alex", verbose=False).eval()
dev = "cuda" if torch.cuda.is_available() else "cpu"
def T_(f):
    t = torch.from_numpy(np.ascontiguousarray(f)).to(dev).permute(2, 0, 1)[None].float() / 127.5 - 1
    return torch.nn.functional.interpolate(t, scale_factor=0.5, mode="area")
def frame(p, i):
    rd = iio.get_reader(str(p)); 
    for j, f in enumerate(rd):
        if j == i: rd.close(); return f
    rd.close(); return f
V = R / "videos"
refs = {}
with torch.no_grad():
    f0 = {n: frame(V / f"{n}_T{cfg['T'][0]}_s0_still.mp4", 0) for n in cfg["images"] if (V / f"{n}_T{cfg['T'][0]}_s0_still.mp4").exists()}
    refs["unrelated_scenes"] = ci([float(lp(T_(f0[a]), T_(f0[b]))) for a, b in itertools.combinations(f0, 2)]) if len(f0) > 1 else None
    Tm = max(r["T"] for r in rows if r["cond"] == "outback")
    spread = []
    for n in cfg["images"]:
        fs = [frame(V / f"{n}_T{Tm}_s{s}_outback.mp4", cfg["lead"] + 2 * Tm) for s in cfg["seeds"] if (V / f"{n}_T{Tm}_s{s}_outback.mp4").exists()]
        spread += [float(lp(T_(a), T_(b))) for a, b in itertools.combinations(fs, 2)]
    refs["seed_spread_at_return_Tmax"] = ci(spread) if spread else None
    refs["Tmax"] = Tm

window_frames = 4 * 6 - 3  # 6 latent frames ~ 21-24 video frames
summary = dict(curve=curve, return_=ret, per_image=per_image, refs=refs, window_frames=window_frames, n_rows=len(rows))
(R / "summary.json").write_text(json.dumps(summary, indent=1))
g = lambda cond, T: next((x["mean"] for x in ret[cond] if x["T"] == T), None)
metrics = {"n_rollouts": len(rows)}
for T in cfg["T"]:
    metrics[f"return_lpips_outback_T{T}"] = g("outback", T); metrics[f"return_lpips_still_T{T}"] = g("still", T)
metrics["ref_unrelated_scenes_lpips"] = refs["unrelated_scenes"][0] if refs["unrelated_scenes"] else None
metrics["ref_seed_spread_lpips"] = refs["seed_spread_at_return_Tmax"][0] if refs["seed_spread_at_return_Tmax"] else None
(HERE / "metrics.json").write_text(json.dumps(metrics, indent=1))
print(json.dumps(metrics, indent=1))
