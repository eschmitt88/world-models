"""Per-frame quality traces for the long rollouts and their drift summaries.

For every 4th frame: sharpness (variance of the Laplacian, grey, 640x352),
colourfulness (Hasler & Suesstrunk 2003), contrast (grey std), brightness,
and 1-second temporal change (LPIPS between frame t and t-25). Each trace is
normalised by its own first-2-seconds mean, so runs from different start
images can be pooled. Writes results/traces.json, results/summary.json and
metrics.json."""
import json
from pathlib import Path
import numpy as np, cv2, torch, lpips, imageio.v2 as iio, yaml

HERE = Path(__file__).parent; R = HERE / "results"
cfg = yaml.safe_load((HERE / "config.yaml").read_text())
runs = [json.loads(l) for l in (R / "runs.jsonl").read_text().splitlines()]
dev = "cuda" if torch.cuda.is_available() else "cpu"
lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
STEP, LAG, FPS = 4, 25, 25

def colourfulness(f):
    r, g, b = [f[..., i].astype(np.float32) for i in range(3)]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

def T_(f):
    t = torch.from_numpy(np.ascontiguousarray(f)).to(dev).permute(2, 0, 1)[None].float() / 127.5 - 1
    return torch.nn.functional.interpolate(t, scale_factor=0.5, mode="area")

traces = {}
with torch.no_grad():
    for r in runs:
        key = f'{r["image"]}_s{r["seed"]}_cn{r["context_noise"]}'
        fr = [f for f in iio.get_reader(str(HERE / r["video"]))]
        idx = list(range(0, len(fr), STEP))
        tr = dict(t=[i / FPS for i in idx], sharp=[], colour=[], contrast=[], bright=[], change=[])
        for i in idx:
            g = cv2.cvtColor(fr[i], cv2.COLOR_RGB2GRAY)
            tr["sharp"].append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
            tr["colour"].append(colourfulness(fr[i])); tr["contrast"].append(float(g.std())); tr["bright"].append(float(g.mean()))
            tr["change"].append(float(lp(T_(fr[i]), T_(fr[i - LAG]))) if i >= LAG else None)
        traces[key] = dict(image=r["image"], seed=r["seed"], cn=r["context_noise"], **tr)
        print(key, "done", flush=True)
(R / "traces.json").write_text(json.dumps(traces))

def norm(x, t, t0=2.0):
    x = np.array([np.nan if v is None else v for v in x], float); t = np.array(t)
    base = np.nanmean(x[t <= t0]) if np.any(~np.isnan(x[t <= t0])) else np.nanmean(x[:20])
    return x / base

summary = {}
for cn in cfg["context_noise"]:
    ks = [k for k, v in traces.items() if v["cn"] == cn]
    if not ks: continue
    out = {}
    for m in ("sharp", "colour", "contrast"):
        late = [np.nanmean(norm(traces[k][m], traces[k]["t"])[np.array(traces[k]["t"]) >= 80]) for k in ks]
        out[f"{m}_late_ratio"] = float(np.mean(late))
    ch = [np.nanmean(np.array([np.nan if v is None else v for v in traces[k]["change"]], float)[np.array(traces[k]["t"]) >= 80]) for k in ks]
    out["change_late"] = float(np.mean(ch))
    early = [np.nanmean(np.array([np.nan if v is None else v for v in traces[k]["change"]], float)[(np.array(traces[k]["t"]) >= 1) & (np.array(traces[k]["t"]) <= 10)]) for k in ks]
    out["change_early"] = float(np.mean(early))
    # darkness collapse: share of sampled frames with mean grey < 20 (of 255)
    out["dark_frac"] = float(np.mean([np.mean(np.array(traces[k]["bright"]) < 20) for k in ks]))
    out["dark_frac_late"] = float(np.mean([np.mean(np.array(traces[k]["bright"])[np.array(traces[k]["t"]) >= 48] < 20) for k in ks]))
    out["per_run_dark_frac"] = {k: float(np.mean(np.array(traces[k]["bright"]) < 20)) for k in ks}
    out["n"] = len(ks)
    summary[str(cn)] = out
(R / "summary.json").write_text(json.dumps(summary, indent=1))
metrics = {f"cn{cn}_{k}": v for cn, d in summary.items() for k, v in d.items() if k not in ("n", "per_run_dark_frac")}
(HERE / "metrics.json").write_text(json.dumps(metrics, indent=1))
print(json.dumps(summary, indent=1))
