"""Shift-aligned return-vs-start LPIPS for MG2 (memory-horizon videos) and MG3
(this experiment), plus the same metric between seeds' return frames.
Writes results/aligned.json."""
import json, sys, itertools
from pathlib import Path
import numpy as np, imageio.v2 as iio, lpips, torch
HERE = Path(__file__).parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wm.aligned import aligned_lpips  # noqa: E402
dev = "cuda" if torch.cuda.is_available() else "cpu"
lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
LEAD = 8
def frames(p, idx):
    out = {}
    for j, f in enumerate(iio.get_reader(str(p))):
        if j in idx: out[j] = f
    return out
res = []
specs = [("MG2", "on", ROOT / "experiments/2026-11-23-mg2-memory-horizon/results/videos", ["0013", "0003", "0009", "0016"], [12, 24, 48, 96, 192], [0, 1, 2], "{img}_T{T}_s{s}_outback.mp4"),
         ("MG3", "on", HERE / "results/videos", ["001", "004", "007", "010"], [24, 48, 96, 120], [0, 1], "{img}_T{T}_s{s}_mem1.mp4"),
         ("MG3", "off", HERE / "results/videos", ["001", "004", "007", "010"], [24, 48, 96, 120], [0, 1], "{img}_T{T}_s{s}_mem0.mp4")]
for model, mem, V, imgs, Ts, seeds, pat in specs:
    for img in imgs:
        for T in Ts:
            rets = {}
            for s in seeds:
                p = V / pat.format(img=img, T=T, s=s)
                if not p.exists(): continue
                fr = frames(p, {LEAD, LEAD + 2 * T})
                v, info = aligned_lpips(lp, fr[LEAD], fr[LEAD + 2 * T], dev)
                rets[s] = fr[LEAD + 2 * T]
                res.append(dict(model=model, memory=mem, image=img, T=T, seed=s, kind="return_vs_start", lpips=v, **info))
            for a, b in itertools.combinations(sorted(rets), 2):
                v, info = aligned_lpips(lp, rets[a], rets[b], dev)
                res.append(dict(model=model, memory=mem, image=img, T=T, seed=f"{a}-{b}", kind="seed_vs_seed", lpips=v, **info))
    print(model, mem, "done", flush=True)
(HERE / "results/aligned.json").write_text(json.dumps(res, indent=0))
import collections
agg = collections.defaultdict(list)
for r in res: agg[(r["model"], r["memory"], r["kind"], r["T"])].append(r["lpips"])
for k in sorted(agg): print(k, round(float(np.mean(agg[k])), 3), len(agg[k]))
