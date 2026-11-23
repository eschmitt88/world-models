"""Coarse-layout distance between the return frame and the start frame:
mean CIE76 Delta-E between 32x18 area-downsampled Lab thumbnails. This keeps
where the big masses of colour sit (wall, sky, road) and ignores the texture
detail a world model re-synthesises each time it renders a place. Reference:
start frames of different images (unrelated scenes). Writes results/layout.json."""
import json, itertools, collections
from pathlib import Path
import numpy as np, cv2, imageio.v2 as iio
HERE = Path(__file__).parent; ROOT = HERE.parents[1]; LEAD = 8
def thumb(f):
    return cv2.cvtColor(cv2.resize(f, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
def dE(a, b): return float(np.sqrt(((thumb(a) - thumb(b)) ** 2).sum(-1)).mean())
def frames(p, idx):
    return {j: f for j, f in enumerate(iio.get_reader(str(p))) if j in idx}
specs = [("MG2", "on", ROOT / "experiments/2026-11-23-mg2-memory-horizon/results/videos", ["0013", "0003", "0009", "0016"], [12, 24, 48, 96, 192], [0, 1, 2], "{img}_T{T}_s{s}_outback.mp4"),
         ("MG3", "on", HERE / "results/videos", ["001", "004", "007", "010"], [24, 48, 96, 120], [0, 1], "{img}_T{T}_s{s}_mem1.mp4"),
         ("MG3", "off", HERE / "results/videos", ["001", "004", "007", "010"], [24, 48, 96, 120], [0, 1], "{img}_T{T}_s{s}_mem0.mp4")]
res, starts = [], collections.defaultdict(dict)
for model, mem, V, imgs, Ts, seeds, pat in specs:
    for img in imgs:
        for T in Ts:
            for s in seeds:
                p = V / pat.format(img=img, T=T, s=s)
                if not p.exists(): continue
                fr = frames(p, {LEAD, LEAD + 2 * T})
                starts[model][img] = fr[LEAD]
                res.append(dict(model=model, memory=mem, image=img, T=T, seed=s, dE=dE(fr[LEAD], fr[LEAD + 2 * T])))
refs = {m: float(np.mean([dE(d[a], d[b]) for a, b in itertools.combinations(d, 2)])) for m, d in starts.items()}
agg = collections.defaultdict(list)
for r in res: agg[f'{r["model"]}|{r["memory"]}|{r["T"]}'].append(r["dE"])
summary = dict(rows=res, unrelated_scenes=refs, mean={k: float(np.mean(v)) for k, v in agg.items()},
               sem={k: float(np.std(v) / np.sqrt(len(v))) for k, v in agg.items()})
(HERE / "results/layout.json").write_text(json.dumps(summary, indent=0))
for k in sorted(summary["mean"]): print(k, round(summary["mean"][k], 2), "+-", round(summary["sem"][k], 2))
print("unrelated", {k: round(v, 2) for k, v in refs.items()})
