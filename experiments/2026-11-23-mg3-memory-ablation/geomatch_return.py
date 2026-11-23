"""SIFT+RANSAC inlier count between start and return frames (place
recognition), for MG2 and MG3 memory on/off, plus an unrelated-scene
reference (start frames of different images). Writes results/geomatch.json."""
import json, sys, itertools, collections
from pathlib import Path
import numpy as np, imageio.v2 as iio
HERE = Path(__file__).parent; ROOT = HERE.parents[1]; LEAD = 8
sys.path.insert(0, str(ROOT / "src"))
from wm.geomatch import inliers  # noqa: E402
def frames(p, idx): return {j: f for j, f in enumerate(iio.get_reader(str(p))) if j in idx}
specs = [("MG2", "on", ROOT / "experiments/2026-11-23-mg2-memory-horizon/results/videos", ["0013", "0003", "0009", "0016"], [12, 24, 48, 96, 192], [0, 1, 2], "{img}_T{T}_s{s}_outback.mp4"),
         ("MG3", "on", HERE / "results/videos", ["001", "004", "007", "010"], [24, 48, 96, 120], [0, 1], "{img}_T{T}_s{s}_mem1.mp4"),
         ("MG3", "off", HERE / "results/videos", ["001", "004", "007", "010"], [24, 48, 96, 120], [0, 1], "{img}_T{T}_s{s}_mem0.mp4")]
rows, starts = [], collections.defaultdict(dict)
for model, mem, V, imgs, Ts, seeds, pat in specs:
    for img in imgs:
        for T in Ts:
            for s in seeds:
                p = V / pat.format(img=img, T=T, s=s)
                if not p.exists(): continue
                fr = frames(p, {LEAD, LEAD + 2 * T})
                starts[model][img] = fr[LEAD]
                rows.append(dict(model=model, memory=mem, image=img, T=T, seed=s, inliers=inliers(fr[LEAD], fr[LEAD + 2 * T])))
refs = {m: [inliers(d[a], d[b]) for a, b in itertools.permutations(d, 2)] for m, d in starts.items()}
agg = collections.defaultdict(list)
for r in rows: agg[f'{r["model"]}|{r["memory"]}|{r["T"]}'].append(r["inliers"])
out = dict(rows=rows, unrelated=refs, median={k: float(np.median(v)) for k, v in agg.items()},
           mean={k: float(np.mean(v)) for k, v in agg.items()},
           frac_recognised={k: float(np.mean(np.array(v) >= 30)) for k, v in agg.items()})
(HERE / "results/geomatch.json").write_text(json.dumps(out, indent=0))
for k in sorted(agg): print(k, "median", out["median"][k], "recognised(>=30)", round(out["frac_recognised"][k], 2), sorted(agg[k]))
print("unrelated", {m: (float(np.median(v)), max(v)) for m, v in refs.items()})
