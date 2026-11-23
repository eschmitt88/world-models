"""Out-and-back memory test for Matrix-Game 2.0.

For each (image, seed, T): hold still for LEAD frames, yaw right at +YAW for
T frames, yaw left at -YAW for T frames, then hold. Frames t_peak-k and
t_peak+k show the same camera heading. Control: the same length with no
action. Writes results/raw.jsonl (one row per rollout, with per-k LPIPS) and
full videos to results/videos/ (gitignored).
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch, yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from wm.mg2 import MG2, MG2_DIR, write_mp4  # noqa: E402

cfg = yaml.safe_load((HERE / "config.yaml").read_text())
out = HERE / "results"; (out / "videos").mkdir(parents=True, exist_ok=True)
raw = out / "raw.jsonl"
done = set()
if raw.exists():
    for line in raw.read_text().splitlines():
        r = json.loads(line); done.add((r["image"], r["seed"], r["T"], r["cond"]))

import lpips  # noqa: E402
lp = lpips.LPIPS(net="alex", verbose=False).cuda().eval()


def to_t(frames):  # uint8 [N,H,W,3] -> [-1,1] tensor at half res
    t = torch.from_numpy(frames).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1
    return torch.nn.functional.interpolate(t, scale_factor=0.5, mode="area")


@torch.no_grad()
def lpips_pairs(v, a_idx, b_idx):
    A, B = to_t(v[a_idx]), to_t(v[b_idx])
    return lp(A, B).flatten().tolist()


m = MG2()
LEAD, YAW = cfg["lead"], cfg["yaw"]
for name in cfg["images"]:
    img = MG2_DIR / f"demo_images/universal/{name}.png"
    for T in cfg["T"]:
        n = LEAD + 2 * T + 1
        L = (n - 1 + 3) // 4 + 1; L = int(np.ceil(L / 3) * 3); F = 4 * (L - 1) + 1
        for seed in cfg["seeds"]:
            for cond in ("outback", "still"):
                if (name, seed, T, cond) in done:
                    continue
                kb = np.zeros((F, 4)); ms = np.zeros((F, 2))
                if cond == "outback":
                    ms[LEAD:LEAD + T, 1] = YAW
                    ms[LEAD + T:LEAD + 2 * T, 1] = -YAW
                t0 = time.time()
                v = m.rollout(img, kb, ms, seed=seed, context_noise=cfg["context_noise"])
                dt = time.time() - t0
                tp = LEAD + T
                ks = list(range(0, T + 1, max(1, T // 24)))
                if ks[-1] != T: ks.append(T)
                # mirror pairs, allowing +-2 frame misalignment on the return side
                best = []
                for k in ks:
                    cands = [tp + k + d for d in range(-2, 3) if 0 <= tp + k + d < len(v)]
                    vals = lpips_pairs(v, [tp - k] * len(cands), cands)
                    best.append(min(vals))
                ret = lpips_pairs(v, [LEAD], [LEAD + 2 * T])[0]
                ret0 = lpips_pairs(v, [0], [LEAD + 2 * T])[0]
                row = dict(image=name, seed=seed, T=T, cond=cond, F=F, sec=round(dt, 1),
                           ks=ks, gap=[2 * k for k in ks], lpips_mirror=best,
                           lpips_return_vs_lead=ret, lpips_return_vs_frame0=ret0,
                           pad_converge=m.pad_converge)
                with raw.open("a") as f: f.write(json.dumps(row) + "\n")
                write_mp4(out / "videos" / f"{name}_T{T}_s{seed}_{cond}.mp4", v)
                print(f"{name} T={T} s={seed} {cond}: return LPIPS {ret:.3f} ({dt:.0f}s)", flush=True)
