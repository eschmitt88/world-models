"""Out-and-back memory test for Matrix-Game 3.0, memory ON vs OFF.
Same protocol and metric as experiments/2026-11-23-mg2-memory-horizon/run.py,
with frames resized to 640x352 before the half-res LPIPS so the numbers are
on MG2's scale."""
import json, sys, time
from pathlib import Path
import numpy as np, torch, yaml, cv2
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from wm.mg3 import MG3, MG3_DIR  # noqa: E402
from wm.mg2 import write_mp4  # noqa: E402

cfg = yaml.safe_load((HERE / "config.yaml").read_text())
out = HERE / "results"; (out / "videos").mkdir(parents=True, exist_ok=True)
raw = out / "raw.jsonl"
done = {(r["image"], r["seed"], r["T"], r["memory"]) for r in map(json.loads, raw.read_text().splitlines())} if raw.exists() else set()

import lpips  # noqa: E402
lp = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
H, W = cfg["lpips_res"]

def to_t(frames):
    fr = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in frames])
    t = torch.from_numpy(fr).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1
    return torch.nn.functional.interpolate(t, scale_factor=0.5, mode="area")

@torch.no_grad()
def lpips_pairs(v, a, b):
    return lp(to_t(v[a]), to_t(v[b])).flatten().tolist()

m = MG3()
LEAD, YAW = cfg["lead"], cfg["yaw"]
for name in cfg["images"]:
    img = Image.open(MG3_DIR / f"demo_images/{name}/image.png")
    prompt = (MG3_DIR / f"demo_images/{name}/prompt.txt").read_text().strip()
    for T in cfg["T"]:
        F = LEAD + 2 * T + 1
        for seed in cfg["seeds"]:
            for mem in cfg["memory"]:
                if (name, seed, T, mem) in done: continue
                kb = np.zeros((F, 6)); ms = np.zeros((F, 2))
                ms[LEAD:LEAD + T, 1] = YAW; ms[LEAD + T:LEAD + 2 * T, 1] = -YAW
                t0 = time.time()
                v, mlog = m.rollout(img, prompt, kb, ms, seed=seed, use_memory=mem, return_memory_log=True)
                dt = time.time() - t0
                tp = LEAD + T
                ks = list(range(0, T + 1, max(1, T // 24)))
                if ks[-1] != T: ks.append(T)
                best = []
                for k in ks:
                    c = [tp + k + d for d in range(-2, 3) if 0 <= tp + k + d < len(v)]
                    best.append(min(lpips_pairs(v, [tp - k] * len(c), c)))
                row = dict(image=name, seed=seed, T=T, memory=mem, F=int(len(v)), sec=round(dt, 1), ks=ks,
                           gap=[2 * k for k in ks], lpips_mirror=best,
                           lpips_return_vs_lead=lpips_pairs(v, [LEAD], [LEAD + 2 * T])[0],
                           memory_log=mlog)
                with raw.open("a") as f: f.write(json.dumps(row) + "\n")
                write_mp4(out / "videos" / f"{name}_T{T}_s{seed}_mem{int(mem)}.mp4", v, scale=0.5)
                print(f"{name} T={T} s={seed} mem={mem}: return LPIPS {row['lpips_return_vs_lead']:.3f} ({dt:.0f}s)", flush=True)
