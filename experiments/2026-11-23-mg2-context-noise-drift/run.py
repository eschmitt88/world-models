"""Long rollouts of Matrix-Game 2.0 under a fixed walking action track, swept
over context_noise. Saves full videos (results/videos/, gitignored) and a row
per run to results/runs.jsonl; per-frame metrics come from analyze.py."""
import json, sys, time
from pathlib import Path
import numpy as np, yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from wm.mg2 import MG2, MG2_DIR, write_mp4  # noqa: E402

cfg = yaml.safe_load((HERE / "config.yaml").read_text())
out = HERE / "results"; (out / "videos").mkdir(parents=True, exist_ok=True)
runs = out / "runs.jsonl"
done = {(r["image"], r["seed"], r["context_noise"]) for r in map(json.loads, runs.read_text().splitlines())} if runs.exists() else set()


def walk(F, w):
    kb = np.zeros((F, 4)); ms = np.zeros((F, 2))
    kb[:, 0] = w["forward"]
    f = np.arange(F)
    ms[:, 1] = w["yaw_amp"] * np.sin(2 * np.pi * f / w["yaw_period"])
    for i, s in enumerate(range(w["strafe_every"], F, w["strafe_every"])):
        kb[s:s + w["strafe_len"], 2 if i % 2 == 0 else 3] = 1
    return kb, ms


m = MG2()
L = cfg["latents"]; F = 4 * (L - 1) + 1
kb, ms = walk(F, cfg["walk"])
np.savez_compressed(out / "actions.npz", keyboard=kb, mouse=ms)
for name in cfg["images"]:
    img = MG2_DIR / f"demo_images/universal/{name}.png"
    for seed in cfg["seeds"]:
        for cn in cfg["context_noise"]:
            if (name, seed, cn) in done:
                continue
            t0 = time.time()
            v = m.rollout(img, kb, ms, seed=seed, context_noise=cn)
            dt = time.time() - t0
            path = out / "videos" / f"{name}_s{seed}_cn{cn}.mp4"
            write_mp4(path, v, crf=17)
            with runs.open("a") as f:
                f.write(json.dumps(dict(image=name, seed=seed, context_noise=cn, F=int(v.shape[0]),
                                        sec=round(dt, 1), video=str(path.relative_to(HERE)))) + "\n")
            print(f"{name} s={seed} cn={cn}: {v.shape[0]} frames in {dt:.0f}s", flush=True)
