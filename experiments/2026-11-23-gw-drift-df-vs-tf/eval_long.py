"""Long-rollout drift evaluation with ground truth, for the df and tf arms.

For each arm (EMA weights), action script and inference context noise:
roll out `horizon` frames from 31 real frames on held-out maps, decode every
generated frame back to tiles, and record per frame:

  acc_det   tile accuracy on cells determined by what the *real* context showed
  offman    share of cells matching no tile (nearest-atlas distance > real-frame p99.9)
  uniform   share of frames that collapsed to (near) a single tile class
  det_frac  share of cells that are determined at all
  acc_aligned  acc_det re-scored at the pose the model actually drew (phase-correlation
               shift vs the true frame), so position drift and world drift separate
  odo_err_px   that shift's L-inf size in pixels

Action scripts (true poses come from the simulator applying the actions):
  still  no-op throughout: every cell is determined, so this is pure drift
  pace   8 steps right, 8 steps left, repeated (16 px sway): moving but determined
  walk   the training distribution's momentum random walk: leaves the seen area

Writes results/long_eval.json and a few side-by-side clips.
Usage: python eval_long.py [--arms df tf] [--horizon 480] [--n 32]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from wm.gridworld import World, Knobs, VIEW, frame_shift  # noqa: E402
from wm.df import WorldDiT, Cfg, rollout  # noqa: E402
from wm.train_df import seen_mask, calibrate_threshold  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arms", nargs="+", default=["df", "tf"])
ap.add_argument("--horizon", type=int, default=480)
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--noise", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2, 0.4])
ap.add_argument("--scripts", nargs="+", default=["still", "pace", "walk"])
ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--root", default=None, help="results dir holding <arm>/ckpt.pt (default: ./results)")
args = ap.parse_args()

R = Path(args.root) if args.root else HERE / "results"
targs = json.loads((R / args.arms[0] / "args.json").read_text())
world = World(targs["n_maps"] + 200, seed=1000, knobs=Knobs(shading=targs["shading"]))
eval_pool = torch.arange(targs["n_maps"], targs["n_maps"] + 200, device="cuda")
thr = calibrate_threshold(world, eval_pool)
K = targs["window"] - 1
H = args.horizon


def scripted(script, mi, a_ctx, p_ctx, gen):
    """Continue from the context's last pose with the given action script."""
    B = len(mi)
    if script == "walk":
        return None
    acts = torch.zeros(B, H, dtype=torch.long, device="cuda")
    if script == "pace":
        ph = (torch.arange(H, device="cuda") // 8) % 2
        acts[:] = torch.where(ph == 0, 4, 3)[None]
    pos = p_ctx[:, -1]; poss = []
    for t in range(H):
        pos, _ = world.step(mi, pos, acts[:, t]); poss.append(pos)
    return torch.cat([a_ctx, acts], 1), torch.cat([p_ctx, torch.stack(poss, 1)], 1)


results = {}
for arm in args.arms:
    sd = torch.load(R / arm / "ckpt.pt", map_location="cuda")
    model = WorldDiT(Cfg(**sd["cfg"])).cuda().eval()
    model.load_state_dict(sd["ema"])
    for script in args.scripts:
        g = torch.Generator().manual_seed(2024)
        if script == "walk":
            mi, a, p = world.random_walks(args.n, K + H, g, map_pool=eval_pool)
        else:
            mi, a0, p0 = world.random_walks(args.n, K, g, map_pool=eval_pool)
            a, p = scripted(script, mi, a0, p0, g)
        real = world.render(mi.repeat_interleave(K), p[:, :K].reshape(-1, 2)).view(args.n, K, 3, VIEW, VIEW)
        seen = seen_mask(world, mi, p[:, :K])
        for cn in args.noise:
            gg = torch.Generator(device="cuda").manual_seed(7)
            with torch.autocast("cuda", torch.bfloat16):
                out = rollout(model, real, a, steps=args.steps, ctx_noise=cn, gen=gg)
            acc, off, uni, det, acc_al, odo = [], [], [], [], [], []
            for k in range(H):
                f = out[:, K + k].float()
                POS = p[:, K + k]
                pred, ty, tx, dist = world.decode(f, POS)
                true = world.true_ids(mi, ty, tx)
                dm = seen[torch.arange(args.n, device="cuda")[:, None, None], ty, tx]
                dm[:, 3, 3] = False
                acc.append(((pred == true) & dm).sum().item() / max(1, dm.sum().item()))
                cells = torch.ones_like(dist, dtype=torch.bool); cells[:, 3, 3] = False
                off.append((dist > thr)[cells].float().mean().item())
                flat = pred.flatten(1); mode = torch.mode(flat, 1).values
                uni.append(((flat == mode[:, None]).float().mean(1) >= 0.95).float().mean().item())
                det.append(dm.float().mean().item())
                # position drift vs world drift: re-score at the pose the model actually drew
                sh = frame_shift(f, world.render(mi, POS))
                pred2, ty2, tx2, _ = world.decode(f, POS - sh)
                true2 = world.true_ids(mi, ty2, tx2)
                dm2 = seen[torch.arange(args.n, device="cuda")[:, None, None], ty2, tx2]; dm2[:, 3, 3] = False
                acc_al.append(((pred2 == true2) & dm2).sum().item() / max(1, dm2.sum().item()))
                odo.append(sh.abs().amax(-1).float().mean().item())
            key = f"{arm}|{script}|{cn}"
            results[key] = dict(acc=acc, offman=off, uniform=uni, det=det, acc_aligned=acc_al, odo_err_px=odo)
            q = lambda x, a_, b_: float(np.mean(x[a_:b_]))
            print(f"{key}: acc 1-16 {q(acc,0,16):.3f} 64-96 {q(acc,64,96):.3f} 224-256 {q(acc,224,256):.3f} {H-32}-{H} {q(acc,H-32,H):.3f} | aligned(last) {q(acc_al,H-32,H):.3f} odo(last) {q(odo,H-32,H):.1f}px | off(last) {q(off,H-32,H):.3f} uniform(last) {q(uni,H-32,H):.2f}", flush=True)
            if cn in (0.0, 0.1) and script in ("still", "pace"):
                import imageio.v2 as iio
                m6 = min(6, args.n)
                vid = torch.cat([world.render(mi[:m6].repeat_interleave(K + H), p[:m6].reshape(-1, 2)).view(m6, K + H, 3, VIEW, VIEW), out[:m6].float()], -1)
                vid = ((vid.permute(1, 0, 3, 4, 2).reshape(K + H, m6 * VIEW, 2 * VIEW, 3) + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()
                (R / "clips").mkdir(exist_ok=True)
                iio.mimwrite(str(R / "clips" / f"{arm}_{script}_cn{cn}.mp4"), list(np.repeat(np.repeat(vid, 2, 1), 2, 2)), fps=25,
                             codec="libx264", quality=None, ffmpeg_params=["-crf", "22", "-pix_fmt", "yuv420p"])
        (R / "long_eval.json").write_text(json.dumps(results))
print("done")
