"""Train a WorldDiT on the gridworld with on-the-fly data, with periodic
ground-truth evaluation. Usage:

    python -m wm.train_df --out DIR [--mode df|tf] [--hours H] [--d 384 --depth 8]

Writes DIR/log.jsonl (train loss and eval metrics), DIR/ckpt.pt (latest, with
EMA weights), DIR/samples/*.mp4.

Eval (held-out maps, fixed seed):
  onestep_acc   tile accuracy of frame 32 generated from 31 real frames (kill-rule metric)
  roll_acc_k    tile accuracy over cells *determined by the real context* (tiles seen in
                the 31 context frames), for generated frames k after the context
  offman_k      fraction of decoded cells whose nearest-atlas distance exceeds the
                99.9th percentile seen on real frames (off-manifold rate)
"""
from __future__ import annotations

import argparse, copy, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wm.gridworld import World, Knobs, VIEW, TILE  # noqa: E402
from wm.df import WorldDiT, Cfg, loss_fn, rollout  # noqa: E402


def seen_mask(world, mi, pos):
    """Tiles visible in any frame of pos [B,K,2] -> bool [B,S,S]."""
    B, K = pos.shape[:2]
    S = world.S
    m = torch.zeros(B, S, S, dtype=torch.bool, device=pos.device)
    top = pos - VIEW // 2
    t0 = torch.div(top, TILE, rounding_mode="floor")
    for k in range(K):
        for dy in range(VIEW // TILE + 1):
            for dx in range(VIEW // TILE + 1):
                ty = (t0[:, k, 0] + dy).clamp(0, S - 1); tx = (t0[:, k, 1] + dx).clamp(0, S - 1)
                m[torch.arange(B), ty, tx] = True
    return m


@torch.no_grad()
def evaluate(model, world, eval_pool, thr, steps=10, n=32, horizon=96, ctx_noise=0.0, seed=1234, save=None):
    g = torch.Generator().manual_seed(seed)
    K = model.cfg.window - 1
    mi, a, p = world.random_walks(n, K + horizon, g, map_pool=eval_pool)
    real = world.render(mi.repeat_interleave(K + horizon), p.reshape(-1, 2)).view(n, K + horizon, 3, VIEW, VIEW)
    gg = torch.Generator(device="cuda").manual_seed(seed)
    with torch.autocast("cuda", torch.bfloat16):
        out = rollout(model, real[:, :K], a, steps=steps, ctx_noise=ctx_noise, gen=gg)
    seen = seen_mask(world, mi, p[:, :K])
    res = {}
    accs, offs, dets = [], [], []
    for k in range(horizon):
        f = out[:, K + k].float()
        pred, ty, tx, dist = world.decode(f, p[:, K + k])
        true = world.true_ids(mi, ty, tx)
        det = seen[torch.arange(n, device=f.device)[:, None, None], ty, tx]
        det[:, 3, 3] = False
        ok = (pred == true)
        accs.append((ok & det).sum().item() / max(1, det.sum().item()))
        offs.append((dist > thr).float()[:, torch.arange(7) != 3].mean().item())
        dets.append(det.float().mean().item())
    res["onestep_acc"] = accs[0]
    for k in (1, 8, 32, 64, 96):
        if k <= horizon:
            res[f"roll_acc_{k}"] = float(np.mean(accs[max(0, k - 4):k]))
            res[f"offman_{k}"] = float(np.mean(offs[max(0, k - 4):k]))
            res[f"det_frac_{k}"] = float(np.mean(dets[max(0, k - 4):k]))
    res["acc_curve"] = accs; res["offman_curve"] = offs
    if save is not None:
        import imageio.v2 as iio
        vid = torch.cat([real[:8], out[:8].float()], -1)             # real | generated, side by side
        vid = ((vid.permute(1, 0, 3, 4, 2).reshape(K + horizon, 8 * VIEW, 2 * VIEW, 3) + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()
        iio.mimwrite(str(save), list(np.repeat(np.repeat(vid, 2, 1), 2, 2)), fps=12, codec="libx264", quality=None,
                     ffmpeg_params=["-crf", "22", "-pix_fmt", "yuv420p"])
    return res


def calibrate_threshold(world, pool, n=256, seed=7):
    g = torch.Generator().manual_seed(seed)
    mi, a, p = world.random_walks(n, 8, g, map_pool=pool)
    fr = world.render(mi.repeat_interleave(8), p.reshape(-1, 2))
    _, _, _, d = world.decode(fr, p.reshape(-1, 2))
    return float(torch.quantile(d.flatten().float(), 0.999))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="df")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--d", type=int, default=384); ap.add_argument("--depth", type=int, default=8); ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1, help="gradient-accumulation micro-batches (effective batch = batch)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_maps", type=int, default=2000)
    ap.add_argument("--shading", type=float, default=0.25)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    torch.manual_seed(args.seed)
    kn = Knobs(shading=args.shading)
    world = World(args.n_maps + 200, seed=1000 + args.seed * 0, knobs=kn)   # maps fixed across seeds
    train_pool = torch.arange(args.n_maps, device="cuda")
    eval_pool = torch.arange(args.n_maps, args.n_maps + 200, device="cuda")
    thr = calibrate_threshold(world, eval_pool)
    cfg = Cfg(d=args.d, depth=args.depth, heads=args.heads, window=args.window, mode=args.mode)
    model = WorldDiT(cfg).cuda()
    ema = copy.deepcopy(model).eval()
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    start = time.time(); step = 0
    ck = out / "ckpt.pt"
    if ck.exists():
        sd = torch.load(ck, map_location="cuda")
        model.load_state_dict(sd["model"]); ema.load_state_dict(sd["ema"]); opt.load_state_dict(sd["opt"]); step = sd["step"]
        start -= sd.get("elapsed", 0)
    log = open(out / "log.jsonl", "a")
    log.write(json.dumps(dict(event="start", params=nparam, thr=thr, step=step)) + "\n"); log.flush()
    print(f"params {nparam/1e6:.1f}M thr {thr:.3f}", flush=True)
    g = torch.Generator().manual_seed(args.seed * 1000 + step)
    T = args.window
    lw, lt = 0.0, time.time()
    while True:
        elapsed = time.time() - start
        done = elapsed > args.hours * 3600
        if step % args.eval_every == 0 or done:
            ema.eval()
            r = evaluate(ema, world, eval_pool, thr, save=out / "samples" / f"step{step:07d}.mp4")
            r.update(event="eval", step=step, elapsed=round(elapsed), frames_seen=step * args.batch * T)
            log.write(json.dumps(r) + "\n"); log.flush()
            print(f"[eval {step}] onestep {r['onestep_acc']:.3f} roll8 {r['roll_acc_8']:.3f} roll32 {r['roll_acc_32']:.3f} roll96 {r['roll_acc_96']:.3f} off96 {r['offman_96']:.3f}", flush=True)
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step,
                            elapsed=elapsed, cfg=vars(cfg)), ck)
            if step % 8000 == 0 and step > 0:   # EMA snapshots for checkpoint-level probes (contraction vs survival)
                (out / "snapshots").mkdir(exist_ok=True)
                torch.save(dict(ema=ema.state_dict(), step=step, cfg=vars(cfg)), out / "snapshots" / f"ema_{step:07d}.pt")
        if done:
            break
        lr = args.lr * min(1.0, (step + 1) / 1000)
        for pg in opt.param_groups: pg["lr"] = lr
        mb = args.batch // args.accum
        loss_acc = 0.0
        for _ in range(args.accum):
            mi, a, p = world.random_walks(mb, T, g, map_pool=train_pool)
            x0 = world.render(mi.repeat_interleave(T), p.reshape(-1, 2)).view(mb, T, 3, VIEW, VIEW)
            with torch.autocast("cuda", torch.bfloat16):
                loss = loss_fn(model, x0, a) / args.accum
            loss.backward()
            loss_acc += loss.item()
        loss = torch.tensor(loss_acc)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            dk = min(0.999, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - dk)
        lw += loss.item(); step += 1
        if step % 100 == 0:
            log.write(json.dumps(dict(event="train", step=step, loss=lw / 100, lr=lr, sps=100 / (time.time() - lt))) + "\n"); log.flush()
            lw, lt = 0.0, time.time()


if __name__ == "__main__":
    main()
