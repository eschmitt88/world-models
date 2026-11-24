"""Equal-compute memory comparison on the gridworld (ADR 0001, experiment 2/3).

Arms (same data, same tokens per sequence, same wall-clock budget):
  win  window of 36 frames, no memory
  mem  window of 32 frames + 4 frames retrieved from older history by pose
Both get a relative-pose embedding (oracle poses) so they differ only in what
they can attend to.

Training data: revisit walks (walk out L ~ U[24,80] steps, retrace exactly,
carry on), with the training window placed around the return (end ~ 2L), so
that remembering the outbound frames pays off.

Eval ("out and back", held-out maps): 31 real context frames, then the model
generates the rest of a walk that keeps going out for G more steps and
retraces all the way back to the start. On the return we score
  real_acc    tile accuracy on cells the *real* context showed (memory vs truth)
  imag_cons   agreement with the model's *own* earlier decoded tiles on cells it
              first saw only in its generated frames (does it keep its imagination?)
as a function of the gap G.

Usage: python -m wm.train_mem --out DIR --arm win|mem --hours H
       python -m wm.train_mem --eval DIR1 DIR2 ... --gaps 8 24 48 96
"""
from __future__ import annotations

import argparse, copy, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wm.gridworld import World, Knobs, VIEW, revisit_walks, N_ACTIONS, frame_shift  # noqa: E402
from wm.df import WorldDiT, Cfg, rollout, rollout_mem, retrieve  # noqa: E402

LTOT = 176
M = 4
# eval scores every rollout twice: raw (at the true pose) and aligned (re-registered to the
# true frame by phase correlation, so position drift does not count as forgetting)


def batch(world, arm, B, gen, pool):
    W = 32 if arm == "mem" else 36
    mi, a, p, L = revisit_walks(world, B, LTOT, gen, map_pool=pool, out_min=24, out_max=80)
    jit = torch.randint(-8, 9, (B,), generator=gen, device="cpu").to(world.device)
    e = (2 * L + jit).clamp(W + M + 3, LTOT - 1)
    ar = torch.arange(W, device=world.device)
    widx = e[:, None] - W + 1 + ar[None]                              # [B,W]
    bi = torch.arange(B, device=world.device)[:, None]
    wpos, wact = p[bi, widx], a[bi, widx]
    if arm == "mem":
        H = LTOT
        hidx = torch.arange(H, device=world.device)[None].expand(B, H)
        valid = hidx < widx[:, :1]
        midx = retrieve(p, wpos, M, valid)
        mpos = p[bi, midx]
        pos = torch.cat([mpos, wpos], 1)
        act = torch.cat([torch.full((B, M), N_ACTIONS, device=world.device, dtype=torch.long), wact], 1)
    else:
        pos, act = wpos, wact
    T = pos.shape[1]
    x0 = world.render(mi.repeat_interleave(T), pos.reshape(-1, 2)).view(B, T, 3, VIEW, VIEW)
    rel = pos - wpos[:, :1]
    return x0, act, rel, (M if arm == "mem" else 0)


def loss_mem(model, x0, act, rel, n_mem):
    B, T = x0.shape[:2]
    eps = torch.randn_like(x0)
    t = torch.rand(B, T, device=x0.device)
    t[:, :n_mem] = 0
    xt = (1 - t)[..., None, None, None] * x0 + t[..., None, None, None] * eps
    v = model(xt, t, act, pose=rel, n_mem=n_mem)
    return F.mse_loss(v[:, n_mem:], (eps - x0)[:, n_mem:])


def train(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    torch.manual_seed(args.seed)
    world = World(args.n_maps + 200, seed=1000, knobs=Knobs(shading=args.shading))
    pool = torch.arange(args.n_maps, device="cuda")
    cfg = Cfg(d=args.d, depth=args.depth, heads=args.heads, window=32 if args.arm == "mem" else 36,
              n_mem=M if args.arm == "mem" else 0, pose=True)
    model = WorldDiT(cfg).cuda(); ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    log = open(out / "log.jsonl", "a")
    log.write(json.dumps(dict(event="start", params=sum(p.numel() for p in model.parameters()), cfg=vars(cfg))) + "\n"); log.flush()
    g = torch.Generator().manual_seed(args.seed)
    start, step, lw, lt = time.time(), 0, 0.0, time.time()
    while time.time() - start < args.hours * 3600:
        for pg in opt.param_groups: pg["lr"] = args.lr * min(1.0, (step + 1) / 1000)
        lacc = 0.0
        for _ in range(args.accum):
            x0, act, rel, nm = batch(world, args.arm, args.batch // args.accum, g, pool)
            with torch.autocast("cuda", torch.bfloat16):
                loss = loss_mem(model, x0, act, rel, nm) / args.accum
            loss.backward(); lacc += loss.item()
        loss = torch.tensor(lacc)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            dk = min(0.999, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm, 1 - dk)
        lw += loss.item(); step += 1
        if step % 100 == 0:
            log.write(json.dumps(dict(event="train", step=step, loss=lw / 100, sps=100 / (time.time() - lt))) + "\n"); log.flush()
            lw, lt = 0.0, time.time()
        if step % 5000 == 0:
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), step=step, cfg=vars(cfg)), out / "ckpt.pt")
    torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), step=step, cfg=vars(cfg)), out / "ckpt.pt")
    log.write(json.dumps(dict(event="end", step=step)) + "\n"); log.flush()


@torch.no_grad()
def evaluate(dirs, gaps, n=24, steps=10, seed=99):
    a0 = json.loads((Path(dirs[0]) / "args.json").read_text())
    world = World(a0.get("n_maps", 2000) + 200, seed=1000, knobs=Knobs(shading=a0.get("shading", 0.25)))
    pool = torch.arange(a0.get("n_maps", 2000), a0.get("n_maps", 2000) + 200, device="cuda")
    K = 31
    res = {}
    for d in dirs:
        sd = torch.load(Path(d) / "ckpt.pt", map_location="cuda")
        cfg = Cfg(**sd["cfg"]); model = WorldDiT(cfg).cuda().eval(); model.load_state_dict(sd["ema"])
        name = Path(d).name
        for G in gaps:
            g = torch.Generator().manual_seed(seed + G)
            Lout = K - 1 + G
            T = 2 * Lout + 1
            mi, a, p, _ = revisit_walks(world, n, T, g, map_pool=pool, out_min=Lout, out_max=Lout)
            real = world.render(mi.repeat_interleave(K), p[:, :K].reshape(-1, 2)).view(n, K, 3, VIEW, VIEW)
            gg = torch.Generator(device="cuda").manual_seed(seed)
            with torch.autocast("cuda", torch.bfloat16):
                if cfg.n_mem or cfg.pose:
                    out = rollout_mem(model, real, p[:, :K], a, p, steps=steps, gen=gg) if cfg.n_mem else \
                          _rollout_pose(model, real, a, p, steps, gg)
                else:
                    out = rollout(model, real, a, steps=steps, gen=gg)
            out = out.float()
            for ALIGN_ in (False, True):
                S = world.S
                first = torch.full((n, S, S), -1, dtype=torch.long, device="cuda")   # tile id at first sight
                src = torch.zeros((n, S, S), dtype=torch.long, device="cuda")        # 1 = real, 2 = imagined
                first_t = torch.zeros((n, S, S), dtype=torch.long, device="cuda")
                stats = {"real": [], "imag": [], "mirror_real": [], "mirror_imag": []}
                outbound = {}   # t -> decoded tile ids (raw pose) of the generated outbound frame
                bi = torch.arange(n, device="cuda")[:, None, None]
                for t in range(T):
                    f = real[:, t] if t < K else out[:, t]
                    pt = p[:, t]
                    if t >= K and ALIGN_:   # score at the pose the model actually drew
                        pt = pt - frame_shift(f, world.render(mi, p[:, t]))
                    pred, ty, tx, dist = world.decode(f, pt)
                    true = world.true_ids(mi, ty, tx)
                    cells = torch.ones_like(pred, dtype=torch.bool); cells[:, 3, 3] = False
                    seenb = first[bi, ty, tx] >= 0
                    if t >= K and t <= Lout and not ALIGN_:
                        outbound[t] = pred.clone()
                    if t > Lout and not ALIGN_ and (2 * Lout - t) in outbound:
                        # mirror test: frame t shares its pose with outbound frame 2L - t, so every
                        # cell sits at the same place in view; compare with the model's own drawing
                        prev = outbound[2 * Lout - t]
                        r = cells & (src[bi, ty, tx] == 1); im = cells & (src[bi, ty, tx] == 2)
                        stats["mirror_real"].append(((pred == true) & r).sum().item() / max(1, r.sum().item()))
                        stats["mirror_imag"].append(((pred == prev) & im).sum().item() / max(1, im.sum().item()))
                    if t >= K and t > Lout:            # on the way back: score revisited cells
                        gap = t - first_t[bi, ty, tx]
                        r = seenb & cells & (src[bi, ty, tx] == 1)
                        im = seenb & cells & (src[bi, ty, tx] == 2)
                        stats["real"].append(((pred == true) & r).sum().item() / max(1, r.sum().item()))
                        stats["imag"].append(((pred == first[bi, ty, tx]) & im).sum().item() / max(1, im.sum().item()))
                    newc = (~seenb) & cells
                    first[bi.expand_as(ty)[newc], ty[newc], tx[newc]] = (true if t < K else pred)[newc]
                    src[bi.expand_as(ty)[newc], ty[newc], tx[newc]] = 1 if t < K else 2
                    first_t[bi.expand_as(ty)[newc], ty[newc], tx[newc]] = t
                # the last 16 frames revisit the real context; earlier return frames revisit imagined ground
                r_last = float(np.mean(stats["real"][-16:]))
                i_mid = float(np.mean([x for x in stats["imag"][:max(1, len(stats["imag"]) - 16)]]))
                mi_ = float(np.mean(stats["mirror_imag"][:max(1, len(stats["mirror_imag"]) - 16)])) if stats["mirror_imag"] else float("nan")
                mr_ = float(np.mean(stats["mirror_real"][-16:])) if stats["mirror_real"] else float("nan")
                res[f"{name}|G{G}|{'aligned' if ALIGN_ else 'raw'}"] = dict(real_acc_end=r_last, imag_cons=i_mid, real_curve=stats["real"], imag_curve=stats["imag"],
                                                                           mirror_imag=mi_, mirror_real=mr_, mirror_imag_curve=stats["mirror_imag"])
                print(f"{name} G={G} {'aligned' if ALIGN_ else 'raw'}: real-context recall at return {r_last:.3f} | imagined self-consistency {i_mid:.3f}"
                      + (f" | mirror: imagined {mi_:.3f} real {mr_:.3f}" if not ALIGN_ else ""), flush=True)
    return res


@torch.no_grad()
def _rollout_pose(model, ctx, actions, poses, steps, gen):
    """Sliding-window rollout for a pose-conditioned model without memory."""
    W = model.cfg.window
    B, K = ctx.shape[:2]; N = actions.shape[1] - K
    frames = [ctx[:, i] for i in range(K)]
    ts = torch.linspace(1, 0, steps + 1, device=ctx.device)
    for n_ in range(N):
        i = K + n_; c0 = max(0, i - (W - 1))
        cf = torch.stack(frames[c0:i], 1); a = actions[:, c0:i + 1]; ps = poses[:, c0:i + 1]
        x = torch.randn((B, 3, VIEW, VIEW), device=ctx.device, generator=gen)
        tc = torch.zeros(B, cf.shape[1], device=ctx.device)
        for k in range(steps):
            v = model(torch.cat([cf, x[:, None]], 1), torch.cat([tc, ts[k].expand(B, 1)], 1), a, pose=ps - ps[:, :1])[:, -1]
            x = x + (ts[k + 1] - ts[k]) * v
        frames.append(x.clamp(-1, 1))
    return torch.stack(frames, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out"); ap.add_argument("--arm", default="mem"); ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--d", type=int, default=384); ap.add_argument("--depth", type=int, default=8); ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--accum", type=int, default=2, help="gradient-accumulation micro-batches (effective batch = batch)")
    ap.add_argument("--n_maps", type=int, default=2000); ap.add_argument("--shading", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", nargs="+"); ap.add_argument("--gaps", nargs="+", type=int, default=[8, 24, 48, 96])
    ap.add_argument("--n", type=int, default=24); ap.add_argument("--save")
    args = ap.parse_args()
    if args.eval:
        r = evaluate(args.eval, args.gaps, n=args.n)
        if args.save: Path(args.save).write_text(json.dumps(r))
    else:
        train(args)
