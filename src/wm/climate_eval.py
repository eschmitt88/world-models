"""Weather / climate / calibration / contraction evaluation (ADR 0002).

Sections (each writes its own JSON, so a partial run is still usable):
  contraction  one-step gain under context perturbation, per checkpoint, plus a
               single-member 256-frame rollout per checkpoint for survival
  ensemble     C contexts x M members per arm: weather (skill on determined cells),
               climate (JS distance of imagined-cell statistics from real), calibration
               (ensemble disagreement vs error), survival

Usage:
  python -m wm.climate_eval --exp DIR --section contraction --ckpts a.pt b.pt ...
  python -m wm.climate_eval --exp DIR --section ensemble --arm NAME CKPT CTX_NOISE SNAP_EVERY [--arm ...]
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wm.gridworld import World, Knobs, VIEW, TILE, N_FLOOR, N_LANDMARK, frame_shift  # noqa: E402
from wm.df import WorldDiT, Cfg  # noqa: E402
from wm.train_df import seen_mask, calibrate_threshold  # noqa: E402

K = 31            # real context frames
STEPS = 10        # sampler steps
DEV = "cuda"


# ---------------------------------------------------------------- utilities
def load(ckpt):
    sd = torch.load(ckpt, map_location=DEV)
    m = WorldDiT(Cfg(**sd["cfg"])).to(DEV).eval()
    m.load_state_dict(sd["ema"])
    return m, sd.get("step", -1)


def coarse(ids):
    """tile id -> 0 wall, 1 floor, 2 landmark"""
    return torch.where(ids == 0, 0, torch.where(ids <= N_FLOOR, 1, 2))


@torch.no_grad()
def gen_frame(model, ctx, tctx, acts, gen):
    """Sample one next frame. ctx [B,C,3,64,64], tctx [B,C] noise levels the model is told,
    acts [B,C+1]."""
    B = ctx.shape[0]
    x = torch.randn((B, 3, VIEW, VIEW), device=DEV, generator=gen)
    ts = torch.linspace(1, 0, STEPS + 1, device=DEV)
    for k in range(STEPS):
        v = model(torch.cat([ctx, x[:, None]], 1), torch.cat([tctx, ts[k].expand(B, 1)], 1), acts)[:, -1]
        x = x + (ts[k + 1] - ts[k]) * v
    return x.clamp(-1, 1)


class Snapper:
    """Truth-free projection onto the tile manifold: register the 8x8 tile grid by the
    sub-tile offset with the best atlas fit, replace each fully visible cell with its
    nearest atlas tile (keeping the cell's brightness, since shading is smooth and
    multiplicative), and restore the agent marker."""

    def __init__(self, world):
        a = world.atlas
        self.A = a / (a.mean(dim=(1, 2, 3), keepdim=True) + 1e-3)
        self.atlas = a

    def cells(self, x, oy, ox):
        n = x.shape[0]
        reg = x[:, :, oy:oy + 7 * TILE, ox:ox + 7 * TILE]
        return reg.reshape(n, 3, 7, TILE, 7, TILE).permute(0, 2, 4, 1, 3, 5).reshape(n, 49, 3, TILE, TILE)

    @torch.no_grad()
    def __call__(self, frames):
        x = (frames.float() + 1) / 2
        n = x.shape[0]
        best = None
        for oy in range(TILE):
            for ox in range(TILE):
                c = self.cells(x, oy, ox)
                cn = c / (c.mean(dim=(-3, -2, -1), keepdim=True) + 1e-3)
                d = ((cn[:, :, None] - self.A[None, None]) ** 2).mean(dim=(-3, -2, -1))   # [n,49,K]
                dist, pred = d.min(-1)
                score = dist.mean(1)
                if best is None:
                    best = [score, torch.full((n,), oy, device=DEV), torch.full((n,), ox, device=DEV), pred, c]
                else:
                    upd = score < best[0]
                    best[0] = torch.where(upd, score, best[0])
                    best[1] = torch.where(upd, oy, best[1]); best[2] = torch.where(upd, ox, best[2])
                    best[3] = torch.where(upd[:, None], pred, best[3])
                    best[4] = torch.where(upd[:, None, None, None, None], c, best[4])
        _, oy, ox, pred, c = best
        bright = c.mean(dim=(-3, -2, -1), keepdim=True) / (self.atlas[pred].mean(dim=(-3, -2, -1), keepdim=True) + 1e-3)
        tiles = (self.atlas[pred] * bright).clamp(0, 1)                                    # [n,49,3,8,8]
        out = x.clone()
        img = tiles.view(n, 7, 7, 3, TILE, TILE).permute(0, 3, 1, 4, 2, 5).reshape(n, 3, 7 * TILE, 7 * TILE)
        for i in range(n):   # offsets differ per frame
            out[i, :, oy[i]:oy[i] + 7 * TILE, ox[i]:ox[i] + 7 * TILE] = img[i]
        cc = VIEW // 2
        out[:, :, cc - 2:cc + 2, cc - 2:cc + 2] = x[:, :, cc - 2:cc + 2, cc - 2:cc + 2]
        return out * 2 - 1


@torch.no_grad()
def rollout(model, real_ctx, acts, ctx_noise=0.0, snap=None, snap_every=0, gen=None):
    """Sliding-window rollout with optional snap-to-tile every `snap_every` generated frames."""
    W = model.cfg.window
    B = real_ctx.shape[0]
    frames = [real_ctx[:, i] for i in range(real_ctx.shape[1])]
    N = acts.shape[1] - len(frames)
    for n in range(N):
        i = len(frames); c0 = max(0, i - (W - 1))
        cf = torch.stack(frames[c0:i], 1)
        if ctx_noise > 0:
            cf = (1 - ctx_noise) * cf + ctx_noise * torch.randn(cf.shape, device=DEV, generator=gen)
        tc = torch.full((B, cf.shape[1]), float(ctx_noise), device=DEV)
        with torch.autocast("cuda", torch.bfloat16):
            f = gen_frame(model, cf, tc, acts[:, c0:i + 1], gen).float()
        if snap is not None and snap_every and (n + 1) % snap_every == 0:
            f = snap(f)
        frames.append(f)
    return torch.stack(frames, 1)


def jsd(p, q):
    p = np.asarray(p, float) + 1e-9; q = np.asarray(q, float) + 1e-9
    p /= p.sum(); q /= q.sum(); m = (p + q) / 2
    return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


class ClimateAcc:
    """Accumulates structural statistics of decoded cells (7x7 grids + a mask of which
    cells count) and returns normalised histograms."""

    def __init__(self):
        self.occ = np.zeros(3); self.col = np.zeros(N_FLOOR); self.pat = np.zeros(81)
        self.same = np.zeros(2); self.lm = np.zeros(2)

    def add(self, pred, mask):
        """pred [N,7,7] tile ids, mask [N,7,7] bool (cells to count)."""
        c = coarse(pred)
        self.occ += np.bincount(c[mask].cpu().numpy(), minlength=3)
        fl = mask & (pred >= 1) & (pred <= N_FLOOR)
        self.col += np.bincount((pred[fl] - 1).cpu().numpy(), minlength=N_FLOOR)
        m4 = mask[:, :-1, :-1] & mask[:, 1:, :-1] & mask[:, :-1, 1:] & mask[:, 1:, 1:]
        code = c[:, :-1, :-1] * 27 + c[:, 1:, :-1] * 9 + c[:, :-1, 1:] * 3 + c[:, 1:, 1:]
        self.pat += np.bincount(code[m4].cpu().numpy(), minlength=81)
        for a, b in ((pred[:, :, :-1], pred[:, :, 1:]), (pred[:, :-1], pred[:, 1:])):
            mm = (mask[:, :, :-1] & mask[:, :, 1:]) if a.shape[-1] == 6 else (mask[:, :-1] & mask[:, 1:])
            ff = mm & (a >= 1) & (a <= N_FLOOR) & (b >= 1) & (b <= N_FLOOR)
            s = (a[ff] == b[ff]).sum().item(); self.same += [s, ff.sum().item() - s]
        lmk = (mask & (pred > N_FLOOR)).sum().item(); flo = (mask & (pred > 0)).sum().item()
        self.lm += [lmk, flo - lmk]

    def dist(self, other):
        return {k: jsd(getattr(self, k), getattr(other, k)) for k in ("occ", "col", "pat", "same", "lm")}

    def n(self):
        return int(self.occ.sum())


# ---------------------------------------------------------------- sections
def setup_world(exp_args):
    world = World(exp_args["n_maps"] + 200, seed=1000, knobs=Knobs(shading=exp_args["shading"]))
    pool = torch.arange(exp_args["n_maps"], exp_args["n_maps"] + 200, device=DEV)
    return world, pool


@torch.no_grad()
def section_contraction(world, pool, ckpts, n_clips=256, chunk=32, surv_n=32, surv_h=256, seed=5):
    thr = calibrate_threshold(world, pool)
    g = torch.Generator().manual_seed(seed)
    mi, a, p = world.random_walks(n_clips, K + 1, g, map_pool=pool)
    real = world.render(mi.repeat_interleave(K + 1), p.reshape(-1, 2)).view(n_clips, K + 1, 3, VIEW, VIEW)
    gs = torch.Generator().manual_seed(seed + 1)
    ms, as_, ps = world.random_walks(surv_n, K + surv_h, gs, map_pool=pool)
    sreal = world.render(ms.repeat_interleave(K), ps[:, :K].reshape(-1, 2)).view(surv_n, K, 3, VIEW, VIEW)
    out = []
    for ck in ckpts:
        model, step = load(ck)
        rec = dict(ckpt=str(ck), step=step, gain={}, onman={})
        for c0 in range(0, n_clips, chunk):
            sl = slice(c0, c0 + chunk)
            ctx = real[sl, :K]; acts = a[sl]; B = ctx.shape[0]
            t0 = torch.zeros(B, K, device=DEV)
            def base():
                gg = torch.Generator(device=DEV).manual_seed(11 + c0)
                with torch.autocast("cuda", torch.bfloat16):
                    return gen_frame(model, ctx, t0, acts, gg).float()
            y0 = base()
            for eps in (0.05, 0.1, 0.2):
                ge = torch.Generator(device=DEV).manual_seed(101 + c0)
                pert = ctx.clone()
                pert[:, -1] = (1 - eps) * ctx[:, -1] + eps * torch.randn(ctx[:, -1].shape, device=DEV, generator=ge)
                for told in ("true", "clean"):
                    tt = t0.clone()
                    if told == "true": tt[:, -1] = eps
                    gg = torch.Generator(device=DEV).manual_seed(11 + c0)
                    with torch.autocast("cuda", torch.bfloat16):
                        y = gen_frame(model, pert, tt, acts, gg).float()
                    gain = ((y - y0).pow(2).mean((1, 2, 3)).sqrt() / (pert[:, -1] - ctx[:, -1]).pow(2).mean((1, 2, 3)).sqrt())
                    rec["gain"].setdefault(f"eps{eps}_{told}", []).extend(gain.tolist())
            # on-manifold perturbations of the last context frame
            truth = real[sl, K]
            pred_t, ty, tx, _ = world.decode(truth, p[sl, K])
            true_ids = world.true_ids(mi[sl], ty, tx)
            shifted = ctx.clone(); shifted[:, -1] = world.render(mi[sl], p[sl, K - 1] + torch.tensor([0, TILE], device=DEV))
            gg = torch.Generator(device=DEV).manual_seed(11 + c0)
            with torch.autocast("cuda", torch.bfloat16):
                ys = gen_frame(model, shifted, t0, acts, gg).float()
            pr, _, _, _ = world.decode(ys, p[sl, K])
            p0, _, _, _ = world.decode(y0, p[sl, K])
            cells = torch.ones_like(pr, dtype=torch.bool); cells[:, 3, 3] = False
            rec["onman"].setdefault("shift_acc", []).extend(((pr == true_ids) & cells).float().flatten(1).sum(1).div(48).tolist())
            rec["onman"].setdefault("base_acc", []).extend(((p0 == true_ids) & cells).float().flatten(1).sum(1).div(48).tolist())
        # survival: single-member rollouts from fixed contexts
        gg = torch.Generator(device=DEV).manual_seed(77)
        ro = rollout(model, sreal, as_, gen=gg)
        offs, unis = [], []
        for k in range(surv_h):
            f = ro[:, K + k]
            pr, _, _, dist = world.decode(f, ps[:, K + k])
            cm = torch.ones_like(dist, dtype=torch.bool); cm[:, 3, 3] = False
            offs.append(((dist > thr) & cm).float().flatten(1).sum(1) / 48)
            flat = pr.flatten(1); mode = torch.mode(flat, 1).values
            unis.append((flat == mode[:, None]).float().mean(1) >= 0.95)
        offs = torch.stack(offs, 1); unis = torch.stack(unis, 1)
        def first(mask):
            any_ = mask.any(1); idx = mask.float().argmax(1)
            return torch.where(any_, idx, torch.full_like(idx, surv_h)).float()
        rec["survival_offman"] = first(offs > 0.3).tolist()
        rec["survival_uniform"] = first(unis).tolist()
        rec["offman_mean"] = offs.mean().item()
        out.append(rec)
        print(f"{Path(ck).parent.parent.name}/{Path(ck).name} step {step}: gain eps0.1 clean {np.mean(rec['gain']['eps0.1_clean']):.3f} "
              f"true {np.mean(rec['gain']['eps0.1_true']):.3f} | shift acc {np.mean(rec['onman']['shift_acc']):.3f} (base {np.mean(rec['onman']['base_acc']):.3f}) "
              f"| survival offman med {np.median(rec['survival_offman']):.0f} uniform med {np.median(rec['survival_uniform']):.0f}", flush=True)
    return out


@torch.no_grad()
def section_ensemble(world, pool, arms, C=16, M=8, H=256, seed=21, bins=32):
    thr = calibrate_threshold(world, pool)
    snapper = Snapper(world)
    g = torch.Generator().manual_seed(seed)
    mi, a, p = world.random_walks(C, K + H, g, map_pool=pool)
    real = world.render(mi.repeat_interleave(K + H), p.reshape(-1, 2)).view(C, K + H, 3, VIEW, VIEW)
    seen = seen_mask(world, mi, p[:, :K])
    # real-frame climate reference on the same trajectories and the same "unseen" mask,
    # with a noise floor from two independent halves of a larger real sample
    def real_climate(mi_, p_, seen_, frames_):
        acc = [ClimateAcc() for _ in range(H // bins)]
        for k in range(H):
            pr, ty, tx, _ = world.decode(frames_[:, K + k], p_[:, K + k])
            um = ~seen_[torch.arange(len(mi_), device=DEV)[:, None, None], ty, tx]; um[:, 3, 3] = False
            acc[k // bins].add(world.true_ids(mi_, ty, tx), um)
        return acc
    ref = real_climate(mi, p, seen, real)
    g2 = torch.Generator().manual_seed(seed + 1000)
    m2, a2, p2 = world.random_walks(2 * C * 4, K + H, g2, map_pool=pool)
    s2 = seen_mask(world, m2, p2[:, :K])
    halves = []
    for h in range(2):
        sl = slice(h * C * 4, (h + 1) * C * 4)
        fr = world.render(m2[sl].repeat_interleave(K + H), p2[sl].reshape(-1, 2)).view(C * 4, K + H, 3, VIEW, VIEW)
        halves.append(real_climate(m2[sl], p2[sl], s2[sl], fr))
    floor = [halves[0][b].dist(halves[1][b]) for b in range(H // bins)]
    results = dict(floor=floor, bins=bins, C=C, M=M, H=H, arms={})
    for name, ck, cn, snap_every in arms:
        model, step = load(ck)
        t_start = time.time()
        gg = torch.Generator(device=DEV).manual_seed(seed + 7)
        ctx = real[:, :K].repeat_interleave(M, 0)
        acts = a.repeat_interleave(M, 0)
        ro = rollout(model, ctx, acts, ctx_noise=cn, snap=snapper if snap_every else None, snap_every=snap_every, gen=gg)
        miM = mi.repeat_interleave(M); pM = p.repeat_interleave(M, 0); seenM = seen.repeat_interleave(M, 0)
        clim = [ClimateAcc() for _ in range(H // bins)]
        acc_b, chance_b, off_b, spread_b, err_b, odo_b = [[] for _ in range(6)]
        surv_off = torch.full((C * M,), float(H), device=DEV); surv_uni = torch.full((C * M,), float(H), device=DEV)
        cal_pairs = []
        for k in range(H):
            f = ro[:, K + k]; pos = pM[:, K + k]
            sh = frame_shift(f, world.render(miM, pos))
            pr, ty, tx, dist = world.decode(f, pos - sh)
            tr = world.true_ids(miM, ty, tx)
            det = seenM[torch.arange(C * M, device=DEV)[:, None, None], ty, tx]; det[:, 3, 3] = False
            und = ~det; und[:, 3, 3] = False
            acc_b.append(((pr == tr) & det).sum().item() / max(1, det.sum().item()))
            cls = torch.bincount(tr[det].flatten(), minlength=1 + N_FLOOR + N_LANDMARK).float()
            chance_b.append(float(((cls / cls.sum().clamp(min=1)) ** 2).sum()))
            clim[k // bins].add(pr, und)
            cm = torch.ones_like(dist, dtype=torch.bool); cm[:, 3, 3] = False
            off = ((dist > thr) & cm).float().flatten(1).sum(1) / 48
            off_b.append(off.mean().item()); odo_b.append(sh.abs().amax(-1).float().mean().item())
            flat = pr.flatten(1); mode = torch.mode(flat, 1).values
            uni = (flat == mode[:, None]).float().mean(1) >= 0.95
            surv_off = torch.where((off > 0.3) & (surv_off == H), torch.full_like(surv_off, k), surv_off)
            surv_uni = torch.where(uni & (surv_uni == H), torch.full_like(surv_uni, k), surv_uni)
            # calibration at the true pose (members comparable cell-by-cell)
            prT, tyT, txT, _ = world.decode(f, pos)
            trT = world.true_ids(miM, tyT, txT)
            detT = seenM[torch.arange(C * M, device=DEV)[:, None, None], tyT, txT]; detT[:, 3, 3] = False
            P = prT.view(C, M, 49); Tt = trT.view(C, M, 49)[:, 0]; D = detT.view(C, M, 49)[:, 0]
            onehot = torch.nn.functional.one_hot(P, 1 + N_FLOOR + N_LANDMARK).float().mean(1)   # [C,49,K]
            modal = onehot.max(-1).values
            disagree = 1 - modal
            err = 1 - torch.gather(onehot, 2, Tt[..., None])[..., 0]
            spread_b.append(disagree[D].mean().item() if D.any() else float("nan"))
            err_b.append(err[D].mean().item() if D.any() else float("nan"))
            if k % 4 == 0 and D.any():
                cal_pairs.append((disagree[D].cpu().numpy(), err[D].cpu().numpy(), k))
        clim_d = [clim[b].dist(ref[b]) for b in range(H // bins)]
        dis = np.concatenate([x[0] for x in cal_pairs]); er = np.concatenate([x[1] for x in cal_pairs])
        try:
            from scipy.stats import spearmanr
            rho = float(spearmanr(dis, er).correlation)
        except Exception:
            rho = float(np.corrcoef(dis, er)[0, 1])
        bm = lambda xs: [float(np.nanmean(xs[i:i + bins])) for i in range(0, H, bins)]
        acc_bn, ch_bn = bm(acc_b), bm(chance_b)
        skill = [(x - c) / max(1e-6, 1 - c) for x, c in zip(acc_bn, ch_bn)]
        results["arms"][name] = dict(ckpt=str(ck), step=step, ctx_noise=cn, snap_every=snap_every,
                                     acc=acc_bn, chance=ch_bn, skill=skill, offman=bm(off_b), odo_px=bm(odo_b),
                                     spread=bm(spread_b), error=bm(err_b), spread_error_spearman=rho,
                                     climate=clim_d, climate_n=[c.n() for c in clim],
                                     survival_offman=surv_off.tolist(), survival_uniform=surv_uni.tolist(),
                                     seconds=round(time.time() - t_start))
        pat = [round(c["pat"], 4) for c in clim_d]; fl = [round(c["pat"], 4) for c in floor]
        print(f"{name}: skill {[round(s, 2) for s in skill]}\n   climate pat JSD {pat}\n   floor pat JSD   {fl}\n"
              f"   offman {[round(x, 3) for x in bm(off_b)]} spread/err rho {rho:.2f} | surv off med {surv_off.median().item():.0f} "
              f"| {time.time() - t_start:.0f}s", flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment dir (results written to DIR/results)")
    ap.add_argument("--train_args", default=None, help="args.json of the training run (world settings)")
    ap.add_argument("--section", choices=["contraction", "ensemble"], required=True)
    ap.add_argument("--ckpts", nargs="*", default=[])
    ap.add_argument("--arm", nargs=4, action="append", default=[], metavar=("NAME", "CKPT", "CTX_NOISE", "SNAP_EVERY"))
    ap.add_argument("--C", type=int, default=16); ap.add_argument("--M", type=int, default=8); ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--n_clips", type=int, default=256); ap.add_argument("--surv_n", type=int, default=32)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    exp = Path(args.exp); (exp / "results").mkdir(parents=True, exist_ok=True)
    targs = json.loads(Path(args.train_args).read_text()) if args.train_args else {"n_maps": 2000, "shading": 0.25}
    world, pool = setup_world(targs)
    if args.section == "contraction":
        r = section_contraction(world, pool, args.ckpts, n_clips=args.n_clips, surv_n=args.surv_n, surv_h=args.H)
    else:
        arms = [(n, c, float(cn), int(se)) for n, c, cn, se in args.arm]
        r = section_ensemble(world, pool, arms, C=args.C, M=args.M, H=args.H)
    (exp / "results" / f"{args.section}{args.tag}.json").write_text(json.dumps(r))
    print("done", flush=True)
