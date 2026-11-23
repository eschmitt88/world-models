"""Pixel-space, block-causal DiT world model trained with diffusion forcing.

Frames are 64x64 RGB in [-1,1], patchified 8x8 -> 64 tokens/frame. Each frame
carries its own noise level t_i (rectified flow: x_t = (1-t) x0 + t eps,
model predicts v = eps - x0). Attention is causal across frames and full
within a frame. Every token is conditioned (adaLN) on its frame's noise level
and on the action that led to that frame.

Training modes:
  "df": diffusion forcing, every frame gets an independent t ~ U(0,1)
  "tf": teacher forcing, implemented as a doubled sequence: a clean copy of
        every frame (t=0) plus a noisy copy; noisy frame i attends to clean
        frames < i and to itself, so the target always sees clean context.

Rollout: sliding window of `window` frames, recomputed each step (no KV
cache; small model). Context frames are fed at noise level `ctx_noise` (the
same knob as MG2's context_noise, here as a fraction of 1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from .gridworld import VIEW, N_ACTIONS

P = 8
TOK = (VIEW // P) ** 2            # 64 tokens per frame
PD = 3 * P * P                     # 192 values per patch

_flex = torch.compile(flex_attention, dynamic=False)


@dataclass
class Cfg:
    d: int = 384
    depth: int = 8
    heads: int = 6
    window: int = 32
    mode: str = "df"
    n_mem: int = 0        # retrieved memory frames prepended to the window (0 = none)
    pose: bool = False    # add a relative-pose embedding to every frame's conditioning


def patchify(x):   # [B,T,3,64,64] -> [B,T*64,192]
    B, T = x.shape[:2]
    x = x.view(B, T, 3, VIEW // P, P, VIEW // P, P).permute(0, 1, 3, 5, 2, 4, 6)
    return x.reshape(B, T * TOK, PD)


def unpatchify(x, T):
    B = x.shape[0]
    x = x.view(B, T, VIEW // P, VIEW // P, 3, P, P).permute(0, 1, 4, 2, 5, 3, 6)
    return x.reshape(B, T, 3, VIEW, VIEW)


def temb(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[..., None] * 1000 * f
    return torch.cat([a.sin(), a.cos()], -1)


class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.h = h
        s.n1 = nn.LayerNorm(d, elementwise_affine=False); s.n2 = nn.LayerNorm(d, elementwise_affine=False)
        s.qkv = nn.Linear(d, 3 * d); s.o = nn.Linear(d, d)
        s.qn = nn.LayerNorm(d // h); s.kn = nn.LayerNorm(d // h)
        s.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))
        s.ada = nn.Linear(d, 6 * d)
        nn.init.zeros_(s.ada.weight); nn.init.zeros_(s.ada.bias)

    def forward(s, x, c, mask):
        """x [B,L,D] tokens, c [B,T,D] per-frame conditioning (L = T*TOK).
        Modulations are computed per frame and broadcast over its tokens."""
        B, L, D = x.shape
        T = c.shape[1]
        a1, b1, g1, a2, b2, g2 = s.ada(c)[:, :, None].chunk(6, -1)      # each [B,T,1,D]
        fr = lambda z: z.view(B, T, L // T, D)
        y = (fr(s.n1(x)) * (1 + a1) + b1).view(B, L, D)
        q, k, v = s.qkv(y).view(B, L, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        q, k = s.qn(q).to(v.dtype), s.kn(k).to(v.dtype)
        y = _flex(q, k, v, block_mask=mask).transpose(1, 2).reshape(B, L, D)
        x = (fr(x) + g1 * fr(s.o(y))).view(B, L, D)
        h = (fr(s.n2(x)) * (1 + a2) + b2).view(B, L, D)
        return (fr(x) + g2 * fr(s.ff(h))).view(B, L, D)


class WorldDiT(nn.Module):
    def __init__(s, cfg: Cfg):
        super().__init__()
        s.cfg = cfg; d = cfg.d
        s.inp = nn.Linear(PD, d)
        s.pos_s = nn.Parameter(torch.randn(1, 1, TOK, d) * 0.02)
        s.pos_t = nn.Parameter(torch.randn(1, cfg.window, 1, d) * 0.02)
        s.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        s.act = nn.Embedding(N_ACTIONS + (1 if cfg.n_mem else 0), d)   # extra id = "memory frame"
        if cfg.n_mem:
            s.pos_mem = nn.Parameter(torch.randn(1, 1, 1, d) * 0.02)
        if cfg.pose:
            s.pose_mlp = nn.Sequential(nn.Linear(24, d), nn.SiLU(), nn.Linear(d, d))
        s.blocks = nn.ModuleList([Block(d, cfg.heads) for _ in range(cfg.depth)])
        s.nf = nn.LayerNorm(d, elementwise_affine=False)
        s.ada_f = nn.Linear(d, 2 * d); nn.init.zeros_(s.ada_f.weight); nn.init.zeros_(s.ada_f.bias)
        s.out = nn.Linear(d, PD); nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)
        s._masks = {}

    def mask(s, T, doubled=False):
        key = (T, doubled)
        if key not in s._masks:
            if not doubled:
                fid = torch.arange(T, device="cuda").repeat_interleave(TOK)
                s._masks[key] = create_block_mask(lambda b, h, q, kv: fid[kv] <= fid[q], None, None, T * TOK, T * TOK)
            else:
                # tokens [clean f0..fT-1 | noisy f0..fT-1]; clean causal among clean;
                # noisy i sees clean j<i and noisy i itself
                fid = torch.arange(T, device="cuda").repeat_interleave(TOK).repeat(2)
                isn = torch.cat([torch.zeros(T * TOK, dtype=torch.bool, device="cuda"), torch.ones(T * TOK, dtype=torch.bool, device="cuda")])
                def mm(b, h, q, kv):
                    clean_q = ~isn[q]
                    to_clean = ~isn[kv]
                    a = clean_q & to_clean & (fid[kv] <= fid[q])
                    bb = isn[q] & to_clean & (fid[kv] < fid[q])
                    c = isn[q] & isn[kv] & (fid[kv] == fid[q])
                    return a | bb | c
                s._masks[key] = create_block_mask(mm, None, None, 2 * T * TOK, 2 * T * TOK)
        return s._masks[key]

    def pose_feat(s, pose):
        """pose [B,T,2] relative pixels -> Fourier features [B,T,24]."""
        f = 2.0 ** torch.arange(6, device=pose.device) * math.pi / 64
        z = pose.float()[..., None] * f                          # [B,T,2,6]
        return torch.cat([z.sin(), z.cos()], -1).flatten(2)

    def forward(s, x, t, a, doubled=False, pose=None, n_mem=0):
        """x [B,T,3,64,64] noisy frames, t [B,T] noise levels, a [B,T] actions -> v [B,T,3,64,64].
        doubled: x/t/a are already [B,2T,...] laid out clean|noisy (TF training)."""
        B, T = x.shape[:2]
        Tw = T // 2 if doubled else T
        h = s.inp(patchify(x)).view(B, T, TOK, -1)
        tpos = s.pos_t[:, :Tw - n_mem]
        if n_mem: tpos = torch.cat([s.pos_mem.expand(1, n_mem, 1, -1), tpos], 1)
        if doubled: tpos = tpos.repeat(1, 2, 1, 1)
        h = (h + s.pos_s + tpos).view(B, T * TOK, -1)
        c = s.t_mlp(temb(t, s.cfg.d)) + s.act(a)                 # [B,T,d], per frame
        if pose is not None and s.cfg.pose:
            c = c + s.pose_mlp(s.pose_feat(pose))
        m = s.mask(Tw, doubled)
        for blk in s.blocks:
            h = blk(h, c, m)
        sh, sc = s.ada_f(c)[:, :, None].chunk(2, -1)
        h = (s.nf(h).view(B, T, TOK, -1) * (1 + sc) + sh).view(B, T * TOK, -1)
        return unpatchify(s.out(h), T)


def loss_fn(model, x0, a):
    """x0 [B,T,3,64,64] clean frames, a [B,T] actions."""
    B, T = x0.shape[:2]
    eps = torch.randn_like(x0)
    if model.cfg.mode == "df":
        t = torch.rand(B, T, device=x0.device)
        xt = (1 - t)[..., None, None, None] * x0 + t[..., None, None, None] * eps
        v = model(xt, t, a)
        return F.mse_loss(v, eps - x0)
    # teacher forcing: [clean | noisy]
    t = torch.rand(B, T, device=x0.device)
    xt = (1 - t)[..., None, None, None] * x0 + t[..., None, None, None] * eps
    x = torch.cat([x0, xt], 1); tt = torch.cat([torch.zeros_like(t), t], 1); aa = torch.cat([a, a], 1)
    v = model(x, tt, aa, doubled=True)[:, T:]
    return F.mse_loss(v, eps - x0)


@torch.no_grad()
def rollout(model, ctx, actions, steps=10, ctx_noise=0.0, gen=None):
    """ctx [B,K,3,64,64] real start frames; actions [B,K+N] (actions[:,i] led to frame i).
    Generates N frames autoregressively with a sliding window. Returns [B,K+N,...]."""
    W = model.cfg.window
    B, K = ctx.shape[:2]
    N = actions.shape[1] - K
    frames = [ctx[:, i] for i in range(K)]
    ts = torch.linspace(1, 0, steps + 1, device=ctx.device)
    for n in range(N):
        i = K + n
        c0 = max(0, i - (W - 1))
        cf = torch.stack(frames[c0:i], 1)                         # [B,C,...]
        C = cf.shape[1]
        if ctx_noise > 0:
            cf = (1 - ctx_noise) * cf + ctx_noise * torch.randn(cf.shape, device=cf.device, generator=gen)
        a = actions[:, c0:i + 1]
        x = torch.randn((B, 3, VIEW, VIEW), device=ctx.device, generator=gen)
        tc = torch.full((B, C), float(ctx_noise), device=ctx.device)
        for k in range(steps):
            t = ts[k].expand(B, 1)
            v = model(torch.cat([cf, x[:, None]], 1), torch.cat([tc, t], 1), a)[:, -1]
            x = x + (ts[k + 1] - ts[k]) * v
        frames.append(x.clamp(-1, 1))
    return torch.stack(frames, 1)


def retrieve(hist_pos, win_pos, n_mem, valid):
    """Pick n_mem history frames closest (L-inf, px) to any window pose.
    hist_pos [B,H,2], win_pos [B,W,2], valid [B,H] bool -> indices [B,n_mem]."""
    d = (hist_pos[:, :, None] - win_pos[:, None]).abs().amax(-1).amin(-1).float()   # [B,H]
    d = d.masked_fill(~valid, 1e9)
    return d.topk(n_mem, largest=False).indices


@torch.no_grad()
def rollout_mem(model, ctx, ctx_pos, actions, poses, steps=10, gen=None):
    """Like `rollout`, for a model with n_mem memory slots and pose conditioning.
    ctx [B,K,...] real frames, ctx_pos [B,K,2]; actions [B,K+N]; poses [B,K+N,2]
    (the pose each frame is rendered at; the oracle poses here). Memory frames are
    retrieved by pose from everything older than the window (real or generated)."""
    W, M = model.cfg.window, model.cfg.n_mem
    B, K = ctx.shape[:2]
    N = actions.shape[1] - K
    frames = [ctx[:, i] for i in range(K)]
    ts = torch.linspace(1, 0, steps + 1, device=ctx.device)
    mem_act = torch.full((B, M), N_ACTIONS, dtype=torch.long, device=ctx.device)
    for n in range(N):
        i = K + n
        c0 = max(0, i - (W - 1))
        cf = torch.stack(frames[c0:i], 1)
        a = actions[:, c0:i + 1]; ps = poses[:, c0:i + 1]
        if M and c0 >= M:
            idx = retrieve(poses[:, :c0], ps, M, torch.ones(B, c0, dtype=torch.bool, device=ctx.device))
            hist = torch.stack(frames[:c0], 1)
            mf = hist[torch.arange(B, device=ctx.device)[:, None], idx]
            mp = poses[:, :c0][torch.arange(B, device=ctx.device)[:, None], idx]
        else:   # not enough history yet: repeat the oldest context frame as memory
            mf = cf[:, :1].expand(B, M, *cf.shape[2:]) if M else cf[:, :0]
            mp = ps[:, :1].expand(B, M, 2) if M else ps[:, :0]
        ref = ps[:, :1]
        x = torch.randn((B, 3, VIEW, VIEW), device=ctx.device, generator=gen)
        tc = torch.zeros(B, M + cf.shape[1], device=ctx.device)
        for k in range(steps):
            tt = torch.cat([tc, ts[k].expand(B, 1)], 1)
            inp = torch.cat([mf, cf, x[:, None]], 1)
            v = model(inp, tt, torch.cat([mem_act, a], 1), pose=torch.cat([mp, ps], 1) - ref, n_mem=M)[:, -1]
            x = x + (ts[k + 1] - ts[k]) * v
        frames.append(x.clamp(-1, 1))
    return torch.stack(frames, 1)
