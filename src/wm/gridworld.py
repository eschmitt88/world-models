"""A persistent 2D tile world with an egocentric, partially observed pixel view.

Design (docs/decisions/0001): the *state* is a 2D tile map plus the agent's
pixel position; the *observation* is a 64x64 crop centred on the agent, so
places leave the view and can be revisited. Every rendered frame can be decoded
back to tile ids given the pose, which gives exact ground truth for drift and
memory metrics.

Everything is batched torch on the GPU so training data is generated in the
same process as the model:

    maps   = make_maps(n, size, seed, knobs)        # [n, S, S] int64 tile ids (+ shading)
    traj   = random_walks(maps, B, T, gen)           # actions [B,T], pos [B,T,2] (pixels)
    frames = render(maps, traj.map_idx, traj.pos)    # [B,T,3,64,64] in [-1,1]

Tiles: 0 = wall, 1..N_FLOOR = floor colours (one per room/region),
then landmark tiles (unique patterns, sparse). Agent moves STEP px per action
in 4 directions; walls block (the agent stays put, so action-integrated pose
and true pose disagree, which is a useful stress test).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

TILE = 8          # px per tile
VIEW = 64         # px, square egocentric view
STEP = 2          # px per move action
N_FLOOR = 6       # floor colour classes (region identity)
N_LANDMARK = 8    # unique landmark tile types
N_TILES = 1 + N_FLOOR + N_LANDMARK
ACTIONS = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}  # noop, up, down, left, right (dy, dx)
N_ACTIONS = len(ACTIONS)


@dataclass
class Knobs:
    size: int = 64                 # map side in tiles
    room_min: int = 4
    room_max: int = 12
    n_rooms: int = 14
    landmark_density: float = 0.02  # fraction of floor tiles that become landmarks
    shading: float = 0.25           # amplitude of smooth multiplicative lighting (0 = crisp tiles)
    shading_scale: int = 8          # tiles per lighting-noise cell


def _atlas(seed: int = 0) -> torch.Tensor:
    """[N_TILES, 3, 8, 8] tile textures in [0,1]. Walls dark brick, floors flat
    colours with faint grain, landmarks high-contrast glyphs."""
    rng = np.random.default_rng(seed)
    a = np.zeros((N_TILES, 3, TILE, TILE), np.float32)
    # wall: dark grey brick
    a[0] = 0.18
    a[0, :, 3, :] = 0.08; a[0, :, 7, :] = 0.08; a[0, :, :4, 3] = 0.08; a[0, :, 4:, 7] = 0.08
    floor_cols = [(0.80, 0.72, 0.55), (0.55, 0.75, 0.55), (0.55, 0.65, 0.85),
                  (0.85, 0.60, 0.60), (0.75, 0.75, 0.75), (0.70, 0.60, 0.85)]
    for i, c in enumerate(floor_cols[:N_FLOOR]):
        grain = rng.normal(0, 0.025, (TILE, TILE)).astype(np.float32)
        for ch in range(3):
            a[1 + i, ch] = np.clip(c[ch] + grain, 0, 1)
    glyph_cols = [(0.95, 0.85, 0.1), (0.1, 0.8, 0.9), (0.9, 0.2, 0.6), (0.2, 0.9, 0.3),
                  (1.0, 0.5, 0.1), (0.5, 0.3, 1.0), (1.0, 1.0, 1.0), (0.9, 0.1, 0.1)]
    for j in range(N_LANDMARK):
        k = 1 + N_FLOOR + j
        bg = np.full((TILE, TILE), 0.12, np.float32)
        m = np.zeros((TILE, TILE), bool)
        if j % 4 == 0: m[2:6, 2:6] = True                      # square
        elif j % 4 == 1: m[np.arange(8), np.arange(8)] = True; m[np.arange(8), 7 - np.arange(8)] = True  # X
        elif j % 4 == 2: m[3:5, :] = True; m[:, 3:5] = True      # plus
        else: m[1:7, 1:7] = True; m[2:6, 2:6] = False            # ring
        if j >= 4: m = m | np.roll(m, 1, 0)                     # bolder variant
        for ch in range(3):
            a[k, ch] = np.where(m, glyph_cols[j][ch], bg)
    return torch.from_numpy(a)


def _one_map(rng: np.random.Generator, kn: Knobs) -> np.ndarray:
    S = kn.size
    m = np.zeros((S, S), np.int64)
    rooms = []
    for _ in range(kn.n_rooms * 4):
        if len(rooms) >= kn.n_rooms: break
        h, w = rng.integers(kn.room_min, kn.room_max + 1, 2)
        y, x = rng.integers(1, S - h - 1), rng.integers(1, S - w - 1)
        if any(y < r[0] + r[2] + 1 and r[0] < y + h + 1 and x < r[1] + r[3] + 1 and r[1] < x + w + 1 for r in rooms):
            continue
        rooms.append((y, x, h, w))
        m[y:y + h, x:x + w] = 1 + rng.integers(N_FLOOR)
    # connect rooms in sequence with L-shaped corridors (corridor colour = source room's)
    order = rng.permutation(len(rooms))
    for a, b in zip(order[:-1], order[1:]):
        ya, xa = rooms[a][0] + rooms[a][2] // 2, rooms[a][1] + rooms[a][3] // 2
        yb, xb = rooms[b][0] + rooms[b][2] // 2, rooms[b][1] + rooms[b][3] // 2
        c = m[ya, xa]
        for x in range(min(xa, xb), max(xa, xb) + 1):
            if m[ya, x] == 0: m[ya, x] = c
            if m[ya + 1, x] == 0: m[ya + 1, x] = c
        for y in range(min(ya, yb), max(ya, yb) + 1):
            if m[y, xb] == 0: m[y, xb] = c
            if m[y, xb + 1] == 0: m[y, xb + 1] = c
    floor = np.argwhere(m > 0)
    n_lm = int(kn.landmark_density * len(floor))
    if n_lm:
        pick = floor[rng.choice(len(floor), n_lm, replace=False)]
        m[pick[:, 0], pick[:, 1]] = 1 + N_FLOOR + rng.integers(N_LANDMARK, size=n_lm)
    return m


class World:
    """A bank of maps on the GPU plus a renderer."""

    def __init__(self, n_maps: int, seed: int, knobs: Knobs | None = None, device="cuda"):
        self.kn = kn = knobs or Knobs()
        self.device = device
        rng = np.random.default_rng(seed)
        self.maps = torch.from_numpy(np.stack([_one_map(rng, kn) for _ in range(n_maps)])).to(device)
        self.atlas = _atlas(0).to(device)                       # [N_TILES,3,8,8]
        # smooth lighting: low-res noise, bilinearly upsampled to pixel res lazily per crop
        g = kn.size // kn.shading_scale + 2
        self.light = torch.from_numpy(rng.uniform(-1, 1, (n_maps, 1, g, g)).astype(np.float32)).to(device)
        self.S = kn.size
        # padded table of floor tile coords per map, for fast vectorised start sampling
        fl = [torch.nonzero(mp > 0) for mp in self.maps]
        self.n_floor = torch.tensor([len(f) for f in fl], device=device)
        L = int(self.n_floor.max())
        self.floor_tab = torch.stack([torch.cat([f, f[:1].expand(L - len(f), 2)]) for f in fl])

    # ---------- trajectories ----------
    def floor_starts(self, map_idx: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """Random floor tile centre per map index -> pixel pos [B,2]."""
        u = torch.rand(len(map_idx), generator=gen, device="cpu").to(self.device)
        j = (u * self.n_floor[map_idx]).long().clamp(max=self.floor_tab.shape[1] - 1)
        t = self.floor_tab[map_idx, j]
        return t * TILE + TILE // 2

    def passable(self, map_idx: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Is the agent's 4x4 px body at pos inside floor? pos [B,2] pixels."""
        ok = torch.ones(len(pos), dtype=torch.bool, device=self.device)
        for dy in (-2, 1):
            for dx in (-2, 1):
                ty = ((pos[:, 0] + dy) // TILE).clamp(0, self.S - 1)
                tx = ((pos[:, 1] + dx) // TILE).clamp(0, self.S - 1)
                ok &= self.maps[map_idx, ty, tx] > 0
        return ok

    def step(self, map_idx, pos, act):
        d = torch.tensor([ACTIONS[i] for i in range(N_ACTIONS)], device=self.device)[act] * STEP
        new = pos + d
        ok = self.passable(map_idx, new)
        return torch.where(ok[:, None], new, pos), ok

    def random_walks(self, B: int, T: int, gen: torch.Generator, map_pool: torch.Tensor | None = None,
                     p_turn: float = 0.08, p_noop: float = 0.05):
        """Momentum random walks. Returns map_idx [B], actions [B,T], pos [B,T,2]
        (pos[:,t] is the pose *after* actions[:,t]; frame t renders pos[:,t]).
        actions[:,0] is a no-op so frame 0 is the start view."""
        pool = map_pool if map_pool is not None else torch.arange(len(self.maps), device=self.device)
        mi = pool[torch.randint(len(pool), (B,), generator=gen, device="cpu").to(self.device)]
        pos = self.floor_starts(mi, gen)
        cur = torch.randint(1, 5, (B,), generator=gen, device="cpu").to(self.device)
        acts, poss = [], []
        for t in range(T):
            if t == 0:
                a = torch.zeros(B, dtype=torch.long, device=self.device)
            else:
                r = torch.rand(B, generator=gen, device="cpu").to(self.device)
                newdir = torch.randint(1, 5, (B,), generator=gen, device="cpu").to(self.device)
                cur = torch.where(r < p_turn, newdir, cur)
                a = torch.where(torch.rand(B, generator=gen, device="cpu").to(self.device) < p_noop, torch.zeros_like(cur), cur)
            pos, ok = self.step(mi, pos, a)
            # bounce: if blocked, pick a new direction next step
            cur = torch.where(ok | (a == 0), cur, torch.randint(1, 5, (B,), generator=gen, device="cpu").to(self.device))
            acts.append(a); poss.append(pos)
        return mi, torch.stack(acts, 1), torch.stack(poss, 1)

    # ---------- rendering ----------
    def render(self, map_idx: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """map_idx [N], pos [N,2] (pixel centre of the view) -> frames [N,3,64,64] in [-1,1]."""
        N = len(pos)
        nt = VIEW // TILE + 1                                    # 9 tiles cover any 64px window
        top = pos - VIEW // 2                                    # top-left pixel of the view
        t0 = torch.div(top, TILE, rounding_mode="floor")         # top-left tile
        off = top - t0 * TILE                                    # sub-tile offset 0..7
        ar = torch.arange(nt, device=self.device)
        ty = (t0[:, 0:1] + ar[None]).clamp(0, self.S - 1)        # [N,nt]
        tx = (t0[:, 1:2] + ar[None]).clamp(0, self.S - 1)
        ids = self.maps[map_idx[:, None, None], ty[:, :, None], tx[:, None, :]]  # [N,nt,nt]
        # out-of-map -> wall
        oob = ((t0[:, 0:1] + ar[None]) < 0) | ((t0[:, 0:1] + ar[None]) >= self.S)
        oobx = ((t0[:, 1:2] + ar[None]) < 0) | ((t0[:, 1:2] + ar[None]) >= self.S)
        ids = ids.masked_fill(oob[:, :, None] | oobx[:, None, :], 0)
        tiles = self.atlas[ids]                                  # [N,nt,nt,3,8,8]
        img = tiles.permute(0, 3, 1, 4, 2, 5).reshape(N, 3, nt * TILE, nt * TILE)
        # crop sub-tile offset via gather on a 72x72 canvas
        iy = (off[:, 0:1] + torch.arange(VIEW, device=self.device)[None])  # [N,64]
        ix = (off[:, 1:2] + torch.arange(VIEW, device=self.device)[None])
        img = img[torch.arange(N, device=self.device)[:, None, None, None], torch.arange(3, device=self.device)[None, :, None, None],
                  iy[:, None, :, None], ix[:, None, None, :]]
        if self.kn.shading > 0:
            # lighting sampled in map-pixel coordinates via grid_sample on the per-map noise grid
            Sp = self.S * TILE
            yy = (top[:, 0:1].float() + torch.arange(VIEW, device=self.device)[None] + 0.5) / Sp
            xx = (top[:, 1:2].float() + torch.arange(VIEW, device=self.device)[None] + 0.5) / Sp
            g = self.light.shape[-1]
            # map [0,1] map-coords onto the inner (g-2) cells of the noise grid
            sy = ((yy * (g - 2) + 1) / (g - 1)) * 2 - 1
            sx = ((xx * (g - 2) + 1) / (g - 1)) * 2 - 1
            grid = torch.stack([sx[:, None, :].expand(N, VIEW, VIEW), sy[:, :, None].expand(N, VIEW, VIEW)], -1)
            L = F.grid_sample(self.light[map_idx], grid, mode="bicubic", align_corners=True)  # [N,1,64,64]
            img = img * (1 + self.kn.shading * L)
        # agent: small fixed marker at the centre (the view is egocentric)
        c = VIEW // 2
        img[:, :, c - 2:c + 2, c - 2:c + 2] = torch.tensor([1.0, 1.0, 1.0], device=self.device)[None, :, None, None]
        return (img.clamp(0, 1) * 2 - 1)

    # ---------- decoding (ground truth for generated frames) ----------
    def decode(self, frames: torch.Tensor, pos: torch.Tensor):
        """Classify each fully visible tile cell of `frames` [N,3,64,64] in [-1,1]
        given the pose `pos` [N,2] it should have been rendered at. Returns
        (pred_ids [N,7,7], true_ids [N,7,7], dist [N,7,7]); dist is the
        nearest-atlas distance after per-cell brightness normalisation (large =
        off-manifold). The central cell (agent marker) is excluded by callers."""
        N = len(frames)
        top = pos - VIEW // 2
        t0 = torch.div(top, TILE, rounding_mode="floor")
        off = top - t0 * TILE
        first = torch.where(off == 0, torch.zeros_like(off), TILE - off)  # pixel of first fully visible tile
        nfull = 7
        x = (frames.clamp(-1, 1) + 1) / 2
        cells, tys, txs = [], [], []
        for i in range(nfull):
            for j in range(nfull):
                py = first[:, 0] + i * TILE; px = first[:, 1] + j * TILE
                iy = py[:, None] + torch.arange(TILE, device=self.device)[None]
                ix = px[:, None] + torch.arange(TILE, device=self.device)[None]
                cell = x[torch.arange(N, device=self.device)[:, None, None, None], torch.arange(3, device=self.device)[None, :, None, None],
                         iy[:, None, :, None], ix[:, None, None, :]]
                cells.append(cell)
                tys.append(t0[:, 0] + (first[:, 0] > 0).long() + i)
                txs.append(t0[:, 1] + (first[:, 1] > 0).long() + j)
        cells = torch.stack(cells, 1)                            # [N,49,3,8,8]
        # brightness-normalise (shading is multiplicative and smooth)
        def nrm(z):
            return z / (z.mean(dim=(-3, -2, -1), keepdim=True) + 1e-3)
        A = nrm(self.atlas)[None, None]                          # [1,1,K,3,8,8]
        d = ((nrm(cells)[:, :, None] - A) ** 2).mean(dim=(-3, -2, -1))  # [N,49,K]
        dist, pred = d.min(-1)
        ty = torch.stack(tys, 1).clamp(0, self.S - 1); tx = torch.stack(txs, 1).clamp(0, self.S - 1)
        return pred.view(N, nfull, nfull), ty.view(N, nfull, nfull), tx.view(N, nfull, nfull), dist.view(N, nfull, nfull)

    def true_ids(self, map_idx, ty, tx):
        return self.maps[map_idx[:, None, None], ty, tx]



def _inverse_action(a: torch.Tensor) -> torch.Tensor:
    return torch.tensor([0, 2, 1, 4, 3], device=a.device)[a]


def simulate(world: "World", mi, start_pos, acts):
    """True poses from a start pose and an action sequence (frame t renders pos[:, t])."""
    pos = start_pos; out = []
    for t in range(acts.shape[1]):
        pos, _ = world.step(mi, pos, acts[:, t]); out.append(pos)
    return torch.stack(out, 1)


def revisit_walks(world: "World", B: int, T: int, gen: torch.Generator, map_pool=None,
                  out_min: int = 16, out_max: int = 96, p_turn: float = 0.08):
    """Walk out for L ~ U[out_min, out_max] steps, retrace by inverting the actions
    in reverse order, then carry on with the original walk's actions. This gives
    revisits with a controlled gap (frame t and frame 2L - t share a pose when no
    wall blocked the way out). Poses are always re-simulated, so a blocked step on the
    way out makes the return approximate. Returns map_idx [B], actions [B,T],
    pos [B,T,2], L [B]."""
    mi, a, p = world.random_walks(B, T, gen, map_pool=map_pool, p_turn=p_turn, p_noop=0.05)
    moved = torch.ones_like(a, dtype=torch.bool)
    moved[:, 1:] = (p[:, 1:] != p[:, :-1]).any(-1)
    a = torch.where(moved, a, torch.zeros_like(a))             # blocked steps become no-ops, so retracing is exact
    L = torch.randint(out_min, out_max + 1, (B,), generator=gen, device="cpu").to(world.device)
    t = torch.arange(T, device=world.device)[None]
    mirror = (2 * L[:, None] + 1 - t).clamp(1, T - 1)          # action that led *to* frame 2L+1-t
    in_back = (t > L[:, None]) & (t <= 2 * L[:, None])
    acts = torch.where(in_back, _inverse_action(torch.gather(a, 1, mirror)), a)
    acts[:, 0] = 0
    return mi, acts, simulate(world, mi, p[:, 0], acts), L


def frame_shift(gen: torch.Tensor, ref: torch.Tensor, max_shift: int = 24):
    """Integer translation (dy, dx) such that gen(y, x) ~ ref(y - dy, x - dx),
    via phase correlation on grey frames. gen, ref [N,3,64,64] in [-1,1].
    A frame rendered at true pose p but drawn as if from pose p - s has shift s,
    so `pos - shift` is the pose the model actually drew."""
    g = gen.float().mean(1); r = ref.float().mean(1)
    g = g - g.mean((-2, -1), keepdim=True); r = r - r.mean((-2, -1), keepdim=True)
    Fg, Fr = torch.fft.rfft2(g), torch.fft.rfft2(r)
    R = Fg * Fr.conj(); R = R / (R.abs() + 1e-6)
    c = torch.fft.irfft2(R, s=g.shape[-2:])                   # [N,64,64]
    H, W = c.shape[-2:]
    ys = torch.arange(H, device=c.device); ys = torch.where(ys > H // 2, ys - H, ys)
    xs = torch.arange(W, device=c.device); xs = torch.where(xs > W // 2, xs - W, xs)
    ok = (ys.abs()[:, None] <= max_shift) & (xs.abs()[None, :] <= max_shift)
    c = c.masked_fill(~ok[None], -1e9)
    idx = c.flatten(1).argmax(1)
    return torch.stack([ys[idx // W], xs[idx % W]], 1)
