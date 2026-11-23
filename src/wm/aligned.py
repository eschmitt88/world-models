"""Shift-aligned LPIPS between two frames.

Two rollouts that return to "the same heading" can land a few degrees apart
(MG3 integrates +yaw and -yaw at slightly different pixel speeds), and plain
LPIPS then scores a correct memory as a miss. Here frame b is registered to
frame a by global translation (phase correlation on grey), both are cropped to
their overlap, and LPIPS is computed on that crop. Shifts larger than
`max_frac` of the width are treated as failed registration (no overlap worth
scoring) and fall back to unaligned LPIPS."""
import numpy as np, cv2, torch

def shift(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    win = cv2.createHanningWindow(ga.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ga, gb, win)
    return dx, dy, resp

def aligned_lpips(lp, a, b, dev="cuda", max_frac=0.35):
    h, w = a.shape[:2]
    dx, dy, resp = shift(a, b)
    ix, iy = int(round(dx)), int(round(dy))
    ok = abs(ix) < max_frac * w and abs(iy) < max_frac * h
    if ok:
        a_c = a[max(0, iy):h + min(0, iy), max(0, ix):w + min(0, ix)]
        b_c = b[max(0, -iy):h + min(0, -iy), max(0, -ix):w + min(0, -ix)]
    else:
        a_c, b_c = a, b
    t = lambda f: torch.nn.functional.interpolate(
        torch.from_numpy(np.ascontiguousarray(f)).to(dev).permute(2, 0, 1)[None].float() / 127.5 - 1,
        scale_factor=0.5, mode="area")
    with torch.no_grad():
        return float(lp(t(a_c), t(b_c))), dict(dx=dx, dy=dy, resp=resp, aligned=ok)
