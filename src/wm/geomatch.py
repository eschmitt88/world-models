"""Geometric place-recognition score between two frames: SIFT keypoints,
Lowe ratio test (0.75), then RANSAC homography (5 px). Returns the inlier
count. Robust to the heading offsets and texture re-synthesis that make
pixel metrics (LPIPS, colour) score a remembered place as a miss; a
different place yields few geometrically consistent matches."""
import numpy as np, cv2

_sift = cv2.SIFT_create(nfeatures=2000)
_bf = cv2.BFMatcher(cv2.NORM_L2)

def inliers(a, b):
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (a, b))
    ka, da = _sift.detectAndCompute(ga, None)
    kb, db = _sift.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    good = [m for m, n in (p for p in _bf.knnMatch(da, db, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return len(good) // 4
    pa = np.float32([ka[m.queryIdx].pt for m in good]); pb = np.float32([kb[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0
