"""Build small side-by-side comparison clips for reports."""
import numpy as np, cv2, imageio.v2 as iio

def read(path):
    return np.stack([f for f in iio.get_reader(str(path))])

def label(img, text, sub=None):
    h, w = img.shape[:2]
    bar = np.full((26, w, 3), 18, np.uint8)
    cv2.putText(bar, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    if sub:
        (tw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(bar, sub, (w - tw - 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 200, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], 0)

def resize(f, w=320):
    h = int(round(f.shape[0] * w / f.shape[1] / 2) * 2)
    return cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)

def phase_text(i, lead, T):
    if i < lead: return "hold"
    if i < lead + T: return "turn right"
    if i < lead + 2 * T: return "turn back"
    return "back at start"

def outback_clip(video, lead, T, title, w=320):
    ref = resize(video[lead], w)
    frames = []
    for i, f in enumerate(video):
        a = label(ref, "start view")
        b = label(resize(f, w), title, phase_text(i, lead, T))
        frames.append(np.concatenate([a, np.full((a.shape[0], 4, 3), 0, np.uint8), b], 1))
    return np.stack(frames)

def write(path, frames, fps=25, crf=26):
    iio.mimwrite(str(path), list(frames), fps=fps, codec="libx264", quality=None,
                 ffmpeg_params=["-crf", str(crf), "-pix_fmt", "yuv420p", "-preset", "slow", "-movflags", "+faststart"])
