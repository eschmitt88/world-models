"""Matrix-Game 2.0 wrapper: load once, run many scripted action episodes.

Upstream code lives in vendor/Matrix-Game/Matrix-Game-2 (gitignored, commit
51913b2); weights in vendor/weights/Matrix-Game-2.0. This module adapts the
upstream `inference.py` so that

- the keyboard/mouse action sequence is supplied by the caller (upstream only
  runs its fixed benchmark actions),
- `context_noise` (the timestep at which a finished block is re-encoded into
  the KV cache, pipeline/causal_inference.py step 3.3) can be changed per run,
- the image condition is encoded once per (image, length) and reused.

Action conventions (from upstream utils/conditions.py):
  keyboard[f] = [W, S, A, D] in {0,1}
  mouse[f]    = [pitch, yaw]; upstream benchmark uses +/-0.1 per frame,
                yaw > 0 turns right.
Video frames F relate to latent frames L by F = 4 * (L - 1) + 1.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MG2_DIR = ROOT / "vendor" / "Matrix-Game" / "Matrix-Game-2"
WEIGHTS = ROOT / "vendor" / "weights" / "Matrix-Game-2.0"


def latents_for_frames(n_frames: int) -> int:
    """Smallest latent count L (multiple of 3, the block size) covering n_frames."""
    L = (n_frames - 1 + 3) // 4 + 1
    return int(np.ceil(L / 3) * 3)


class MG2:
    ENC_PREFIX = 21  # latent frames actually VAE-encoded for the image condition

    def __init__(self, config="configs/inference_yaml/inference_universal.yaml",
                 checkpoint="base_distilled_model/base_distill.safetensors",
                 compile_decoder: bool = False):
        sys.path.insert(0, str(MG2_DIR))
        sys.path.insert(0, str(Path(__file__).parent / "shims"))
        cwd = os.getcwd()
        os.chdir(MG2_DIR)  # upstream resolves model_config paths relative to its root
        try:
            from omegaconf import OmegaConf
            from safetensors.torch import load_file
            from torchvision.transforms import v2
            from pipeline import CausalInferencePipeline
            from wan.vae.wanx_vae import get_wanx_vae_wrapper
            from demo_utils.vae_block3 import VAEDecoderWrapper
            from utils.wan_wrapper import WanDiffusionWrapper

            self.device = torch.device("cuda")
            self.dtype = torch.bfloat16
            self.config = OmegaConf.load(config)
            self.mode = self.config.pop("mode")
            generator = WanDiffusionWrapper(**self.config.get("model_kwargs", {}), is_causal=True)
            dec = VAEDecoderWrapper()
            sd = torch.load(WEIGHTS / "Wan2.1_VAE.pth", map_location="cpu")
            dec.load_state_dict({k: v for k, v in sd.items() if "decoder." in k or "conv2" in k})
            dec.to(self.device, torch.float16).requires_grad_(False).eval()
            if compile_decoder:
                dec.compile(mode="max-autotune-no-cudagraphs")
            pipe = CausalInferencePipeline(self.config, generator=generator, vae_decoder=dec)
            pipe.generator.load_state_dict(load_file(str(WEIGHTS / checkpoint)))
            self.pipe = pipe.to(device=self.device, dtype=self.dtype)
            self.pipe.vae_decoder.to(torch.float16)
            vae = get_wanx_vae_wrapper(str(WEIGHTS), torch.float16)
            vae.requires_grad_(False).eval()
            self.vae = vae.to(self.device, self.dtype)
            self.local_attn_size = self.pipe.local_attn_size
            self.tf = v2.Compose([
                v2.Resize(size=(352, 640), antialias=True), v2.ToTensor(),
                v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
        finally:
            os.chdir(cwd)
        self._cond_cache: dict = {}

    @staticmethod
    def _resizecrop(image, th=352, tw=640):
        w, h = image.size
        if h / w > th / tw:
            nw, nh = w, int(w * th / tw)
        else:
            nh, nw = h, int(h * tw / th)
        l, t = (w - nw) / 2, (h - nh) / 2
        return image.crop((l, t, l + nw, t + nh))

    def load_image(self, img_path) -> np.ndarray:
        from diffusers.utils import load_image
        im = self._resizecrop(load_image(str(img_path))).resize((640, 352))
        return np.asarray(im.convert("RGB"))

    @torch.no_grad()
    def _cond(self, img_path, n_latent):
        key = (str(img_path), n_latent)
        if key in self._cond_cache:
            return self._cond_cache[key]
        from diffusers.utils import load_image
        self.vae.to(self.device, self.dtype)
        image = self._resizecrop(load_image(str(img_path)))
        image = self.tf(image)[None, :, None].to(self.device, self.dtype)
        # Upstream encodes [image, 4(L-1) black frames] in one pass, which needs
        # >16 GB for long rollouts. The VAE is causal in time, so the latents of
        # trailing black frames converge to a fixed value: encode a short prefix
        # and repeat its last latent. `self.pad_converge` records the max abs
        # difference between the last two prefix latents (should be ~0).
        k = min(n_latent, self.ENC_PREFIX)
        pad = torch.zeros_like(image).repeat(1, 1, 4 * (k - 1), 1, 1)
        img_cond = self.vae.encode(torch.cat([image, pad], 2), device=self.device,
                                   tiled=True, tile_size=[44, 80], tile_stride=[23, 38]).to(self.device)
        self.pad_converge = float((img_cond[:, :, -1] - img_cond[:, :, -2]).abs().max()) if k > 2 else None
        if n_latent > k:
            img_cond = torch.cat([img_cond, img_cond[:, :, -1:].repeat(1, 1, n_latent - k, 1, 1)], 2)
        mask = torch.ones_like(img_cond)
        mask[:, :, 1:] = 0
        cond_concat = torch.cat([mask[:, :4], img_cond], 1).to(self.dtype)
        vis = self.vae.clip.encode_video(image).to(self.dtype)
        # Encoder + CLIP are needed once per start image; park them on the CPU so
        # the DiT, KV cache and streaming decoder fit in 16 GB.
        self.vae.to("cpu", self.dtype)
        torch.cuda.empty_cache()
        self._cond_cache = {key: (cond_concat, vis)}  # keep only the latest
        return cond_concat, vis

    @torch.no_grad()
    def rollout(self, img_path, keyboard: np.ndarray, mouse: np.ndarray, seed: int = 0,
                context_noise: int = 0) -> np.ndarray:
        """Run one episode. keyboard [F,4], mouse [F,2] with F = 4(L-1)+1.

        Returns uint8 video [F, 352, 640, 3]."""
        F_ = keyboard.shape[0]
        assert (F_ - 1) % 4 == 0 and mouse.shape[0] == F_
        L = (F_ - 1) // 4 + 1
        assert L % self.pipe.num_frame_per_block == 0, f"latent count {L} must be a multiple of 3"
        cond_concat, vis = self._cond(img_path, L)
        torch.manual_seed(seed)
        noise = torch.randn([1, 16, L, 44, 80], device=self.device, dtype=self.dtype)
        cd = {
            "cond_concat": cond_concat,
            "visual_context": vis,
            "keyboard_cond": torch.as_tensor(keyboard, dtype=self.dtype, device=self.device)[None],
            "mouse_cond": torch.as_tensor(mouse, dtype=self.dtype, device=self.device)[None],
        }
        self.pipe.args.context_noise = int(context_noise)
        vids = self.pipe.inference(noise=noise, conditional_dict=cd, return_latents=False, mode=self.mode)
        v = torch.cat(vids, 1)[0].permute(0, 2, 3, 1)  # T H W C
        return ((v.float() + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()


def write_mp4(path, frames: np.ndarray, fps: int = 25, scale: float = 1.0, crf: int = 23):
    import imageio.v2 as imageio
    if scale != 1.0:
        import cv2
        h, w = frames.shape[1:3]
        frames = np.stack([cv2.resize(f, (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2),
                                      interpolation=cv2.INTER_AREA) for f in frames])
    imageio.mimwrite(str(path), list(frames), fps=fps, codec="libx264", quality=None,
                     ffmpeg_params=["-crf", str(crf), "-pix_fmt", "yuv420p", "-preset", "slow"])
