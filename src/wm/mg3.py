"""Matrix-Game 3.0 wrapper for a single 16 GB GPU.

Adapted from upstream `pipeline/inference_pipeline.py::MatrixGame3Pipeline.generate`
(vendor/Matrix-Game/Matrix-Game-3, commit 51913b2) with these changes:

- T5-XXL stays on the CPU and each prompt is encoded once (upstream moves the
  11 GB encoder to the GPU, which cannot coexist with the DiT in 16 GB);
- DiT linear q/k/v/o layers are int8 (upstream `--use_int8`), the rest bf16;
- the caller supplies the keyboard [F,6] / mouse [F,2] action sequence
  (upstream uses its fixed benchmark actions); camera poses are integrated
  from the actions exactly as upstream (`compute_all_poses_from_actions`);
- `use_memory=False` drops the retrieved memory latents (x_memory=None), the
  same path upstream uses for its classifier-free-guidance null branch, so
  memory can be ablated inside one model;
- distilled model only (3 steps, no CFG), synchronous VAE decode, frames are
  returned rather than written.

Frame schedule (upstream): first clip 57 frames, then +40 frames per
iteration (56-frame window, 16 frames of overlap), and 5 memory latents chosen
by camera field-of-view overlap (`select_memory_idx_fov`), the last pinned to
the first-frame latent.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MG3_DIR = ROOT / "vendor" / "Matrix-Game" / "Matrix-Game-3"
WEIGHTS = ROOT / "vendor" / "weights" / "Matrix-Game-3.0"


def frames_for_iterations(n_iter: int) -> int:
    return 57 + (n_iter - 1) * 40


def iterations_for_frames(n_frames: int) -> int:
    return max(1, int(np.ceil((n_frames - 57) / 40)) + 1)


class MG3:
    def __init__(self, height=704, width=1280, steps=3, vae_type="mg_lightvae_v2", int8_ffn=True,
                 offload_blocks=10):
        sys.path.insert(0, str(Path(__file__).parent / "shims"))
        sys.path.insert(0, str(MG3_DIR))
        cwd = os.getcwd(); os.chdir(MG3_DIR)
        try:
            from wan.configs import WAN_CONFIGS
            from wan.modules import WanModel
            from wan.modules.model import convert_model_to_int8
            from wan.modules.t5 import T5EncoderModel
            from pipeline.vae_config import load_vae
            self.cfg = cfg = WAN_CONFIGS["matrix_game3"]
            self.device = torch.device("cuda")
            self.dtype = torch.bfloat16
            self.t5 = T5EncoderModel(text_len=cfg.text_len, dtype=cfg.t5_dtype, device=torch.device("cpu"),
                                     checkpoint_path=str(WEIGHTS / cfg.t5_checkpoint),
                                     tokenizer_path=str(WEIGHTS / cfg.t5_tokenizer))
            mdir = str(WEIGHTS / "base_distilled_model")
            mcfg = WanModel.load_config(mdir)
            mcfg = mcfg[0] if isinstance(mcfg, tuple) else mcfg
            model = WanModel.from_pretrained(mdir, torch_dtype=torch.bfloat16, **mcfg)
            model = convert_model_to_int8(model, target_layers=["q", "k", "v", "o"])
            if int8_ffn:
                # Beyond upstream: int8 *every* large nn.Linear inside the DiT blocks
                # (FFN, camera injector, action module), same per-row absmax scheme
                # as upstream's q/k/v/o. Needed to leave room for 720p activations
                # in 16 GB. Small (<256-in) layers stay bf16.
                from wan.modules.model import Int8Linear as _I8

                class Int8Linear(_I8):
                    # upstream reads `ffn[0].weight.dtype` to choose the activation
                    # dtype; report bf16 so activations are not cast to int8
                    @property
                    def weight(self):
                        return type("W", (), {"dtype": torch.bfloat16})()

                def _q(mod):
                    for name, ch in mod.named_children():
                        if name == "action_model":
                            # keep bf16: its MLPs concatenate the small mouse/keyboard
                            # signal with large hidden states, and the int8 path's
                            # per-token activation quantization rounds the action away
                            # (first attempt: yaw input produced no camera rotation)
                            continue
                        if isinstance(ch, torch.nn.Linear) and ch.in_features >= 256 and ch.out_features >= 256:
                            q = Int8Linear(ch.in_features, ch.out_features, bias=ch.bias is not None,
                                           device=ch.weight.device)
                            q.from_float(ch.weight.data)
                            if ch.bias is not None:
                                q.bias.data.copy_(ch.bias.data.to(torch.bfloat16))
                            setattr(mod, name, q)
                        else:
                            _q(ch)
                _q(model.blocks)
            # FFNs act token-wise, so splitting the sequence is exact; it caps the
            # 14336-wide intermediate (and its int8/int32 temporaries) at 4k tokens.
            class _ChunkedFFN(torch.nn.Sequential):
                CHUNK = 4096
                def forward(self, x):
                    if x.dim() == 3 and x.shape[1] > self.CHUNK:
                        return torch.cat([super(_ChunkedFFN, self).forward(c) for c in x.split(self.CHUNK, 1)], 1)
                    return super().forward(x)
            for blk in model.blocks:
                blk.ffn.__class__ = _ChunkedFFN
            nb = sum(b.numel() * b.element_size() for b in list(model.parameters()) + list(model.buffers()))
            print(f"[mg3] DiT weights {nb/1e9:.2f} GB", flush=True)
            self.model = model.to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
            self._offload(offload_blocks)
            self.vae = load_vae(0, args=SimpleNamespace(vae_type=vae_type, ckpt_dir=str(WEIGHTS),
                                                        lightvae_pruning_rate=None))
            # the full Wan2.2 encoder (1.4 GB) only encodes the start image: keep it on CPU
            self.vae.encoder_model.to("cpu")
        finally:
            os.chdir(cwd)
        self.h, self.w, self.steps = height, width, steps
        self.vae_segment = 2  # upstream default 4; smaller decode segments for 16 GB
        self.vae_stride, self.patch = cfg.vae_stride, cfg.patch_size
        self._prompt_cache: dict = {}

    def _offload(self, n):
        """Keep the weights of the last n DiT blocks in pinned CPU memory and
        stream each block to the GPU just before it runs (exact; ~0.22 GB per
        block over PCIe). Upstream's fp32-autocast attention with fp64 RoPE needs
        ~6 GB of activations once the 5 memory latents join the window."""
        if n <= 0:
            return
        self.offloaded = []
        for blk in self.model.blocks[-n:]:
            ts = [t for t in list(blk.parameters()) + list(blk.buffers())]
            cpu = [t.data.to("cpu").pin_memory() for t in ts]
            for t, c in zip(ts, cpu):
                t.data = c
            def pre(mod, args, kwargs, ts=ts, cpu=cpu):
                for t, c in zip(ts, cpu):
                    t.data = c.to(self.device, non_blocking=True)
            def post(mod, args, kwargs, out, ts=ts, cpu=cpu):
                for t, c in zip(ts, cpu):
                    t.data = c
            blk.register_forward_pre_hook(pre, with_kwargs=True)
            blk.register_forward_hook(post, with_kwargs=True)
            self.offloaded.append(blk)
        torch.cuda.empty_cache()

    @torch.no_grad()
    def encode_prompt(self, text):
        if text not in self._prompt_cache:
            ctx = self.t5([text], device=torch.device("cpu"))
            self._prompt_cache[text] = [c.to(self.device, self.dtype) for c in ctx]
        return self._prompt_cache[text]

    @torch.no_grad()
    def rollout(self, pil_image, prompt, keyboard: np.ndarray, mouse: np.ndarray, seed=0,
                use_memory=True, return_memory_log=False):
        """keyboard [F,6], mouse [F,2]; F is rounded up to 57+40k. Returns uint8 [F,H,W,3]."""
        from einops import rearrange
        from utils.utils import compute_all_poses_from_actions, build_plucker_from_c2ws, build_plucker_from_pose
        from utils.cam_utils import (compute_relative_poses, select_memory_idx_fov, get_intrinsics,
                                     _interpolate_camera_poses_handedness, get_extrinsics)
        from utils.transform import get_video_transform
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        n_iter = iterations_for_frames(len(keyboard))
        num_frames = frames_for_iterations(n_iter)
        kb = np.zeros((num_frames, 6), np.float32); ms = np.zeros((num_frames, 2), np.float32)
        kb[:len(keyboard)] = keyboard; ms[:len(mouse)] = mouse
        kb_t = torch.from_numpy(kb); ms_t = torch.from_numpy(ms)
        poses = compute_all_poses_from_actions(kb_t, ms_t, first_pose=np.zeros(5))
        rot = np.concatenate([np.zeros((len(poses), 1)), poses[:, 3:5]], 1).tolist()
        extr = get_extrinsics(rot, poses[:, :3].tolist())
        keyboard_all = kb_t.to(self.device, self.dtype)[None]
        mouse_all = ms_t.to(self.device, self.dtype)[None]

        img = torch.from_numpy(np.array(pil_image.convert("RGB")))[None].permute(0, 3, 1, 2)
        img = get_video_transform(self.h, self.w, lambda x: 2. * x - 1.)(img).transpose(0, 1)[None]
        img = img.to(self.device, self.dtype)
        cond = self.encode_prompt(prompt)

        max_area = self.h * self.w
        ar = img.shape[-2] / img.shape[-1]
        lat_h = round(np.sqrt(max_area * ar) // self.vae_stride[1] // self.patch[1] * self.patch[1])
        lat_w = round(np.sqrt(max_area / ar) // self.vae_stride[2] // self.patch[2] * self.patch[2])
        th, tw = lat_h * self.vae_stride[1], lat_w * self.vae_stride[2]
        base_K = get_intrinsics(th, tw)
        self.vae.encoder_model.to(self.device)
        img_cond = self.vae.encode([img[0]])[0][None].to(self.device, self.dtype).contiguous()
        self.vae.encoder_model.to("cpu"); torch.cuda.empty_cache()

        clip_frame, first_clip_frame, past_frame = 56, 57, 16
        max_lat_f = (first_clip_frame - 1) // self.vae_stride[0] + 1
        max_seq_len = (max_lat_f + 5) * lat_h * lat_w // (self.patch[1] * self.patch[2])
        gen = torch.Generator(device=self.device).manual_seed(seed)
        lat_idx_of = lambda f: (f - 1) // 4 + 1
        align = lambda f: (f - 1) // 4 * 4 + 1 if f > 0 else 1
        vae_cache = [None] * 32
        latents_hist, videos, mem_log = [], [], []

        for ci in range(n_iter):
            first = ci == 0
            end = first_clip_frame if first else first_clip_frame + ci * (clip_frame - past_frame)
            start = 0 if first else end - clip_frame
            src_i = np.linspace(start, end - 1, first_clip_frame if first else clip_frame)
            tgt_len = (first_clip_frame - 1) // 4 + 1 if first else clip_frame // 4
            tgt_i = np.linspace(0 if first else start + 3, end - 1, tgt_len)
            plucker = build_plucker_from_c2ws(extr[start:end].to(self.device), src_i, tgt_i, framewise=True,
                                              base_K=base_K, target_h=th, target_w=tw, lat_h=lat_h, lat_w=lat_w)
            x_mem = t_mem = kb_mem = ms_mem = mem_lat_idx = None
            if not first and use_memory:
                base_idx = [end - o for o in range(1, 34, 8)]
                sel = select_memory_idx_fov(extr, start, base_idx, use_gpu=True)
                sel[-1] = 4
                mem_log.append(dict(iter=ci, start=start, end=end, selected=[int(s) for s in sel]))
                pl, mem_lat_idx = [], []
                for mi, ri in zip(sel, base_idx):
                    mem_lat_idx.append(lat_idx_of(mi))
                    ma = align(mi)
                    blk = extr[ma:ma + 4]
                    mp = _interpolate_camera_poses_handedness(
                        src_indices=np.linspace(ma, ma + 3, blk.shape[0]), src_rot_mat=blk[:, :3, :3].cpu().numpy(),
                        src_trans_vec=blk[:, :3, 3].cpu().numpy(), tgt_indices=np.array([ma + 3], np.float32))
                    rel = compute_relative_poses(torch.cat([extr[ri:ri + 1], mp], 0), framewise=False)[1:2]
                    pl.append(build_plucker_from_pose(rel.to(self.device), base_K=base_K, target_h=th, target_w=tw,
                                                      lat_h=lat_h, lat_w=lat_w))
                plucker = torch.cat(pl + [plucker], 2)
                x_mem = torch.cat(latents_hist, 2)[:, :, mem_lat_idx]
                ms_mem = torch.ones((1, len(sel), 2), device=self.device, dtype=self.dtype)
                kb_mem = -torch.ones((1, len(sel), 6), device=self.device, dtype=self.dtype)
                t_mem = x_mem.new_zeros((1, x_mem.shape[2] * x_mem.shape[3] * x_mem.shape[4] // 4))
            sched = FlowUniPCMultistepScheduler()
            sched.set_timesteps(self.steps, device=self.device, shift=self.cfg.sample_shift)
            ls, le = lat_idx_of(start), lat_idx_of(end)
            lat = torch.randn((1, 48, le - ls, img_cond.shape[-2], img_cond.shape[-1]), generator=gen,
                              device=self.device, dtype=self.dtype)
            lat = torch.cat([img_cond, lat[:, :, img_cond.shape[2]:]], 2)
            conds = dict(mouse_cond=mouse_all[:, start:end], keyboard_cond=keyboard_all[:, start:end], context=cond,
                         plucker_emb=plucker.to(self.device, self.dtype), x_memory=x_mem, timestep_memory=t_mem,
                         keyboard_cond_memory=kb_mem, mouse_cond_memory=ms_mem, memory_latent_idx=mem_lat_idx,
                         predict_latent_idx=(ls, le), fa_version=None)
            for t in sched.timesteps:
                ts = lat.new_full((lat.shape[2], lat.shape[3] * lat.shape[4] // 4), t)
                ts[:img_cond.shape[2]].zero_()
                pred = self.model(x=lat, t=ts.flatten()[None], seq_len=max_seq_len, **conds)
                lat = sched.step(pred, t, lat, return_dict=False)[0]
                lat = torch.cat([img_cond, lat[:, :, img_cond.shape[2]:]], 2)
            img_cond = lat[:, :, -4:]
            den = lat if first else lat[:, :, -10:]
            del pred, conds, plucker
            torch.cuda.empty_cache()  # DiT activations are fragmented; free before the VAE
            video, vae_cache = self.vae.stream_decode(den.to(dtype=self.vae.dtype), vae_cache, first_chunk=first,
                                                      segment_size=self.vae_segment, profiler={}, compile_decoder=False)
            videos.append(video.cpu())
            latents_hist.append(den)
        v = torch.cat(videos, 2)[0]
        v = ((rearrange(v, "C T H W -> T H W C").float() + 1) * 127.5).clamp(0, 255).byte().numpy()
        v = v[:len(keyboard)]
        return (v, mem_log) if return_memory_log else v
