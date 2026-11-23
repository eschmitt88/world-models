---
source_url: https://worldlabs.ai/blog/atlas
fetched: 2026-11-23
title: "Atlas: A World Model for Spatial Intelligence (World Labs, 2026-09-01)"
note: "Condensed summary produced by a fetch tool (the page could not be saved verbatim); it is not the page text. Raw snapshot; do not edit. Re-ingest from the page if exact wording is needed."
---

## Architecture
- Atlas is a "multimodal autoregressive diffusion transformer", pretrained from scratch to handle text, images, video and 3D.
- All inputs are combined into a shared "spatial context". Each image is anchored at a 3D position so generated content stays consistent in 3D.
- Inputs: text, images, camera poses, depth maps. Videos are treated as sequences of images. Each image and depth map is conditioned on an explicit camera pose.
- It is a rectified-flow (latent diffusion) model, so quality can be traded against speed by changing the number of denoising steps.
- It borrows from LLMs (KV caching, serving optimisations) and from video and image models (diffusion distillation, classifier-free guidance, VAE design).

## Capabilities
- Camera-controlled generation: images and video from one to six inputs with precise camera paths, up to one minute of 1440p video; can interpolate between unrelated images placed in the spatial context.
- Spatial reconstruction: from sparse inputs (two or three images can be enough, over a hundred can be used); outputs novel views plus explicit 3D as point clouds or 3D Gaussian splats, the same representation as Marble.
- Space-time simulation: multi-camera footage (about three to five views) reframed into "bullet time" shots; Real-to-Sim workflows generating RGB and depth sensor data for robot navigation and manipulation.
- Image generation: images and 360-degree panoramas from text or image prompts (stated not to be the primary focus).

## Training and scaling
- Performance "improves with increased training compute"; a series of models of increasing size was trained; "each new level of compute unlocked new model capabilities". No dataset sizes, compute figures or parameter counts are given.

## Benchmarks (company-reported)
- Camera-controlled generation: third-party raters preferred Atlas over five video and image models roughly 75% to 94% of the time; baselines received text descriptions of camera motion because they cannot take poses.
- 3D reconstruction: mean absolute-relative pointmap error across DTU, ETH3D, KITTI, NRGBD, 7-Scenes, T&T, ScanNet; Atlas reports the lowest average (about 8.6 against about 9.7 for VGGT, the next best). Chart values are approximate.

## Availability
- Available through World Labs web apps and a developer API.
- No separate technical report or arXiv paper is linked; the only citation offered is a BibTeX entry for the blog post itself.
