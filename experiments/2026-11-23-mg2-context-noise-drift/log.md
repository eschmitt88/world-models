# log — mg2-context-noise-drift
2026-11-23 launched run.py (2 images x 2 seeds x cn{0,50,100,200,400}, 600 latents)
2026-11-23 first launch OOM at start (allocator env not applied, 5.2 GB reserved-unallocated); relaunched with PYTORCH_CUDA_ALLOC_CONF and PYTORCH_ALLOC_CONF both set
2026-11-23 second launch OOM: upstream keeps all decoded frames on GPU (~3 GB at 2,400 frames). src/wm/mg2.py now moves each decoded block to CPU. Third launch -> results/run3.log
