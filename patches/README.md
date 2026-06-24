# Patches applied to `third_party/ReLA`

`setup.sh` applies `msdeformattn_fallback.patch` after cloning ReLA. It contains **two** hunks,
both forced by running on the NVIDIA GB10 (Blackwell / aarch64 / CUDA 13, compute `sm_121`):

1. **`ops/functions/ms_deform_attn_func.py` — MSDeformAttn pure-PyTorch fallback.**
   The hand-written `MultiScaleDeformableAttention` CUDA op cannot be compiled (system nvcc 13.0
   vs the torch cu128/12.8 build). Instead of hard-failing on the missing import, fall back to
   ReLA's own `ms_deform_attn_core_pytorch`, which runs on GPU via `F.grid_sample` (the reference
   implementation of the op; numerically close, slightly slower).

2. **`modeling/pixel_decoder/msdeformattn.py` — `sm_121` nvrtc fix.**
   `spatial_shapes.prod(1)` on an int64 tensor falls to the PyTorch nvrtc *jiterator*, whose cu128
   nvrtc rejects `-arch=compute_121` (*"invalid value for --gpu-architecture"*). Replace it with the
   identical elementwise product `spatial_shapes[:,0]*spatial_shapes[:,1]`, which uses a precompiled
   kernel. **Without this hunk a fresh clone crashes on the first forward pass on Blackwell.**

## Apply manually
```bash
git apply patches/msdeformattn_fallback.patch     # run from the repo root (paths are third_party/ReLA/...)
```
Check it applies first with `git apply --check patches/msdeformattn_fallback.patch`.
