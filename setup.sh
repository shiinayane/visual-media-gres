#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproducible setup for GRES/ReLA on the NVIDIA GB10 (Grace-Blackwell, aarch64,
# CUDA 13) server. The original ReLA stack (torch 1.11 / cu118 / detectron2 0.6)
# does NOT run on Blackwell; this script builds a working modern stack instead.
# See report Section 4 for the rationale behind each step.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

PY="${PY:-python3.11}"
echo ">> using python: $($PY --version)"

# 1) venv ---------------------------------------------------------------------
$PY -m venv .venv
. .venv/bin/activate
pip install --upgrade pip setuptools wheel ninja

# 2) PyTorch with CUDA for Blackwell -----------------------------------------
#    The default PyPI aarch64 wheel is CPU-only; pull the cu128 build.
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 3) third-party source: ReLA, gRefCOCO, detectron2 --------------------------
mkdir -p third_party && cd third_party
[ -d ReLA ]       || git clone https://github.com/henghuiding/ReLA.git
[ -d gRefCOCO ]   || git clone https://github.com/henghuiding/gRefCOCO.git
[ -d detectron2 ] || git clone https://github.com/facebookresearch/detectron2.git
cd "$ROOT"

# 3a) detectron2: build CPU-only C++ ops (FORCE the CppExtension path so we do
#     NOT compile CUDA kernels with the mismatched system toolkit, nvcc 13 vs the
#     torch cu128 build). Hiding the GPU makes torch.cuda.is_available() False,
#     which is exactly the switch detectron2/setup.py keys off.
( cd third_party/detectron2 && CUDA_VISIBLE_DEVICES="" FORCE_CUDA=0 \
    pip install -e . --no-build-isolation )

# 3b) patch ReLA's MSDeformAttn so a missing custom CUDA op falls back to the
#     pure-PyTorch (GPU grid_sample) implementation instead of hard-failing.
git apply patches/msdeformattn_fallback.patch || \
    echo "   (patch already applied or applied manually -- see patches/README.md)"

echo ">> setup complete. Activate with:  source .venv/bin/activate"
echo ">> next: download data + weights (see README 'Data & weights')."
