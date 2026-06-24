# GRES / ReLA on gRefCOCO — failure analysis + an abstain/clarify detector

Final assignment for **Visual Media (映像メディア学), UTokyo**.

Target paper: **GRES: Generalized Referring Expression Segmentation** (network **ReLA**),
Liu, Ding, Jiang, *CVPR 2023 (Highlight)* — [arXiv:2306.00968](https://arxiv.org/abs/2306.00968).
Official code: <https://github.com/henghuiding/ReLA> · dataset: <https://github.com/henghuiding/gRefCOCO>.

This repo reproduces the ReLA baseline on gRefCOCO, quantifies **two failure modes**
(no-target hallucination; multi-target under-segmentation), and adds a **post-hoc
abstain-or-clarify detector** on the frozen model — turning GRES's no-target / low-confidence
signals into an explicit "ask or abstain instead of guessing" policy for human-robot use.

> Report: `report/report.md` (export to PDF before submission). Repo URL: **TODO (human)**.

## Why this setup is non-trivial
The paper targets torch-1.11 / CUDA-11.8 / detectron2-0.6. This server is an
**NVIDIA GB10 (Grace-Blackwell, aarch64), CUDA 13, compute sm_121** — that old stack will
not run. The working stack we built (see `report/report.md` §4 and `setup.sh`):

| component | version | note |
|---|---|---|
| python | 3.11 | venv |
| torch | **2.7.1+cu128** | from the cu128 wheel index; default PyPI aarch64 wheel is CPU-only |
| torchvision | 0.22.1 | |
| detectron2 | 0.6 (built from source) | CPU-only C++ ops (`CUDA_VISIBLE_DEVICES="" FORCE_CUDA=0`) |
| numpy | 1.26.4 | `<2` for the pycocotools / d2 ABI |
| transformers | 4.40.2 | BERT text encoder |

Two one-line source patches (`patches/`), applied by `setup.sh`:
1. **MSDeformAttn fallback** — the hand-written CUDA op can't be compiled (system nvcc 13.0
   vs torch cu128); fall back to the pure-PyTorch `grid_sample` path, which runs on GPU.
2. **sm_121 nvrtc fix** — int64 `spatial_shapes.prod(1)` falls to the nvrtc jiterator, whose
   cu128 nvrtc rejects `-arch=compute_121`; replace with the identical elementwise product.

## Setup
```bash
bash setup.sh                       # venv + torch(cu128) + deps + detectron2 + patches
source .venv/bin/activate
```

## Data & weights
```
datasets/
  grefcoco/grefs(unc).json   instances.json     # from HuggingFace FudanCVL/gRefCOCO
  images/train2014/COCO_train2014_*.jpg         # from http://images.cocodataset.org/zips/train2014.zip
models/
  gres_swin_tiny.pth   model_gres_swin_base.pth # ReLA authors' Google Drive
```
Helper:
```bash
python - <<'PY'
from huggingface_hub import hf_hub_download; import shutil,os
os.makedirs('datasets/grefcoco',exist_ok=True)
for fn in ['grefs(unc).json','instances.json']:
    shutil.copy(hf_hub_download('FudanCVL/gRefCOCO',fn,repo_type='dataset'),f'datasets/grefcoco/{fn}')
PY
wget http://images.cocodataset.org/zips/train2014.zip -P datasets/images && unzip -q datasets/images/train2014.zip -d datasets/images
gdown --folder https://drive.google.com/drive/folders/1Jw7GKiN-Y2tgLL6ueOKOKfikiWVOl2-n -O models
```

## Reproduce (everything regenerates from scripts)
```bash
# 1) one inference pass dumps per-sample signals + reproduces the baseline metrics
python scripts/run_inference.py  --backbone swin_tiny --split val
python scripts/run_inference.py  --backbone swin_tiny --split testA   # held-out for the detector

# 2) failure analysis (tables + figures)
python scripts/failure_no_target.py    --backbone swin_tiny --split val
python scripts/failure_multi_target.py --backbone swin_tiny --split val

# 3) improvement: calibrate the abstain/clarify detector on val, report on testA
python scripts/abstain_detector.py --backbone swin_tiny --cal_split val --test_split testA

# 4) qualitative figure panels
python scripts/make_figures.py --backbone swin_tiny --split val --category nt_hallucination
python scripts/make_figures.py --backbone swin_tiny --split val --category multi_target
```

Metrics (ported 1:1 from the official evaluator, `scripts/metrics.py`):
**gIoU**, **cIoU**, **N-acc** = TP/(TP+FN) (no-target recall), **T-acc** = TN/(TN+FP)
(target retention), **Pr@{0.7,0.8,0.9}**.

## Layout
```
scripts/    gres_lib, metrics, run_inference, failure_no_target, failure_multi_target,
            abstain_detector, make_figures
results/    *_records.pkl (signals), tables/*.json, figures/*.png
report/     report.md (+ figures) -> export report.pdf
patches/    source patches applied to third_party/ReLA
setup.sh    one-shot environment build
```

## Results summary (Swin-T, gRefCOCO)
Regenerate this table any time with `python scripts/report_numbers.py --backbone swin_tiny`.

| | gIoU | cIoU | N-acc | T-acc |
|---|---|---|---|---|
| **baseline, val** (paper 56.9/57.7) | 55.76 | 55.64 | 46.3 | 99.9 |
| + abstain detector, safe τ (val) | 66.92 | 58.22 | 65.0 | 95.1 |
| **baseline, testA** | 65.03 | 65.42 | 50.6 | 99.0 |
| + abstain detector, safe τ (testA, held out) | 65.94 | 64.46 | 63.3 | 90.6 |

- **F1 (no-target hallucination):** the model paints a confident mask on **53.7 %** of
  absent-referent expressions (N-acc 46.3 %); the NT score is bimodal and collapses into the
  present-target distribution.
- **F2 (multi-target under-segmentation):** within a multi-target expression it covers the best
  instance at 0.93 but the worst at only 0.68, and drops an instance entirely 26 % of the time.
- **Improvement:** a frozen-model abstain/clarify detector. The safety-constrained operating point
  (T-acc ≥ 0.95) raises gIoU/N-acc on val and **transfers** to held-out testA (N-acc +12.7); the
  gIoU-optimal threshold overfits val's no-target prior — reported honestly in §7.

Nothing in the report is hand-typed; every table/figure regenerates from `scripts/`.
