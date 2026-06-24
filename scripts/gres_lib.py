"""
Shared setup for GRES/ReLA inference on gRefCOCO.

This module is the single place that knows how to:
  * put the vendored ReLA code (third_party/ReLA) on the path,
  * build a frozen detectron2 cfg for a given backbone,
  * build the model and load a pretrained checkpoint,
  * build a single-GPU test data loader,
  * extract the rich per-sample signals we need for the failure analysis and
    the abstain/clarify detector (NT logits, mask confidence, area, #components).

Everything downstream (failure_*, abstain_detector, make_figures) consumes the
pickle produced by run_inference.py, so the GPU is only needed once.
"""
import os
import sys
import numpy as np
import torch
from scipy import ndimage

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELA_ROOT = os.path.join(REPO_ROOT, "third_party", "ReLA")
if RELA_ROOT not in sys.path:
    sys.path.insert(0, RELA_ROOT)

# Make sure the dataset registry points at our datasets/ folder.
os.environ.setdefault("DETECTRON2_DATASETS", os.path.join(REPO_ROOT, "datasets"))

from detectron2.config import get_cfg                       # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from detectron2.checkpoint import DetectionCheckpointer     # noqa: E402
from detectron2.data import build_detection_test_loader     # noqa: E402
from detectron2.modeling import build_model                 # noqa: E402

import gres_model  # noqa: E402,F401  (registers datasets + config + mapper + model)
from gres_model import add_maskformer2_config, add_refcoco_config, RefCOCOMapper  # noqa: E402

CONFIGS = {
    "swin_tiny": "configs/referring_swin_tiny.yaml",
    "swin_base": "configs/referring_swin_base.yaml",
    "R50": "configs/referring_R50.yaml",
}
WEIGHTS = {
    "swin_tiny": "models/gres_swin_tiny.pth",
    "swin_base": "models/model_gres_swin_base.pth",
    "R50": "models/gres_r50.pth",
}


def setup_cfg(backbone="swin_tiny", weights=None, opts=None):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_refcoco_config(cfg)
    cfg.merge_from_file(os.path.join(RELA_ROOT, CONFIGS[backbone]))
    cfg.MODEL.WEIGHTS = weights or os.path.join(REPO_ROOT, WEIGHTS[backbone])
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if opts:
        cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


def build_gres(cfg):
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    return model


def build_loader(cfg, dataset_name):
    mapper = RefCOCOMapper(cfg, is_train=False)
    return build_detection_test_loader(cfg, dataset_name, mapper=mapper)


def _as_hw(mask):
    """Squeeze a (1,H,W) or (H,W) tensor/array to a numpy (H,W) uint8."""
    m = mask
    if hasattr(m, "detach"):
        m = m.detach().cpu().numpy()
    m = np.asarray(m)
    if m.ndim == 3:
        m = m[0]
    return m.astype(np.uint8)


def extract_signals(inp, out):
    """Turn one (input, output) pair into a flat dict of scalars + small arrays.

    output['ref_seg']  : (2,H,W) sigmoid foreground/background-ish channels
    output['nt_label'] : (2,)    sigmoid no-target logits (idx 1 == "no target")
    """
    ref_seg = out["ref_seg"].detach().cpu()      # (2,H,W)
    fg_prob = ref_seg[1]                          # foreground channel
    pred_mask = ref_seg.argmax(0).numpy().astype(np.uint8)   # (H,W) binary

    nt = out["nt_label"].detach().cpu().numpy()  # (2,)
    nt_score = float(nt[1])                       # "no-target" probability-like score
    nt_margin = float(nt[1] - nt[0])
    pred_nt = bool(nt[1] > nt[0])

    area_px = int(pred_mask.sum())
    H, W = pred_mask.shape
    area_frac = area_px / float(H * W)

    if area_px > 0:
        fg = fg_prob.numpy()
        mask_conf_mean = float(fg[pred_mask == 1].mean())
        mask_conf_max = float(fg.max())
        # connected components on the predicted mask
        lab, ncomp = ndimage.label(pred_mask)
        comp_sizes = np.bincount(lab.ravel())[1:] if ncomp > 0 else np.array([])
        largest_frac = float(comp_sizes.max() / area_px) if ncomp > 0 else 0.0
    else:
        mask_conf_mean = 0.0
        mask_conf_max = float(fg_prob.max())
        ncomp = 0
        largest_frac = 0.0

    gt_merged = _as_hw(inp["gt_mask_merged"])
    gt_nt = bool(inp.get("empty", False))

    # IoU against merged GT (same convention as the official ReferEvaluator)
    I = int(np.logical_and(pred_mask, gt_merged).sum())
    U = int(np.logical_or(pred_mask, gt_merged).sum())

    # Per-instance recall for multi-target analysis (non-empty samples only).
    n_gt_inst = 0
    n_hit_inst = 0
    inst_cov = []
    if not gt_nt and "gt_mask" in inp:
        gms = inp["gt_mask"]
        if hasattr(gms, "numpy"):
            gms = gms.numpy()
        gms = np.asarray(gms)
        if gms.ndim == 3:
            n_gt_inst = int(gms.shape[0])
            for k in range(n_gt_inst):
                g = gms[k].astype(np.uint8)
                ga = int(g.sum())
                if ga == 0:
                    continue
                inter = int(np.logical_and(pred_mask, g).sum())
                cov = inter / float(ga)        # recall of this instance
                inst_cov.append(round(cov, 4))
                if cov >= 0.5:
                    n_hit_inst += 1

    rec = dict(
        img_id=int(inp["image_id"]),
        ref_id=int(inp["sentence"].get("ref_id", -1)) if "ref_id" in inp["sentence"] else -1,
        sent_id=int(inp["sentence"].get("sent_id", -1)) if "sent_id" in inp["sentence"] else -1,
        sent=inp["sentence"]["raw"],
        source=inp["source"],
        gt_nt=gt_nt,
        pred_nt=pred_nt,
        nt_score=nt_score,
        nt_margin=nt_margin,
        nt0=float(nt[0]),
        nt1=float(nt[1]),
        I=I, U=U,
        area_px=area_px, area_frac=round(area_frac, 6),
        mask_conf_mean=round(mask_conf_mean, 6),
        mask_conf_max=round(mask_conf_max, 6),
        ncomp=int(ncomp),
        largest_frac=round(largest_frac, 6),
        gt_area_px=int(gt_merged.sum()),
        n_gt_inst=n_gt_inst,
        n_hit_inst=n_hit_inst,
        inst_cov=inst_cov,
    )
    return rec, pred_mask, gt_merged
