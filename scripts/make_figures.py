"""
Qualitative figure panels from the figure cache produced by run_inference.py.

Produces, for a given category, an N-row panel: [image] [image+GT] [image+pred],
annotated with the expression, the NT score and the IoU. Categories:
  nt_hallucination -- absent referent, model painted a mask (F1)
  multi_target     -- >=2 GT instances (F2)
  abstain_flip     -- cases the abstain detector flips wrong-mask -> correct-abstain

Usage:
  python scripts/make_figures.py --backbone swin_tiny --split val \
      --category nt_hallucination --n 6
"""
import argparse, os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import gres_lib as G
from detectron2.data import transforms as T


def unpack(rec, key):
    h, w = rec["shape"]
    bits = np.unpackbits(rec[key])[: h * w].reshape(h, w)
    return bits.astype(np.uint8)


def load_resized_image(file_name, size=480):
    img = np.array(Image.open(file_name).convert("RGB"))
    aug = T.Resize((size, size))
    img2, _ = T.apply_transform_gens([aug], img)
    return img2.astype(np.uint8)


def overlay(ax, img, mask, color=(255, 0, 0), alpha=0.5, title=""):
    ax.imshow(img)
    if mask is not None and mask.sum() > 0:
        ov = np.zeros((*mask.shape, 4), dtype=np.float32)
        ov[mask == 1] = (color[0] / 255, color[1] / 255, color[2] / 255, alpha)
        ax.imshow(ov)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="swin_tiny")
    ap.add_argument("--split", default="val")
    ap.add_argument("--category", default="nt_hallucination")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fc = pickle.load(open(os.path.join(
        G.REPO_ROOT, "results", "%s_%s_figcache.pkl" % (args.backbone, args.split)), "rb"))
    items = [r for r in fc if r["category"] == args.category][: args.n]
    if not items:
        cats = sorted({r["category"] for r in fc})
        print("[warn] no items for category=%s. available: %s" % (args.category, cats)); return

    rows = len(items)
    fig, axs = plt.subplots(rows, 3, figsize=(9, 3 * rows))
    if rows == 1:
        axs = axs[None, :]
    for i, r in enumerate(items):
        img = load_resized_image(r["file_name"])
        pred = unpack(r, "pred_mask")
        gt = unpack(r, "gt_mask")
        iou = (r["I"] / r["U"]) if r["U"] > 0 else float("nan")
        sent = r["sent"][:42]
        overlay(axs[i, 0], img, None, title='"%s"' % sent)
        overlay(axs[i, 1], img, gt, color=(0, 180, 0),
                title="GT  (nt=%s)" % r["gt_nt"])
        overlay(axs[i, 2], img, pred, color=(220, 0, 0),
                title="pred  nt_score=%.2f  IoU=%.2f" % (r["nt_score"], iou))
    fig.suptitle("%s  (%s, %s)" % (args.category, args.backbone, args.split), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = args.out or os.path.join(G.REPO_ROOT, "report", "figures",
                                   "qual_%s_%s.png" % (args.category, args.backbone))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("[saved] %s  (%d rows)" % (out, rows))


if __name__ == "__main__":
    main()
