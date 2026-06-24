"""
Run GRES/ReLA inference once over a gRefCOCO split and dump:
  * results/<backbone>_<split>_records.pkl   -- compact per-sample signals (all samples)
  * results/<backbone>_<split>_figcache.pkl  -- image+masks for a curated subset (for figures)
  * results/tables/<backbone>_<split>_baseline.json -- reproduced baseline metrics

Usage:
  python scripts/run_inference.py --backbone swin_tiny --split val
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
from PIL import Image

import gres_lib as G
from metrics import compute_metrics, fmt


def curate(rec, counters, cap=12):
    """Decide whether to cache full masks for this sample (for qualitative figs)."""
    if rec["gt_nt"] and not rec["pred_nt"]:
        key = "nt_hallucination"          # no-target but model drew a mask
    elif rec["gt_nt"] and rec["pred_nt"]:
        key = "nt_correct"
    elif (not rec["gt_nt"]) and rec["n_gt_inst"] >= 2:
        key = "multi_target"
    elif (not rec["gt_nt"]) and rec["U"] > 0 and rec["I"] / rec["U"] >= 0.7:
        key = "single_good"
    else:
        key = "single_other"
    if counters.get(key, 0) < cap:
        counters[key] = counters.get(key, 0) + 1
        return key
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="swin_tiny", choices=list(G.CONFIGS))
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap #samples")
    ap.add_argument("--figcap", type=int, default=12, help="per-category figure cache cap")
    args = ap.parse_args()

    dataset_name = "grefcoco_unc_%s" % args.split
    cfg = G.setup_cfg(args.backbone)
    print("[cfg] backbone=%s weights=%s device=%s" % (args.backbone, cfg.MODEL.WEIGHTS, cfg.MODEL.DEVICE))
    model = G.build_gres(cfg)
    loader = G.build_loader(cfg, dataset_name)

    records = []
    figcache = []
    counters = {}
    t0 = time.time()
    n = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch)
            for inp, out in zip(batch, outputs):
                rec, pred_mask, gt_merged = G.extract_signals(inp, out)
                records.append(rec)
                key = curate(rec, counters, cap=args.figcap)
                if key is not None:
                    figcache.append(dict(
                        category=key,
                        img_id=rec["img_id"],
                        file_name=inp["file_name"],
                        sent=rec["sent"],
                        gt_nt=rec["gt_nt"],
                        pred_nt=rec["pred_nt"],
                        nt_score=rec["nt_score"],
                        I=rec["I"], U=rec["U"],
                        n_gt_inst=rec["n_gt_inst"], n_hit_inst=rec["n_hit_inst"],
                        pred_mask=np.packbits(pred_mask),     # H*W bits
                        gt_mask=np.packbits(gt_merged.astype(np.uint8)),
                        shape=pred_mask.shape,
                    ))
                n += 1
            if n % 500 == 0:
                dt = time.time() - t0
                print("  %5d samples  %.1fs  (%.3fs/sample)" % (n, dt, dt / max(n, 1)), flush=True)
            if args.limit and n >= args.limit:
                break

    dt = time.time() - t0
    print("[done] %d samples in %.1fs (%.3fs/sample)" % (n, dt, dt / max(n, 1)))

    os.makedirs(os.path.join(G.REPO_ROOT, "results", "tables"), exist_ok=True)
    base = os.path.join(G.REPO_ROOT, "results", "%s_%s" % (args.backbone, args.split))
    with open(base + "_records.pkl", "wb") as f:
        pickle.dump(records, f)
    with open(base + "_figcache.pkl", "wb") as f:
        pickle.dump(figcache, f)

    res = compute_metrics(records)
    print("[baseline] " + fmt(res))
    print("           empty=%d not_empty=%d total=%d  NT(TP/TN/FP/FN)=%d/%d/%d/%d"
          % (res["_empty"], res["_not_empty"], res["_total"],
             res["_TP"], res["_TN"], res["_FP"], res["_FN"]))
    with open(os.path.join(G.REPO_ROOT, "results", "tables",
                           "%s_%s_baseline.json" % (args.backbone, args.split)), "w") as f:
        json.dump(res, f, indent=2)
    print("[saved] " + base + "_records.pkl / _figcache.pkl")


if __name__ == "__main__":
    main()
