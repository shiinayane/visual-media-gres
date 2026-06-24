"""
gRefCOCO metrics, ported 1:1 from gres_model/evaluation/refer_evaluation.py.

A "record" is a dict with at least: I, U, gt_nt (bool), pred_nt (bool).
`pred_nt_override` lets the abstain/clarify detector replace the model's own
no-target decision without re-running inference: when a sample is abstained on,
we set pred_nt=True (i.e. the system emits an empty mask), which is exactly how
the official metric scores a no-target prediction.
"""
import numpy as np

PR_THRES = [0.7, 0.8, 0.9]


def compute_metrics(records, pred_nt_key="pred_nt"):
    accum_I = accum_U = 0
    accum_IoU = 0.0
    total = 0
    not_empty = 0
    pr_count = {t: 0 for t in PR_THRES}
    nt = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    for r in records:
        I, U = int(r["I"]), int(r["U"])
        gt_nt = bool(r["gt_nt"])
        pred_nt = bool(r[pred_nt_key])

        if gt_nt:
            if pred_nt:                       # correctly said "no target"
                nt["TP"] += 1
                accum_IoU += 1.0
            else:                             # missed the no-target -> hallucinated mask
                nt["FN"] += 1
                accum_U += U
        else:
            if pred_nt:                       # wrongly said "no target"
                nt["FP"] += 1
                I = 0
            else:
                nt["TN"] += 1
            this_iou = 0.0 if U == 0 else I / U
            accum_IoU += this_iou
            accum_I += I
            accum_U += U
            not_empty += 1
            for t in PR_THRES:
                if this_iou >= t:
                    pr_count[t] += 1
        total += 1

    res = {}
    res["gIoU"] = 100.0 * accum_IoU / max(total, 1)
    res["cIoU"] = 100.0 * accum_I / max(accum_U, 1)
    res["N_acc"] = nt["TP"] / max(nt["TP"] + nt["FN"], 1)
    res["T_acc"] = nt["TN"] / max(nt["TN"] + nt["FP"], 1)
    for t in PR_THRES:
        res["Pr@%.1f" % t] = 100.0 * pr_count[t] / max(not_empty, 1)
    res.update({"_" + k: v for k, v in nt.items()})
    res["_total"] = total
    res["_not_empty"] = not_empty
    res["_empty"] = total - not_empty
    return res


def fmt(res):
    keys = ["gIoU", "cIoU", "N_acc", "T_acc", "Pr@0.7", "Pr@0.8", "Pr@0.9"]
    return "  ".join("%s=%.2f" % (k, res[k]) for k in keys)
