"""
Failure mode F1 -- NO-TARGET HALLUCINATION.

Question: when the referent is ABSENT (gt_nt=True), how often does the model still
emit a non-empty mask, and why?

Outputs:
  results/tables/F1_no_target_<backbone>_<split>.json
  results/figures/F1_nt_score_hist.png
  results/figures/F1_halluc_area_hist.png
"""
import argparse, json, os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gres_lib as G
from metrics import compute_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="swin_tiny")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    base = os.path.join(G.REPO_ROOT, "results", "%s_%s" % (args.backbone, args.split))
    recs = pickle.load(open(base + "_records.pkl", "rb"))

    empty = [r for r in recs if r["gt_nt"]]
    targ = [r for r in recs if not r["gt_nt"]]

    # A "hallucination" = no-target sample where the model emitted a non-empty mask
    # (pred_nt == False). Equivalent to a False Negative in the NT confusion matrix.
    halluc = [r for r in empty if not r["pred_nt"]]
    n_emp = len(empty)
    halluc_rate = len(halluc) / max(n_emp, 1)

    res_all = compute_metrics(recs)

    # area statistics of hallucinated masks (fraction of image covered)
    halluc_area = np.array([r["area_frac"] for r in halluc]) if halluc else np.array([0.0])
    halluc_conf = np.array([r["mask_conf_mean"] for r in halluc]) if halluc else np.array([0.0])

    # nt_score distributions (model's internal no-target score, idx-1 sigmoid)
    nt_emp = np.array([r["nt_score"] for r in empty])
    nt_targ = np.array([r["nt_score"] for r in targ])

    out = dict(
        backbone=args.backbone, split=args.split,
        n_total=len(recs), n_empty=n_emp, n_targeted=len(targ),
        N_acc=res_all["N_acc"], T_acc=res_all["T_acc"],
        hallucination_rate=halluc_rate,
        n_hallucinations=len(halluc),
        halluc_area_frac_mean=float(halluc_area.mean()),
        halluc_area_frac_median=float(np.median(halluc_area)),
        halluc_mask_conf_mean=float(halluc_conf.mean()),
        nt_score_empty_mean=float(nt_emp.mean()),
        nt_score_targeted_mean=float(nt_targ.mean()),
        gIoU_all=res_all["gIoU"], cIoU_all=res_all["cIoU"],
    )
    out["nt_score_empty_below_0.5_frac"] = float((nt_emp < 0.5).mean())
    os.makedirs(os.path.join(G.REPO_ROOT, "results", "tables"), exist_ok=True)
    with open(os.path.join(G.REPO_ROOT, "results", "tables",
                           "F1_no_target_%s_%s.json" % (args.backbone, args.split)), "w") as f:
        json.dump(out, f, indent=2)
    for k, v in out.items():
        print("  %-32s %s" % (k, v))

    # ---- figure 1: nt_score distribution, empty vs targeted ----
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(nt_targ, bins=bins, alpha=0.55, density=True, label="targeted (target present)")
    ax.hist(nt_emp, bins=bins, alpha=0.55, density=True, label="no-target (target absent)")
    ax.axvline(0.5, color="k", ls="--", lw=1, label="default decision (0.5)")
    ax.set_xlabel("no-target score  $s_{nt}=\\sigma(\\mathrm{nt\\_logit})_{[1]}$")
    ax.set_ylabel("density")
    ax.set_title("F1: on absent referents the NT score is bimodal; a large low mode\n"
                 "collapses to ~0 (inside the present-target mass) so 53%% fall below 0.5\n(%s, %s)"
                 % (args.backbone, args.split))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(G.REPO_ROOT, "results", "figures", "F1_nt_score_hist.png"), dpi=140)

    # ---- figure 2: hallucinated mask area ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(halluc_area, bins=np.linspace(0, max(0.05, halluc_area.max()), 40), color="#c0392b", alpha=0.8)
    ax.set_xlabel("predicted mask area (fraction of image)")
    ax.set_ylabel("# hallucinated masks")
    ax.set_title("F1: on absent referents the model paints sizeable masks\n"
                 "median=%.3f of image, n=%d" % (np.median(halluc_area), len(halluc)))
    fig.tight_layout()
    fig.savefig(os.path.join(G.REPO_ROOT, "results", "figures", "F1_halluc_area_hist.png"), dpi=140)
    print("[saved] F1 tables + figures")


if __name__ == "__main__":
    main()
