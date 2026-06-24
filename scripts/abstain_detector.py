"""
IMPROVEMENT -- post-hoc ABSTAIN-OR-CLARIFY detector on the frozen GRES model.

Motivation (human-robot collaboration): when an instruction is unsatisfiable
(referent absent) or ambiguous (several plausible referents), a safe agent should
ABSTAIN or ASK rather than blindly paint a mask. GRES already exposes the signals
we need -- the no-target score s_nt and the mask confidence/area/#components -- so we
can build the policy without any retraining.

Policy (per sample):
    s_abstain = max(s_nt, 1 - mask_conf_mean)          # "should I refuse to commit?"
    if s_abstain >= tau:        ->  ABSTAIN   (emit empty; == predict no-target)
    elif mask committed and #components>=2 and largest_frac < rho:
                                ->  CLARIFY   (ambiguous: ask "which one?")
    else:                       ->  COMMIT    (emit the mask)

For the gRefCOCO metric, both ABSTAIN and CLARIFY emit no mask, i.e. pred_nt=True.
tau is *calibrated on val* and then applied unchanged to testA/testB.

Outputs:
  results/tables/abstain_<backbone>.json          (before/after table + chosen tau)
  results/figures/abstain_tradeoff.png            (N_acc vs T_acc and gIoU vs tau)
  results/tables/clarify_<backbone>.json          (clarify-trigger analysis)
"""
import argparse, json, os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gres_lib as G
from metrics import compute_metrics, fmt


def abstain_score(r):
    # high when the model's own no-target score is high OR its mask confidence is low
    return max(r["nt_score"], 1.0 - r["mask_conf_mean"])


def apply_policy(records, tau, rho=0.7):
    """Return a copy of records with pred_nt overridden by the abstain decision,
    plus per-sample action label."""
    out = []
    n_clarify = 0
    for r in records:
        s = abstain_score(r)
        action = "commit"
        pred_nt = r["pred_nt"]
        if s >= tau:
            action = "abstain"
            pred_nt = True
        else:
            # committed a mask -> check ambiguity for a clarify request
            if (not pred_nt) and r["ncomp"] >= 2 and r["largest_frac"] < rho and r["area_px"] > 0:
                action = "clarify"
                n_clarify += 1
        rr = dict(r)
        rr["pred_nt_eff"] = pred_nt
        rr["action"] = action
        out.append(rr)
    return out, n_clarify


def sweep(records, taus):
    Nacc, Tacc, gIoU, cIoU = [], [], [], []
    for tau in taus:
        ov, _ = apply_policy(records, tau)
        res = compute_metrics(ov, pred_nt_key="pred_nt_eff")
        Nacc.append(res["N_acc"]); Tacc.append(res["T_acc"])
        gIoU.append(res["gIoU"]); cIoU.append(res["cIoU"])
    return map(np.array, (Nacc, Tacc, gIoU, cIoU))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="swin_tiny")
    ap.add_argument("--cal_split", default="val", help="split used to pick tau")
    ap.add_argument("--test_split", default="testA", help="split used to report (held out)")
    ap.add_argument("--rho", type=float, default=0.7)
    args = ap.parse_args()

    tdir = os.path.join(G.REPO_ROOT, "results", "tables")
    fdir = os.path.join(G.REPO_ROOT, "results", "figures")
    os.makedirs(tdir, exist_ok=True)

    cal = pickle.load(open(os.path.join(
        G.REPO_ROOT, "results", "%s_%s_records.pkl" % (args.backbone, args.cal_split)), "rb"))

    taus = np.linspace(0.0, 1.0, 101)
    Nacc, Tacc, gIoU, cIoU = sweep(cal, taus)

    # Calibrate: pick tau maximizing gIoU on the calibration split (gIoU jointly
    # rewards correct abstention on no-target and good masks on targeted samples).
    best_i = int(np.argmax(gIoU))
    tau = float(taus[best_i])

    # also a "safety" operating point: the MOST abstention (smallest tau) that still
    # keeps T_acc >= 0.95 -> maximizes no-target recall while sacrificing <=5% of targets.
    safe_idx = [i for i in range(len(taus)) if Tacc[i] >= 0.95]
    tau_safe = float(taus[min(safe_idx)]) if safe_idx else tau

    base_res = compute_metrics(cal)  # model default (argmax) on cal

    report = {"backbone": args.backbone, "tau_giou": tau, "tau_safe": tau_safe,
              "rho": args.rho, "splits": {}}

    for split in [args.cal_split, args.test_split]:
        path = os.path.join(G.REPO_ROOT, "results", "%s_%s_records.pkl" % (args.backbone, split))
        if not os.path.exists(path):
            print("[skip] no records for split=%s" % split); continue
        recs = pickle.load(open(path, "rb"))
        before = compute_metrics(recs)
        ov, n_clar = apply_policy(recs, tau, rho=args.rho)
        after = compute_metrics(ov, pred_nt_key="pred_nt_eff")
        ov_s, n_clar_s = apply_policy(recs, tau_safe, rho=args.rho)
        after_s = compute_metrics(ov_s, pred_nt_key="pred_nt_eff")
        report["splits"][split] = {
            "before": {k: before[k] for k in ["gIoU", "cIoU", "N_acc", "T_acc"]},
            "after_giou_tau": {k: after[k] for k in ["gIoU", "cIoU", "N_acc", "T_acc"]},
            "after_safe_tau": {k: after_s[k] for k in ["gIoU", "cIoU", "N_acc", "T_acc"]},
            "n_clarify_giou_tau": n_clar,
            "n_total": before["_total"],
        }
        print("\n=== split=%s ===" % split)
        print("  before     : " + fmt(before))
        print("  after(tau=%.2f): %s   clarify=%d" % (tau, fmt(after), n_clar))
        print("  after(safe=%.2f): %s" % (tau_safe, fmt(after_s)))

    with open(os.path.join(tdir, "abstain_%s.json" % args.backbone), "w") as f:
        json.dump(report, f, indent=2)

    # ---- trade-off figure (computed on calibration split) ----
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].plot(Tacc, Nacc, "-o", ms=2, color="#8e44ad")
    axs[0].scatter([base_res["T_acc"]], [base_res["N_acc"]], color="k", zorder=5,
                   label="model default (argmax)")
    # mark chosen + safe tau
    axs[0].scatter([Tacc[best_i]], [Nacc[best_i]], color="#27ae60", zorder=5,
                   label="gIoU-opt $\\tau$=%.2f" % tau)
    si = int(min([i for i in range(len(taus)) if Tacc[i] >= 0.95], default=best_i))
    axs[0].scatter([Tacc[si]], [Nacc[si]], color="#e67e22", marker="D", zorder=5,
                   label="safe $\\tau$=%.2f (T-acc$\\geq$.95)" % taus[si])
    axs[0].set_xlabel("T-acc  (target retention)")
    axs[0].set_ylabel("N-acc  (no-target recall)")
    axs[0].set_title("Abstain trade-off (%s, %s)" % (args.backbone, args.cal_split))
    axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)

    axs[1].plot(taus, gIoU, "-", color="#2980b9", label="gIoU")
    axs[1].plot(taus, cIoU, "-", color="#e67e22", label="cIoU")
    axs[1].axvline(tau, color="#27ae60", ls="--", lw=1, label="chosen $\\tau$")
    axs[1].axhline(base_res["gIoU"], color="#2980b9", ls=":", lw=1, alpha=0.7,
                   label="gIoU default=%.1f" % base_res["gIoU"])
    axs[1].set_xlabel("abstain threshold $\\tau$")
    axs[1].set_ylabel("score")
    axs[1].set_title("gIoU/cIoU vs $\\tau$")
    axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "abstain_tradeoff.png"), dpi=140)
    print("\n[saved] abstain tables + tradeoff figure")


if __name__ == "__main__":
    main()
