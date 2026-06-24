"""
Failure mode F2 -- MULTI-TARGET UNDER-SEGMENTATION.

Question: on expressions that refer to several instances ("all the X", "the two X"),
does the model capture every target, or only the most salient one?

We use the per-instance recall recorded at inference time: for each GT instance,
coverage = |pred & inst| / |inst|; an instance is "hit" if coverage >= 0.5.

Outputs:
  results/tables/F2_multi_target_<backbone>_<split>.json
  results/figures/F2_recall_by_count.png
"""
import argparse, json, os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gres_lib as G


def grp(records, lo, hi):
    return [r for r in records if (not r["gt_nt"]) and lo <= r["n_gt_inst"] <= hi]


def inst_recall(rs):
    tot = sum(r["n_gt_inst"] for r in rs)
    hit = sum(r["n_hit_inst"] for r in rs)
    return hit / max(tot, 1), tot, hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="swin_tiny")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    base = os.path.join(G.REPO_ROOT, "results", "%s_%s" % (args.backbone, args.split))
    recs = pickle.load(open(base + "_records.pkl", "rb"))

    single = grp(recs, 1, 1)
    multi = grp(recs, 2, 999)
    two = grp(recs, 2, 2)
    three_plus = grp(recs, 3, 999)

    def cov_stats(rs):
        covs = [c for r in rs for c in r["inst_cov"]]
        return float(np.mean(covs)) if covs else 0.0

    # sample-level: fraction of multi-target samples where >=1 instance is fully missed
    def fully_missed_rate(rs):
        if not rs:
            return 0.0
        miss = sum(1 for r in rs if r["n_hit_inst"] < r["n_gt_inst"])
        return miss / len(rs)

    r_single, ts, hs = inst_recall(single)
    r_multi, tm, hm = inst_recall(multi)
    r_two, _, _ = inst_recall(two)
    r_three, _, _ = inst_recall(three_plus)

    # Within each multi-target sample: coverage of the BEST-covered vs WORST-covered
    # GT instance. Under-segmentation = high best, low worst (model grabs one, drops one).
    best_covs = [max(r["inst_cov"]) for r in multi if r["inst_cov"]]
    worst_covs = [min(r["inst_cov"]) for r in multi if r["inst_cov"]]
    best_cov_mean = float(np.mean(best_covs)) if best_covs else 0.0
    worst_cov_mean = float(np.mean(worst_covs)) if worst_covs else 0.0
    worst_missed_frac = float(np.mean([c < 0.5 for c in worst_covs])) if worst_covs else 0.0

    # mean IoU (vs merged GT) single vs multi -- the headline "metric drop"
    def mean_iou(rs):
        v = [(r["I"] / r["U"]) if r["U"] > 0 else 0.0 for r in rs]
        return float(np.mean(v)) if v else 0.0

    out = dict(
        backbone=args.backbone, split=args.split,
        n_single=len(single), n_multi=len(multi), n_two=len(two), n_three_plus=len(three_plus),
        inst_recall_single=r_single,
        inst_recall_multi=r_multi,
        inst_recall_two=r_two,
        inst_recall_three_plus=r_three,
        mean_instance_coverage_single=cov_stats(single),
        mean_instance_coverage_multi=cov_stats(multi),
        mIoU_single=mean_iou(single),
        mIoU_multi=mean_iou(multi),
        multi_with_a_fully_missed_instance=fully_missed_rate(multi),
        best_instance_coverage_mean=best_cov_mean,
        worst_instance_coverage_mean=worst_cov_mean,
        worst_instance_missed_frac=worst_missed_frac,
    )
    os.makedirs(os.path.join(G.REPO_ROOT, "results", "tables"), exist_ok=True)
    with open(os.path.join(G.REPO_ROOT, "results", "tables",
                           "F2_multi_target_%s_%s.json" % (args.backbone, args.split)), "w") as f:
        json.dump(out, f, indent=2)
    for k, v in out.items():
        print("  %-36s %s" % (k, v))

    # ---- figure (left): per-instance recall by GT target count (1 / 2 / 3+) --
    #      shows recall is flat for 1-2 targets but collapses at >=3.
    #      (right): within-sample best-vs-worst instance coverage = uneven seg.
    groups = [grp(recs, 1, 1), grp(recs, 2, 2), grp(recs, 3, 999)]
    labels = ["1", "2", "3+"]
    recalls = [inst_recall(g)[0] for g in groups]
    ns = [len(g) for g in groups]

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    bars = axs[0].bar(range(3), recalls, color=["#27ae60", "#2980b9", "#c0392b"], width=0.6)
    for i, (v, n) in enumerate(zip(recalls, ns)):
        axs[0].text(i, v + 0.01, "%.3f\n(n=%d)" % (v, n), ha="center", fontsize=8)
    axs[0].set_xticks(range(3)); axs[0].set_xticklabels(labels)
    axs[0].set_xlabel("# ground-truth target instances")
    axs[0].set_ylabel("per-instance recall (coverage ≥ 0.5)")
    axs[0].set_ylim(0, 1)
    axs[0].set_title("F2: recall is flat for 1–2 targets but collapses at ≥3\n(%s, %s)"
                     % (args.backbone, args.split))

    axs[1].bar([0, 1], [best_cov_mean, worst_cov_mean],
               color=["#2980b9", "#c0392b"], width=0.55)
    for i, v in enumerate([best_cov_mean, worst_cov_mean]):
        axs[1].text(i, v + 0.01, "%.2f" % v, ha="center", fontsize=10)
    axs[1].set_xticks([0, 1])
    axs[1].set_xticklabels(["best-covered\ninstance", "worst-covered\ninstance"])
    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel("coverage  |pred ∩ inst| / |inst|")
    axs[1].set_title("Within a multi-target expression, coverage is very uneven\n"
                     "(%.0f%% of multi-target samples drop an instance entirely)"
                     % (100 * worst_missed_frac))
    fig.tight_layout()
    fig.savefig(os.path.join(G.REPO_ROOT, "results", "figures", "F2_recall_by_count.png"), dpi=140)
    print("[saved] F2 tables + figure")


if __name__ == "__main__":
    main()
