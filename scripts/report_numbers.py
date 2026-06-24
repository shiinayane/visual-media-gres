"""Print every headline number used in the report, straight from results/tables/*.json.
This is the single source of truth -- the report prose cites these, nothing is hand-typed.

  python scripts/report_numbers.py --backbone swin_tiny
"""
import argparse, glob, json, os
import gres_lib as G


def load(name):
    p = os.path.join(G.REPO_ROOT, "results", "tables", name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="swin_tiny")
    a = ap.parse_args()
    bb = a.backbone

    print("=" * 70)
    print("BASELINE (paper Swin-T: cIoU 57.73, gIoU 56.86)")
    for split in ["val", "testA", "testB"]:
        r = load(f"{bb}_{split}_baseline.json")
        if r:
            print(f"  {split:6s} gIoU={r['gIoU']:.2f} cIoU={r['cIoU']:.2f} "
                  f"N_acc={r['N_acc']*100:.1f} T_acc={r['T_acc']*100:.1f} "
                  f"Pr@0.7={r['Pr@0.7']:.1f}  (n={r['_total']}, empty={r['_empty']})")

    print("=" * 70)
    print("F1  NO-TARGET HALLUCINATION (val)")
    r = load(f"F1_no_target_{bb}_val.json")
    if r:
        for k in ["n_empty", "N_acc", "hallucination_rate", "n_hallucinations",
                  "halluc_area_frac_median", "halluc_mask_conf_mean",
                  "nt_score_empty_mean", "nt_score_targeted_mean",
                  "nt_score_empty_below_0.5_frac"]:
            print(f"  {k:32s} {r[k]}")

    print("=" * 70)
    print("F2  MULTI-TARGET UNDER-SEGMENTATION (val)")
    r = load(f"F2_multi_target_{bb}_val.json")
    if r:
        for k in ["n_single", "n_multi", "inst_recall_single", "inst_recall_two",
                  "inst_recall_three_plus", "mIoU_single", "mIoU_multi",
                  "multi_with_a_fully_missed_instance"]:
            print(f"  {k:34s} {r[k]}")

    print("=" * 70)
    print("IMPROVEMENT  abstain/clarify")
    r = load(f"abstain_{bb}.json")
    if r:
        print(f"  tau_giou={r['tau_giou']:.2f}  tau_safe={r['tau_safe']:.2f}  rho={r['rho']}")
        for split, d in r["splits"].items():
            b, ag, asf = d["before"], d["after_giou_tau"], d["after_safe_tau"]
            print(f"  [{split}] n={d['n_total']} clarify={d['n_clarify_giou_tau']}")
            print(f"    before     gIoU={b['gIoU']:.2f} cIoU={b['cIoU']:.2f} "
                  f"N_acc={b['N_acc']*100:.1f} T_acc={b['T_acc']*100:.1f}")
            print(f"    +abstain(τ) gIoU={ag['gIoU']:.2f} cIoU={ag['cIoU']:.2f} "
                  f"N_acc={ag['N_acc']*100:.1f} T_acc={ag['T_acc']*100:.1f}")
            print(f"    +safe(τ)    gIoU={asf['gIoU']:.2f} cIoU={asf['cIoU']:.2f} "
                  f"N_acc={asf['N_acc']*100:.1f} T_acc={asf['T_acc']*100:.1f}")


if __name__ == "__main__":
    main()
