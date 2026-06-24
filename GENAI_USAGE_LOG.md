# Generative-AI Usage Log

> Honest running log of how generative AI was used to produce this assignment.
> **The human (report author) must review and correct this section before submission.**
> Placeholder identity fields and the GitHub URL are intentionally left as TODO for the human.

## Tool
- **Claude Code (Anthropic), model "Opus 4.8 (1M context)"**, run as an autonomous coding
  agent directly on the GPU server. It had shell access and executed every command,
  wrote every script, ran the experiments, and drafted the report text.

## What the AI did (with human oversight of the final result)
1. **Environment bring-up.** Diagnosed that the server is a Blackwell/aarch64 + CUDA 13
   machine on which the paper's original torch-1.11/cu118/detectron2-0.6 stack cannot run.
   Selected and installed a working stack (torch 2.7.1+cu128, detectron2 built from source
   CPU-only, MSDeformAttn pure-PyTorch fallback). All version choices and the two build
   workarounds were the AI's; they are documented in report §4 and `setup.sh`.
2. **Code reading.** Read the ReLA / gRefCOCO source to confirm the *actual* metric
   definitions (gIoU, cIoU, N-acc=TP/(TP+FN), T-acc=TN/(TN+FP), Pr@X) and the model's
   no-target / mask outputs, rather than trusting the assignment brief.
3. **Scripts.** Wrote all scripts in `scripts/` (inference dump, the ported metric, the two
   failure analyses, the abstain/clarify detector, the figure maker).
4. **Experiments.** Ran baseline inference, the failure-mode quantifications, the detector
   calibration/evaluation, and generated all tables and figures. No numbers in the report
   were typed by hand; every figure/table is regenerated from `scripts/`.
5. **Report draft.** Drafted the English prose of all 10 sections from the real results.

## What a human still must verify / revise
- The identity block (name, ID, department, lab, own research topic) — TODO placeholders.
- The honesty of this very log.
- That the conceptual framing (abstain/clarify ↔ the author's VLA / ambiguity-clarification
  research) genuinely matches the author's intent.
- A final read of the report against the 10 required sections and the prohibitions.

## Running command log (high level)
(Appended chronologically as work proceeded — see git history for exact commands.)
- setup: venv (py3.11), torch 2.7.1+cu128 (CUDA verified on GB10, cap 12.1), deps pinned.
- detectron2 0.6 built from source (CPU C++ ops) against torch 2.7.1.
- data: gRefCOCO annotations from HuggingFace (FudanCVL/gRefCOCO); COCO train2014 images;
  ReLA Swin-T & Swin-B checkpoints from the authors' Google Drive.
- baseline reproduced (val, Swin-T): gIoU 55.76 / cIoU 55.64 (paper 56.86 / 57.73 -> within ~1-2 pts).
- F1 no-target hallucination: N-acc 46.3%, 53.7% of absent-referent samples get a confident mask.
- F2 multi-target under-seg: best-instance coverage 0.93 vs worst 0.68; 26% drop one instance.
- abstain/clarify detector: val safe-tau gIoU +11.2 / N-acc +19; held-out testA safe-tau N-acc +12,
  gIoU +0.9 (gIoU-optimal tau overfits val's no-target prior -> reported honestly as a finding).
- AI judgement the human should sanity-check: the choice to FROZEN-model post-hoc detector, the
  abstain-score definition max(s_nt, 1-mask_conf), the gIoU-vs-safe operating-point framing, and
  the cross-split prior-shift interpretation are all the agent's analysis, not the paper's.
