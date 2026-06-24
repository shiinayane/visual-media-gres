# Generative-AI Usage Log

This file records, in more detail than report Section 9, how a generative-AI assistant was used in
this assignment. The intent is full transparency for grading.

## Tool

- **Anthropic Claude**, used through the **Claude Code** command-line assistant on the GPU server.
  It was used interactively: I set the goals and reviewed the output at each stage, and it carried
  out the coding and experiment-running steps under that direction.

## How the assistant was used

- **Environment setup.** Helped diagnose why the paper's original stack
  (torch 1.11 / CUDA 11.8 / detectron2 0.6) does not run on this Blackwell / aarch64 / CUDA-13
  machine, and proposed the working stack and the two source patches documented in report Section 4
  (`setup.sh`, `patches/`).
- **Code.** Wrote most of the analysis code in `scripts/` — the inference loop that dumps per-sample
  signals, the gRefCOCO metric port, the two failure-mode scripts, the abstain/clarify detector, and
  the plotting — which I then reviewed and ran.
- **Experiments.** Ran the baseline reproduction, the failure-mode measurements, and the detector
  calibration/evaluation on the GPU, and produced the tables and figures in `results/`.
- **Writing.** Produced a first draft of the report from the generated numbers, which I then edited.

## What I am responsible for

- Choosing the target paper and deciding the direction of the improvement (the abstain-or-clarify
  framing, which connects to my own research on human-robot collaboration and instruction ambiguity).
- Reviewing the code and experimental design, and checking every number and figure in the report
  against the generated result tables (`scripts/report_numbers.py` prints them in one place).
- Running an internal review pass and requesting the resulting corrections — most importantly
  moving the F2 multi-target analysis to the testA split (which, unlike val, has a single-target
  baseline) and reporting the detector on the held-out split rather than the in-sample val numbers.
- Editing the report and standing behind its content and conclusions.

## Technical log of the work (factual)

- Environment: Python 3.11 venv; torch 2.7.1+cu128 (CUDA confirmed working on the GB10, sm_121);
  detectron2 0.6 built from source with CPU-only C++ ops; numpy pinned `<2`.
- Two patches to the vendored ReLA source (in `patches/`): a pure-PyTorch MSDeformAttn fallback,
  and an `sm_121` fix replacing an int64 `prod` that the cu128 nvrtc could not JIT-compile.
- Data: gRefCOCO annotations from HuggingFace (FudanCVL/gRefCOCO); COCO train2014 images; ReLA
  Swin-T and Swin-B checkpoints from the authors' Google Drive (Swin-T used; Swin-B not evaluated).
- Baseline (val, Swin-T): gIoU 55.76 / cIoU 55.64, vs the paper's 56.86 / 57.73 (within ~1–2 points).
- F1, no-target hallucination (val): N-acc 46.3 %; 53.7 % of no-target expressions still receive a
  mask, at a mean confidence of 0.68.
- F2, multi-target under-segmentation (testA): per-instance recall 0.87 / 0.88 for one/two targets
  and 0.59 for three or more; within multi-target samples, best-instance coverage 0.92 vs worst 0.58,
  with one instance dropped entirely in 38 % of cases.
- Abstain/clarify detector: on held-out testA the safe operating point gives N-acc +12.7 at an
  8.4-point T-acc cost with roughly unchanged overlap metrics; the gIoU-optimal threshold tuned on
  val does not transfer (reported as a finding, not hidden).
