# Server-AI task prompt (paste the block below to the GPU-server agent)

> Note: this meta-prompt can be removed from the repo before final submission if you
> prefer not to include it. Keeping it is also fine (transparent about AI usage).

```text
ROLE
You are a research/coding agent on a GPU server. Your job is to COMPLETE a graduate
course final assignment end-to-end: run a paper, do experimental failure analysis,
implement one improvement, and write a 4–8 page report plus a reproducible code repo.
Work autonomously. When something cannot run, report the exact error and your fix —
never fabricate results or numbers.

================================================================================
1. ASSIGNMENT RULES (course: "Visual Media" / 映像メディア学, UTokyo)
================================================================================
The required deliverables are:
  (A) A report (PDF, 4–8 pages) with EXACTLY these 10 sections:
       1. Identity block: name, student ID, department, laboratory, own research topic
          — LEAVE THESE AS CLEARLY MARKED TODO PLACEHOLDERS for the human to fill.
       2. Summary of the target paper
       3. Understanding of the method
       4. Execution environment (hardware, CUDA, library versions, how to reproduce)
       5. Failure Case Analysis  (THE MOST IMPORTANT SECTION — see §4)
       6. The improvement you made
       7. Before/after comparison (quantitative tables + qualitative figures)
       8. Limitations
       9. Generative-AI usage  (tools, purposes, what was human-revised — see §7)
      10. Discussion
  (B) Source code: a clean GitHub-ready repo (README, requirements/lockfile, scripts,
      configs, figures). Put the repo URL placeholder in the report.
  (C) A generative-AI usage log (simple is fine): tools used, main purposes, the parts
      a human revised/judged. Keep an honest running log as YOU work.

HARD PROHIBITIONS (auto-fail if violated):
  - "Only ran the code" with no analysis.
  - No failure analysis.
  - Verbatim AI-generated prose dumped as the report.
  - Opinion essay with no experiments.
  - Using copyright-violating data.
  - Calling an API/model without understanding its internals.
Therefore: every claim in the report must be backed by an experiment YOU ran, with
numbers and figures, and you must explain the MECHANISM (why it fails), not just that
it fails.

Report language: write the report in ENGLISH by default. Add a one-line note at the top
flagging that the human may want it in Japanese instead.

================================================================================
2. TARGET PAPER
================================================================================
GRES: Generalized Referring Expression Segmentation  (network name: ReLA)
  Venue: CVPR 2023 (Highlight).  arXiv: 2306.00968
  Code:    https://github.com/henghuiding/ReLA
  Dataset/eval (gRefCOCO + GREC metric): https://github.com/henghuiding/gRefCOCO
  Backbones with pretrained weights: ResNet-50, Swin-Tiny, Swin-Base.
  Built on Mask2Former / Detectron2.

Why this paper: classic referring-expression segmentation assumes exactly one target.
GRES generalizes it to allow MULTI-TARGET ("the two bottles on the left") and crucially
NO-TARGET expressions (the referent is absent from the image), via region-region and
region-language modeling plus a no-target prediction. It ships the gRefCOCO benchmark
and metrics (read the paper/repo for exact metric definitions — typically gIoU, cIoU,
and no-target accuracy N-acc / T-acc; CONFIRM the exact names and eval commands from the
repo, do not trust this list blindly).

================================================================================
3. WHO THIS IS FOR (so the improvement is aligned)
================================================================================
The human's own research is on Vision-Language-Action / human-robot collaboration,
object grounding, and especially AMBIGUITY & CLARIFICATION and SAFETY in benchmarks
(a system that, when an instruction is ambiguous or unsatisfiable, ASKS a clarifying
question or ABSTAINS instead of acting blindly). The improvement in §5 must reuse that
idea: turn GRES's no-target / low-confidence behavior into an explicit
abstain-or-clarify mechanism. This is the conceptual through-line of the report.

================================================================================
4. STEP-BY-STEP PLAN
================================================================================
STEP 0 — Setup & sanity
  - Clone ReLA + gRefCOCO, set up the Detectron2/Mask2Former env (expect CUDA 11.8 /
    PyTorch 1.11-era stack; resolve version conflicts and DOCUMENT the final working
    versions for report §4). Download gRefCOCO and a pretrained checkpoint (start with
    Swin-Tiny or ResNet-50 to stay light — inference only, no full retraining).
  - Reproduce the paper's reported evaluation on gRefCOCO val. Confirm your numbers are
    in the right ballpark of the paper; if not, debug and note the gap. This is your
    BASELINE — record it.

STEP 1 — Failure Case Analysis (≥2 conditions, the most important section)
  Use gRefCOCO's own splits so failures are measurable, not anecdotal. Implement AT
  LEAST TWO of the following as quantified experiments, each with: (a) the metric drop,
  (b) 4–8 qualitative example figures (input image + expression + predicted mask vs GT),
  (c) a mechanism explanation of WHY it breaks.
    F1. NO-TARGET HALLUCINATION (primary): feed expressions whose referent is ABSENT.
        Measure how often the model still emits a non-empty mask (false-positive rate /
        N-acc). Analyze the no-target signal distribution to explain why.
    F2. MULTI-TARGET UNDER-SEGMENTATION: on "all the X" / "the two X" queries, measure
        per-target recall — show it tends to capture only the most salient instance.
    F3. (optional) SPATIAL/RELATIONAL or COUNTING expressions ("third from the left"):
        show accuracy collapses vs attribute-only queries.
  Build a small, reproducible eval script per failure mode that prints the metric and
  dumps the example figures.

STEP 2 — Improvement (≥1, headline = abstain/clarify detector)
  PRIMARY IMPROVEMENT: a post-hoc ABSTAIN-OR-CLARIFY detector on top of the frozen model
  (no expensive retraining). Using the model's existing signals (no-target logit/prob,
  mask confidence, predicted mask area, #connected components), define a decision policy:
      - confident single region  -> output mask
      - no-target / low-confidence -> ABSTAIN (output empty) or emit a clarification
        request instead of forcing a mask
      - multiple plausible regions -> emit a clarification ("did you mean A or B?")
  Calibrate the thresholds on val. This directly mirrors a human-robot "ask before
  acting" policy. Evaluate the precision/recall trade-off: does it raise N-acc / cut
  false-positive masks, and at what cost to true-positive recall? Plot the trade-off
  curve.
  (Optional secondary improvement if time allows: confidence re-ranking or threshold
  tuning to improve multi-target recall.)

STEP 3 — Before/after comparison (report §7)
  Same eval, baseline vs improved. Produce: (a) a metric table (baseline vs +detector),
  (b) the precision/recall trade-off figure, (c) qualitative cases that flip from
  wrong-mask to correct-abstain/clarify.

STEP 4 — Write report + finalize repo
  Fill all 10 sections from §1 with YOUR real results. Generate every figure/table from
  scripts in the repo (no hand-faked numbers). Produce a clean README with exact repro
  commands, a requirements/lockfile, and organized scripts (e.g. scripts/eval_baseline,
  scripts/failure_no_target, scripts/failure_multi_target, scripts/abstain_detector,
  scripts/make_figures). Keep the GenAI usage log updated.

================================================================================
5. DELIVERABLE LAYOUT (produce all of this)
================================================================================
  report/        report.md (or .tex) + report.pdf, all figures
  src/ or scripts/   reproducible eval + failure + improvement scripts
  README.md      setup + exact reproduce commands + result summary
  requirements / environment file (pinned, working versions)
  GENAI_USAGE_LOG.md   running log of AI assistance
  results/       metric tables (csv/json) + figures, regenerable from scripts

================================================================================
6. GUARDRAILS
================================================================================
  - Inference-only baseline; keep any fine-tuning minimal. Prefer Swin-T/R50.
  - Use only the official gRefCOCO/RefCOCO data (license-clean). Do not scrape images.
  - If a result is negative or messy, REPORT IT HONESTLY — negative findings with
    analysis are valid and expected.
  - Confirm exact metric names, splits, and eval commands from the repo/paper before
    quoting them; correct anything in this prompt that the repo contradicts.
  - Cite the paper correctly as CVPR 2023.

================================================================================
7. WHAT TO LEAVE FOR THE HUMAN (do NOT invent these)
================================================================================
  - Identity block (name, student ID, department, laboratory, own research topic):
    leave clearly marked TODO placeholders.
  - GitHub repo URL: placeholder.
  - The Generative-AI usage section must list what YOU (the agent) did; the human will
    review and adjust it for honesty before submission. Do not overstate human authorship.

================================================================================
8. FINAL REPORT-BACK
================================================================================
End by giving the human: (1) the headline baseline-vs-improved numbers, (2) which two
failure conditions you demonstrated and the mechanism for each, (3) any steps that did
not run and how you worked around them, (4) a checklist mapping each of the 10 required
report sections to where it is satisfied, (5) what the human still must do before
submitting (fill identity block, review GenAI log, push to GitHub, export PDF, submit to
UTOL by the deadline).
```
