# Visual Media (映像メディア学) — Final Report

### Reproducing ReLA/GRES on gRefCOCO, analysing its failure modes, and adding an abstain-or-clarify detector

Every number and figure in this report is produced by the scripts in the accompanying
repository ([shiinayane/visual-media-gres](https://github.com/shiinayane/visual-media-gres),
`scripts/` and `results/`) and can be regenerated from a single inference pass per split.

---

## 1. Identity block

- **Name:** WANG Yankai
- **Student ID:** 48256454
- **Department:** 情報理工学系研究科電子情報学専攻
- **Laboratory:** 鈴村研究室
- **Own research topic:** Vision-Language-Action / human-robot collaboration

---

## 2. Summary of the target paper

**GRES: Generalized Referring Expression Segmentation** (Liu, Ding, Jiang, *CVPR 2023
Highlight*; network name **ReLA**) generalises classic Referring Expression Segmentation
(RES). Classic RES assumes **exactly one** target object per expression. GRES relaxes this
to allow:

- **multi-target** expressions ("the two bottles on the left", "everyone except the kid"), and
- **no-target** expressions, whose referent is **absent** from the image.

To support and measure this, the authors release **gRefCOCO** (built on COCO train2014 images,
RefCOCO-style expressions) with train/val/testA/testB splits, and an evaluation protocol with
**gIoU** (mean per-sample IoU; a correctly-predicted no-target sample scores 1), **cIoU**
(cumulative intersection / cumulative union), **N-acc** (no-target recall, = TP/(TP+FN)),
**T-acc** (target retention, = TN/(TN+FP)) and **Pr@{0.7,0.8,0.9}**.

The network, **ReLA**, is built on Mask2Former / Detectron2. It explicitly models
**Region-Image (RIA)** and **Region-Language (RLA)** interactions: the image is divided into
soft regions; each region attends to the language and to image features, and the regions then
attend to one another so the model can reason about *relationships between regions* rather than
classifying one box. A dedicated **no-target (NT) head** predicts whether the referent exists
at all. With these, ReLA reaches state-of-the-art on gRefCOCO and remains competitive on
classic RefCOCO/+/g.

## 3. Understanding of the method

Reading the released code (`gres_model/`), the inference path is:

1. **Backbone** (Swin-T/-B or R50) extracts multi-scale visual features; a **BERT-base** text
   encoder embeds the expression. The backbone is *language-aware* — language features are fed
   in so vision and language are fused early.
2. **MSDeformAttn pixel decoder** (the Mask2Former deformable-attention encoder) builds a
   high-resolution mask feature map and multi-scale features.
3. **ReLA transformer decoder** (`MultiScaleMaskedReferringDecoder`): 100 learnable
   *region queries* iterate through (i) **RIA** cross-attention to image features, (ii) on the
   first layer a full **RLA** language cross-attention gated by a learned `lang_weight`, and
   (iii) region-region self-attention. Two prediction heads matter at test time:
   - a **minimap / mask** branch — `minimap_embed` produces per-region 2-way logits that are
     combined (`einsum`) with region embeddings and the mask feature map to yield a **2-channel
     target mask** `pred_masks`∈ℝ^{2×H×W}; `argmax` over the 2 channels is the binary output.
   - a **no-target head** — `nt_embed` (a small MLP) is applied per region and **averaged over
     all 100 regions** to give a single 2-way **NT logit**; `argmax` decides "no target".
4. **Inference** (`GRES.refer_inference`) simply sigmoids both outputs. The official evaluator
   compares `argmax(ref_seg)` to the merged GT mask and `argmax(nt_label)` to the empty flag.

One observation motivates the rest of the report. The model already exposes two scalars at
inference, the no-target score `σ(nt_logit)` and the foreground mask confidence. These are the
quantities a system would need in order to decide whether to act, abstain, or ask a clarifying
question. ReLA does not use them this way: it collapses each with a hard `argmax` at 0.5 and always
commits to either a mask or "empty". Section 6 keeps the trained model unchanged and replaces this
hard decision with a calibrated abstain-or-clarify policy.

## 4. Execution environment (hardware, CUDA, libraries, reproduction)

**Hardware.** NVIDIA **GB10** (Grace-Blackwell, **aarch64**), CUDA driver 580 / **CUDA 13.0**,
device compute capability **sm_121**, 20 CPU cores, 119 GB RAM.

Most of the setup effort went here. The stack the paper targets
(torch 1.11 / CUDA 11.8 / detectron2 0.6, Python 3.7–3.9) does not run on a Blackwell/aarch64
machine, so a modern equivalent had to be assembled. The working configuration (single GPU,
inference only) is:

| component    | working version             | why / workaround                                                                                                                                                                                              |
| ------------ | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python       | 3.11 (venv)                 |                                                                                                                                                                                                               |
| PyTorch      | **2.7.1+cu128**             | the default PyPI aarch64 wheel is **CPU-only**; installed from the `cu128` index. CUDA verified on GB10 (a 2048² matmul runs; arch list `sm_90/100/120`).                                                     |
| torchvision  | 0.22.1                      | matched to torch 2.7.1                                                                                                                                                                                        |
| detectron2   | 0.6 (**built from source**) | not on PyPI for this stack; built **CPU-only** C++ ops via `CUDA_VISIBLE_DEVICES="" FORCE_CUDA=0 pip install -e . --no-build-isolation` (avoids compiling CUDA kernels with the mismatched system nvcc 13.0). |
| numpy        | 1.26.4                      | pinned `<2` for the pycocotools / detectron2 ABI                                                                                                                                                              |
| transformers | 4.40.2                      | BERT-base text encoder                                                                                                                                                                                        |

**Two source patches** (kept in `patches/`, applied by `setup.sh`), both forced by the
Blackwell + CUDA-13 mismatch:

1. **MSDeformAttn fallback.** The hand-written `MultiScaleDeformableAttention` CUDA op cannot be
   compiled (system nvcc **13.0** vs torch **cu128/12.8**). I let the import fail gracefully and
   fall back to ReLA's own pure-PyTorch `ms_deform_attn_core_pytorch`, which runs on GPU via
   `F.grid_sample`. This is the *reference* implementation of the op, so results are numerically
   **close** (not bit-identical — `grid_sample` boundary handling and fp accumulation differ from
   the hand-written CUDA kernel); I attribute the roughly 1-point baseline gap below partly to this.
2. **sm_121 nvrtc fix.** `msdeformattn.py` computes `spatial_shapes.prod(1)` on an int64 tensor.
   Integer `prod` has no precompiled kernel, so PyTorch JIT-compiles it with **nvrtc**, which in
   the cu128 build rejects `-arch=compute_121` (*"invalid value for --gpu-architecture"*). I
   replace it with the identical elementwise product `shapes[:,0]*shapes[:,1]`, which uses a
   precompiled kernel. (Any int-reduction that falls to the jiterator hits the same wall on
   sm_121 + cu128 — worth noting for anyone reproducing on Blackwell.)

**Reproduction.** `bash setup.sh`, fetch data/weights (README), then run the four script steps
in the README. One inference pass (`run_inference.py`) dumps compact per-sample signals to
`results/*_records.pkl`; **all** downstream analysis is offline and deterministic. Inference
runs at ≈ 0.24 s/sample (grid_sample fallback) on one GB10.

**Baseline reproduction (gRefCOCO val, Swin-T, 14 229 expressions).** I reproduce
**gIoU = 55.76, cIoU = 55.64**, N-acc = 46.3 %, T-acc = 99.9 %, Pr@0.7 = 66.5. The paper
reports Swin-T cIoU 57.73 / gIoU 56.86, so the result lands within about 1–2 points of the paper, a faithful
reproduction. The small gap is consistent with the `grid_sample` MSDeformAttn fallback and
single-GPU evaluation; it does not affect any of the qualitative conclusions below.

## 5. Failure Case Analysis

I study two failure conditions on gRefCOCO's official splits, so each claim is a measured rate
not an anecdote. F1 (no-target) is analysed on **val** (63 % of its samples are no-target — the
ideal stress test); F2 (multi-target) is analysed on **testA**, the split that actually contains a
single-target baseline to compare against. Inference is run once per split; the analysis scripts
then quantify each mode and dump qualitative panels.

### F1 — No-target hallucination *(primary)*

**Setup.** Restrict to the **no-target** subset of val (referent absent, `gt_nt=True`,
n = 8 905, i.e. 63 % of val). A *hallucination* is a no-target sample on which the model still
emits a non-empty mask (`pred_nt=False`).

**Result.** The model hallucinates on 53.7 % of no-target samples (4 784 / 8 905); its no-target
recall is N-acc = 46.3 %. The opposite error is almost absent: T-acc = 99.9 % (only 7 of 5 324
present-target samples are wrongly called empty). ReLA is therefore strongly biased toward
declaring a target present and rarely abstains, even when it should. The hallucinated masks are
not small artefacts either: they cover a median of 7.2 % of the image (mean 9.1 %) and carry a
mean foreground confidence of 0.68, so the model tends to be wrong with high confidence.

**Mechanism.** Figure `F1_nt_score_hist.png` plots the model's own no-target score
`s_nt=σ(nt_logit)_1` for present vs absent referents. Present-target samples pile up tightly at
`s_nt≈0` (mean 0.012). Absent-referent samples are **bimodal**: a high mode (~37 % land above 0.9,
correctly "no target") and a **low mode (~38 % collapse to `s_nt≈0`, landing inside the
present-target distribution)**, with the rest spread between — so overall **53 % of no-target
samples fall below the 0.5 threshold** and hallucinate (the empty-subset mean of 0.48 is just the
average of the two modes, not a typical value). Architecturally the NT head is a *single* scalar
**averaged over all 100 region queries** and thresholded at 0.5 (`nt_label = nt_embed(·).mean(dim=1)`);
a plausible explanation is that when a few regions confidently match a salient distractor that
loosely fits the words, they pull the mean below 0.5 and a mask is committed. I did not log the
per-region NT logits, so this account is consistent with the architecture but not directly
verified (see Section 8). As noted above, these hallucinated masks are sizeable (median 7.2 % of
the image) rather than thin slivers. For a system that acts on language instructions, this kind of
high-confidence false positive is the most problematic error to make. Qualitative examples
(empty ground truth, non-empty prediction) are in `report/figures/qual_nt_hallucination_swin_tiny.png`.

<figure>
<img src="../results/figures/F1_nt_score_hist.png" style="width:58%">
<figcaption><b>Figure F1a.</b> No-target score for present vs absent referents. On absent referents
the score is bimodal: one mode goes correctly to ≈1, but a large mode collapses to ≈0, inside the
present-target distribution, so 53 % fall below the 0.5 threshold and a mask is committed.</figcaption>
</figure>

<figure>
<img src="figures/qual_nt_hallucination_swin_tiny.png" style="width:44%">
<figcaption><b>Figure F1b.</b> Each row is one <i>no-target</i> expression (left), the empty
ground truth (middle), and the model's confident hallucinated mask in red (right).</figcaption>
</figure>

### F2 — Multi-target under-segmentation

**Setup.** I analyse **testA** here because — unlike val, whose targeted expressions are *all*
multi-instance — testA contains a **single-instance baseline** (5 917 one-target, 5 940 two-target,
2 895 three-or-more expressions), so I can ask whether multi-target is genuinely harder than
single-target. For each GT instance I record coverage = |pred ∩ inst|/|inst|, call it a *hit* if
coverage ≥ 0.5, and report per-instance **recall** = hits / GT instances.

**Result (two distinct effects).** (i) Per-instance recall is **essentially flat from one to two
targets** — 0.868 (single) vs 0.881 (two) — and then **collapses to 0.594 at ≥ 3 targets**: one
2-channel mask can represent one or two regions but saturates beyond. (ii) Independently of the
count, within a multi-target expression the coverage is **very uneven** — the best-covered instance
averages **0.92** but the worst only **0.58**, and the model **drops at least one instance entirely
in 37.8 %** of multi-target testA samples (26.1 % on val). So the under-segmentation is real, but it
is *not* "two targets are worse than one on average"; it is (a) a hard **capacity ceiling at ≥3
targets** and (b) **lopsided coverage** that sacrifices the less salient instance even when the
sample-average looks acceptable.

**Mechanism.** ReLA predicts a single 2-channel "target" mask, formed by an `einsum` over the
per-region minimap weights. One mask can represent a single region, or two when the soft region
weighting is shared between them, but it cannot spread probability mass across many disjoint
instances. This matches both observations: recall drops sharply once there are three or more
targets, and within a pair the weaker (less salient) instance is the one that gets dropped.
Qualitative examples (ground-truth instances in green, prediction in red, with the prediction
often covering only the more salient instance) are in `report/figures/qual_multi_target_swin_tiny.png`.

<figure>
<img src="../results/figures/F2_recall_by_count.png" style="width:74%">
<figcaption><b>Figure F2a.</b> testA. Left: per-instance recall is flat for 1–2 targets (0.87, 0.88)
but collapses at ≥3 (0.59). Right: within a multi-target expression the best instance is covered at
0.92 but the worst at only 0.58 — the model sacrifices the less salient instance (38% of samples
drop one entirely).</figcaption>
</figure>

<figure>
<img src="figures/qual_multi_target_swin_tiny.png" style="width:44%">
<figcaption><b>Figure F2b.</b> Qualitative multi-target cases (val, which exhibits the same
behaviour). Each row is one expression; ground-truth instances in green (middle), prediction in red
(right). The model frequently captures one instance and drops the other.</figcaption>
</figure>

## 6. The improvement — a post-hoc abstain-or-clarify detector

The improvement is motivated by my own research interest in human-robot collaboration, where a
system facing an unsatisfiable or ambiguous instruction should ask or abstain rather than commit
to a confident output. As noted in Section 3, GRES already exposes the signals needed for this.
I keep the trained model unchanged (no retraining) and replace its hard `argmax` with a calibrated
post-hoc policy. For each sample I define

  `s_abstain = max(s_nt, 1 − mask_conf_mean)`

— high when the model itself leans "no target", **or** when its committed mask is low-confidence.
The policy is:

- `s_abstain ≥ τ` → **ABSTAIN** (emit empty; scored as a no-target prediction),
- else if a mask is committed but has **≥2 connected components** with the largest covering
  `< ρ` of the area → **CLARIFY** ("did you mean A or B?"),
- else → **COMMIT** the mask.

The threshold `τ` is calibrated on val (chosen to maximise gIoU, which rewards both correct
abstention on no-target samples and good masks on targeted ones) and then applied unchanged to the
held-out testA split. I also report a safety operating point `τ_safe`, the most aggressive
abstention that still keeps T-acc ≥ 0.95 (sacrificing at most 5 % of present targets). The gRefCOCO
metric in Section 7 reflects only the ABSTAIN decision (abstain produces an empty mask, scored as a
no-target prediction). CLARIFY has no counterpart in the benchmark, so for scoring I keep the
model's best-guess mask and report the CLARIFY trigger separately, as the number of committed
predictions the policy would turn into a "which one?" question. Calibration gives τ = 0.23
(gIoU-optimal) and τ_safe = 0.38, with ρ = 0.7.

## 7. Before/after comparison

All numbers from `results/tables/abstain_swin_tiny.json`; trade-off in
`results/figures/abstain_tradeoff.png`. τ is calibrated on **val** and applied **unchanged** to
the held-out **testA** split.

**gRefCOCO val (Swin-T, n = 14 229):**

| operating point             | gIoU      | cIoU      | N-acc    | T-acc |
| --------------------------- | --------- | --------- | -------- | ----- |
| baseline (model argmax)     | 55.76     | 55.64     | 46.3     | 99.9  |
| **+ detector, safe τ=0.38** | **66.92** | **58.22** | **65.0** | 95.1  |
| + detector, gIoU-opt τ=0.23 | 74.07     | 52.39     | 87.7     | 62.5  |

**gRefCOCO testA (held out, τ transferred unchanged from val; n = 19 200):**

| operating point             | gIoU      | cIoU  | N-acc    | T-acc |
| --------------------------- | --------- | ----- | -------- | ----- |
| baseline (model argmax)     | 65.03     | 65.42 | 50.6     | 99.0  |
| **+ detector, safe τ=0.38** | **65.94** | 64.46 | **63.3** | 90.6  |
| + detector, gIoU-opt τ=0.23 | 54.09     | 49.31 | 83.6     | 54.6  |

The val row should be read as a calibration curve rather than a result: τ is both chosen and scored
on val, so those gains (gIoU +11.2, cIoU +2.6, N-acc +19 at the safe point; gIoU +18 at the
gIoU-optimal point) are in-sample and optimistic. I include them only to show the shape of the
trade-off. The honest evaluation is the held-out testA split.

On testA, with τ transferred unchanged, the detector is best described as a safety re-weighting
rather than an improvement in segmentation quality. The safe point (τ = 0.38) raises N-acc by 12.7
points (50.6 to 63.3) at a cost of 8.4 points of T-acc (99.0 to 90.6), while the overlap metrics
are essentially unchanged (gIoU +0.9, within the roughly 1-point reproduction gap, and cIoU −1.0).
What the detector buys is therefore a tunable, calibrated willingness to abstain on unsatisfiable
instructions, at a bounded cost in target retention. The aggressive gIoU-optimal threshold
(τ = 0.23) does not transfer: it was tuned to val's 63 %-no-target prior, and on the 23 %-no-target
testA it over-abstains and lowers gIoU by 11 points (65.0 to 54.1). This itself is a useful
observation: a single global abstention threshold depends on the no-target base-rate of the
deployment distribution, so only the T-acc-constrained operating point transfers safely, and a
practical detector would calibrate τ to the expected base-rate. Overall the experiment supports
evaluating and operating grounding models with an explicit abstain option, while showing that the
benefit on standard overlap metrics is modest and prior-dependent.

<figure>
<img src="../results/figures/abstain_tradeoff.png" style="width:99%">
<figcaption><b>Figure 7.</b> Abstain/clarify trade-off, calibrated on val. Left: N-acc vs T-acc as
τ sweeps; the model's default argmax point (black) sits at the extreme no-abstention corner, while
the safe (orange diamond) and gIoU-optimal (green) points move up the frontier. Right: gIoU and
cIoU vs τ both peak well above the default (dotted).</figcaption>
</figure>

The trade-off figure (left: N-acc vs T-acc as τ sweeps, with the model's default `argmax` point
marked; right: gIoU/cIoU vs τ) shows the default operating point sits at the extreme T-acc corner
of the frontier — it almost never abstains — so moving along the curve trades target retention for
no-target recall (and, on the no-target-heavy val split, gIoU). The **clarify** branch (separate
from abstention) fired on **1 131** val / **1 712** testA committed predictions at τ=0.23; these are
multi-component masks for which asking "which one?" is a reasonable alternative to committing to a
single region. This is an illustrative remedy for F2; I report the trigger count only, since
gRefCOCO has no clarification label against which to measure its precision (Section 8). The same
no-target examples in `report/figures/qual_nt_hallucination_swin_tiny.png` also show cases that the
detector flips from a confident wrong mask to a correct abstention.

## 8. Limitations

- Inference-only on the frozen Swin-T checkpoint (Swin-B was downloaded but, per the
  resource-friendly guideline, not evaluated). With no retraining the detector can only reuse
  existing signals, not fix the region-weighting bias behind F2.
- The grid_sample fallback for MSDeformAttn is the reference op (numerically close, not identical)
  but slower; I did not compile the custom CUDA op, so the ~1-point baseline gap is plausibly but
  not provably benign.
- The F1 mechanism is a hypothesis: I logged only the pooled `s_nt`, not per-region NT logits, so
  the averaging-dilution account is consistent with the architecture but not directly measured.
- The abstain score `max(s_nt, 1−mask_conf)` is a simple interpretable rule whose two terms are
  correlated; a learned calibrator over {s_nt, mask_conf, area, #components} would likely do better.
- On held-out data the detector is a safety re-weighting, not a segmentation-quality gain, and a
  single global τ depends on the no-target base-rate (the gIoU-optimal τ does not transfer).
- The CLARIFY trigger is a geometric heuristic reported only as a count; gRefCOCO has no
  clarification label, so its precision is unmeasured, and it does not yet read the expression.

## 9. Generative-AI usage

I used a generative-AI coding assistant (Anthropic's Claude, via the Claude Code CLI) throughout
this project, and disclose its use as follows.

**What the AI assisted with.** Debugging the Blackwell / CUDA-13 environment and proposing the
working software stack and the two source patches in Section 4; writing the bulk of the analysis
code in `scripts/` (the inference loop, the metric port, the failure-mode and detector scripts, and
the plotting); running the experiments on the GPU server; and producing a first draft of the report
text from the resulting numbers.

**What I am responsible for.** I selected the target paper and defined the direction of the work,
including the choice to frame the improvement as an abstain-or-clarify mechanism connected to my own
research on human-robot collaboration. I reviewed the code and the experimental setup, checked every
reported number and figure against the generated result tables, requested the corrections that
followed an internal review (for example re-centring the F2 analysis on the testA split and
reporting the held-out rather than in-sample detector results), and edited the report into its final
form. I take responsibility for the content and conclusions. A more detailed running log is kept in
`GENAI_USAGE_LOG.md`.

## 10. Discussion

The two failure modes have a common origin. ReLA makes a single hard decision at inference (one
mask, one 0.5 no-target threshold), whereas the generalized task is partly a question of
uncertainty: whether a referent is present at all, and how many there are. The no-target and
multi-target subsets are exactly the cases where committing to a single argmax is the wrong default.
The detector in Section 6 does not add capacity; it reuses the uncertainty the model already encodes
and turns it into three possible actions (act, abstain, ask), giving a calibrated and tunable level
of abstention at a bounded retention cost that transfers to a held-out split.

For a benchmark aimed at human-robot use, this suggests that an explicit abstain/clarify option is
worth including in how grounding models are both scored and operated, since the no-target and
multi-target cases are where a confident but wrong mask is most costly. The gains here are modest and
prior-dependent; natural next steps would be a small learned calibrator over the four signals, an
expression-aware CLARIFY trigger, and an abstention-aware metric (e.g. risk-coverage) reported
alongside gIoU/cIoU.

---
*Citation:* Chang Liu, Henghui Ding, Xudong Jiang. "GRES: Generalized Referring Expression
Segmentation." **CVPR 2023** (Highlight). arXiv:2306.00968.
