# Visual Media (映像メディア学) — Final Report Assignment

Target paper: **GRES: Generalized Referring Expression Segmentation (ReLA)**, CVPR 2023 (Highlight).
- Paper: https://arxiv.org/abs/2306.00968
- Code: https://github.com/henghuiding/ReLA
- Dataset/eval: https://github.com/henghuiding/gRefCOCO

Plan: run ReLA on gRefCOCO → analyze ≥2 failure modes (no-target hallucination,
multi-target under-segmentation) → add an **abstain/clarify detector** (reuses my
LabMate ambiguity→clarification work) → 4–8 page report + this repo.

## Deadline
- EN assignment PDF says **2026-07-31 23:59 JST**; JP PDF says 2027 — **CONFIRM the year
  with the instructor.** Submit via **UTOL** (email submission not accepted).

## Workflow
1. (done) local repo with `server_prompt.md` + `.gitignore`.
2. Create GitHub repo and push this skeleton (see below).
3. On the GPU server: `git clone` this repo, paste `server_prompt.md` to the agent.
4. Agent fills `report/`, `src/`/`scripts/`, `results/`, `GENAI_USAGE_LOG.md`, pushes back.
5. `git pull` here → review against the checklist → export `report.pdf` → submit to UTOL.

## I must do personally (do NOT let the AI fake these)
- [ ] Fill the identity block (name, student ID, department, lab, my research topic).
- [ ] Review & correct `GENAI_USAGE_LOG.md` for honesty.
- [ ] Confirm the deadline year with the instructor.
- [ ] Pick report language (English default; switch to Japanese if required).
- [ ] Final read of the report against the 10 required sections + prohibitions.

## Repo layout (filled by the server agent)
```
report/   report.(md|tex) + report.pdf + figures
src/ | scripts/   eval_baseline, failure_*, abstain_detector, make_figures
results/  metric tables + figures
README.md / requirements / GENAI_USAGE_LOG.md
```
