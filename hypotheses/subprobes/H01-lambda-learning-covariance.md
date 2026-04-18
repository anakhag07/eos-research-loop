---
id: H01-lambda-learning-covariance
title: lambda_max co-varies with learning on specific prototype groups
parent: H_primary-eos-selective-tradeoff
status: open
linked_sweeps: []
evidence: []
created: 2026-04-18
updated: 2026-04-18
source: program.md:77
---

# lambda_max co-varies with learning on specific prototype groups

## Question
Does `lambda_max` co-vary with learning on specific prototype groups?

## Rationale
The research objective is interpretable signal at the curvature × learning
interface, treating prototype groups as first-class analysis units. If
`lambda_max` tracks learning (i.e. subset-group loss drops) for some groups
but not others, that is a decomposition of global training dynamics into
group-conditional regimes — the target phenomenon.

## Proposed test
Fixed hyperparameters, vary prototype composition only. For each run compute:
- `group.<g>.slope.full_loss.slope` (does the group learn?)
- `group.<g>.eos_crossing.lambda_max.step` (when does its curvature reach EoS?)
- `group.<g>.stability_ratio.mean`
Test: is there a systematic relationship (rank correlation) between when a
group's `lambda_max` crosses EoS and when its loss starts dropping? If so,
the co-variance is real; if random, refuted.

## Signals of interest
- `group.<group>.eos_crossing.lambda_max.step`
- `group.<group>.slope.full_loss.slope`
- `group.<group>.final.full_loss`
- `group_separation.full_loss.ratio`

## Decision log
<!-- append-only: YYYY-MM-DD — <sweep_id> — <status change> — <note> -->
- 2026-04-18 — tick `20260418_102905` — proposed sweep `eoss-H01-proto-counts` (awaiting authorization).
- 2026-04-18 — tick `20260418_104807` — proposed sweep `eoss-H01-proto-counts` (awaiting authorization).
- 2026-04-18 — submitted sweep `eoss-H01-proto-counts` (slurm jobs 12168406–12168414, 9 × resnet-sgd-lr0.05 with boundary×inliers ∈ {5,10,20}²).
- 2026-04-18 — all 9 jobs failed <3 min: `_generation_pool_sizes` in training.py used `max(inliers, x_outlier, y_outlier)` for the inlier pool, but `_build_prototype_injection_subsets` needs `inliers + x_outlier + y_outlier` disjoint samples per class. Patched to use sum.
- 2026-04-18 — resubmitted `eoss-H01-proto-counts` (slurm jobs 12170904–12170912) after pool-size fix.
- 2026-04-18 — cancelled 12170904–12170912 to preempt inf/nan after first steps; lowered default `INIT_SCALE` 0.2→0.05 in `train_eoss.slurm`; resubmitted as 12171216–12171224.
- 2026-04-18 — tick `20260418_142104` — proposed sweep `eoss-H01-proto-counts` (awaiting authorization).
- 2026-04-18 — tick `20260418_185022` — proposed sweep `eoss-H01-proto-counts` (awaiting authorization).
