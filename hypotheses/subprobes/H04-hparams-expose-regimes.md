---
id: H04-hparams-expose-regimes
title: lr / batch / schedule / prototype composition expose distinct curvature-learning regimes
parent: H_primary-eos-selective-tradeoff
status: open
linked_sweeps: []
evidence: []
created: 2026-04-18
updated: 2026-04-18
source: program.md:80
---

# hparams expose distinct curvature-learning regimes

## Question
Does changing learning rate, batch size, schedule, or prototype composition
expose different regimes of curvature-learning interaction?

## Rationale
A "regime" here means a qualitatively different mapping between curvature and
learning — e.g. one setting where `lambda_max` plateaus and loss improves,
another where `lambda_max` grows unboundedly and loss stalls. Identifying
regimes is the first step toward writing them into the paper's analysis.

## Proposed test
Small targeted grids, one factor at a time (program.md:135–140 pattern):
- fixed prototype composition, sweep `lr` over 3 values
- fixed `lr`, sweep `batch` over 3 values
- compare `--lmax-schedule none` vs `drop`
For each cell, compare:
- `global.eos_crossing.lambda_max.step` (entry timing)
- `global.slope.lambda_max.slope` post-crossing (escape / stability)
- `group_separation.full_loss.ratio` at end of training
A regime shift = a discontinuity or reordering of these signals across cells.

## Signals of interest
- `global.eos_crossing.lambda_max.*`
- `global.slope.lambda_max.*`
- `group_separation.full_loss.ratio`
- `hparam.lr`, `hparam.batch_size`, `hparam.lmax_schedule`

## Decision log
<!-- append-only: YYYY-MM-DD — <sweep_id> — <status change> — <note> -->
- 2026-04-18 — tick `20260418_102905` — proposed sweep `eoss-H04-lr-sweep` (awaiting authorization).
- 2026-04-18 — tick `20260418_104807` — proposed sweep `eoss-H04-lr-sweep` (awaiting authorization).
- 2026-04-18 — tick `20260418_142104` — proposed sweep `eoss-H04-lr-sweep` (awaiting authorization).
- 2026-04-18 — submitted sweep `eoss-H04-lr-sweep` (slurm jobs 12172242–12172244, lr ∈ {0.02, 0.05, 0.1} at fixed composition boundary=10, inliers=10, x=5, y=5).
- 2026-04-18 — tick `20260418_185022` — proposed sweep `eoss-H04-lr-sweep` (awaiting authorization).
- 2026-04-18 — tick `20260418_192415` — proposed sweep `eoss-H04-lr-sweep` (awaiting authorization).
- 2026-04-18 — tick `20260418_192620` — proposed sweep `eoss-H04-lr-sweep` (awaiting authorization).
- 2026-04-18 — tick `20260418_201921` — proposed sweep `eoss-H04-lr-sweep` (awaiting authorization).
- 2026-04-18 — tick `20260418_205846` — proposed sweep `eoss-H04-lr-sweep` (awaiting authorization).
