---
id: H_primary-eos-selective-tradeoff
title: Edge of Stability selectively trades performance across prototype groups
status: open
stage: seed_1
verification_ladder: [seed_1, seeds_N, archs, losses, optimizers]
subprobes: [H01-lambda-learning-covariance, H02-stability-ratio-group-separator, H03-cosine-precedes-loss-drop, H04-hparams-expose-regimes]
linked_sweeps: []
evidence: []
created: 2026-04-18
updated: 2026-04-18
source: program.md / NeurIPS-2026-Draft/main.tex
---

# Primary hypothesis — EoS selectively trades performance across prototype groups

## Claim

Once training enters the Edge of Stability (`lambda_max ≥ 2/η` for SGD / full-GD,
`38/η` for Adam), the dynamics do not treat prototype groups uniformly: some
groups (e.g. `boundary`, `injected_x_outlier`) experience a qualitatively
different curvature-learning interaction than others (`inliers`,
`injected_inliers`), producing a *selective* tradeoff in loss reduction, gradient
alignment with `v_1`, and post-crossing stability.

The claim is group-conditional. Global EoS entry is a necessary precondition —
we only care about the regime *after* the first global `2/η` crossing — but the
interesting signal is the *difference between groups* measured from that point
forward.

## What "confirmed" means

We walk the ladder in `research-tick-results/budget.yaml:verification_stages`:

1. **`seed_1`** — one run, MLP + full-GD + MSE + seed 7777. Look for a strong
   visual signal in per-group plots: `group.<g>.eos_crossing.lambda_max.step`
   differs across groups by more than their within-group noise; at least one
   prototype group shows a distinct `group.<g>.loss_decline_onset.full_loss.step`
   relative to others. *Gate: visual — user confirms.*
2. **`seeds_N`** — 5 seeds at the same config. The sign of the per-group ordering
   (e.g. `boundary` crosses before `x_outlier`) must hold on ≥ 4/5 seeds.
   *Gate: auto.*
3. **`archs`** — repeat across MLP, CNN, ResNet at 3 seeds each. *Gate: visual.*
4. **`losses`** — add CE to MSE at the same 3 archs × 3 seeds. *Gate: visual.*
5. **`optimizers`** — add SGD and Adam (38/η threshold) at MLP × 3 seeds ×
   {MSE, CE}. *Gate: visual.*

Only after stage 5 passes does the claim become a verified finding and get
written to `NeurIPS-2026-Draft/research-agents/H_primary.tex` (Phase E).

## Current stage: `seed_1`

All in-flight experiments (H01 proto-count grid, H04 lr-sweep) are running
resnet-sgd rather than the canonical MLP-full-GD at seed 7777. The next
`probe` submitted by the tick should be the MLP-full-GD-mse seed_1 config
from `budget.yaml:verification_stages[0]`, scoped to ~500 steps with the
queue drained per the budget fence.

## Signals of interest

From the per-run digest (all per-group, indexed by `<g>` ∈ `{boundary, inliers,
injected_boundary, injected_inliers, injected_x_outlier, injected_y_outlier}`):

- `group.<g>.eos_crossing.lambda_max.step` — when this group enters EoS
- `group.<g>.loss_decline_onset.full_loss.{step,slope,r2}` — first 10-step window
  of sustained full_loss decrease after the global crossing
- `group.<g>.cos_crossing.grad_vmax_cos2.{step,value_at_step}` — first step
  where the gradient aligns with `v_1` at ≥ 0.1 squared-cosine
- `group.<g>.stability_ratio.{mean,max,last}` — `grad_hessian_grad / lambda_max`
- `group.<g>.slope.full_loss.slope` — second-half linear slope
- `group_separation.{full_loss,lambda_max,stability_ratio}.ratio` — how much the
  between-group variance dominates within-group variance on each metric

## Why this hypothesis subsumes H01–H04

Each open hypothesis measures one facet of the same claim:

- **H01** (`lambda-learning-covariance`) — per-group `lambda_max` vs
  `full_loss` slope correlation → the *co-variance* shape of the tradeoff.
- **H02** (`stability-ratio-group-separator`) — whether the `grad·Hg / λ_max`
  ratio is a stronger group-separator than its parts → the *metric choice*
  for measuring the tradeoff.
- **H03** (`cosine-precedes-loss-drop`) — whether `grad_vmax_cos2` leads
  `loss_decline_onset` per group → the *temporal order* inside the tradeoff.
- **H04** (`hparams-expose-regimes`) — whether lr/batch/composition produces
  qualitatively distinct tradeoff shapes → the *knobs* that move the tradeoff.

They are sub-probes, not competitors: H_primary advances on the ladder only
when the sub-probes' signals point the same direction for the canonical
configuration at each stage.

## Decision log

<!-- append-only: YYYY-MM-DD — <tick/sweep_id> — <status change> — <note> -->
- 2026-04-18 — created as collapse of H01–H04; initial stage `seed_1`.
- 2026-04-18 — tick `20260418_192415` — proposed sweep `eoss-Hprimary-seed_1` (awaiting authorization).
- 2026-04-18 — tick `20260418_192620` — proposed sweep `eoss-Hprimary-seed_1` (awaiting authorization).
- 2026-04-18 — submitted sweep `eoss-Hprimary-seed_1` (slurm job 12182636, mlp-fullgd-mse-lr0.05, 500-step probe at seed_1 stage).
- 2026-04-18 — seed_1 DECISION: fail (configuration, not science). Run `mrjq80yn` did not reach EoS (`global.eos_crossing.lambda_max.step: None`; threshold=40) and per-group `subset_metrics_rule` only fired once at step 0 (next cadence step = 512 > 500). Bumped `budget.yaml:jobs.probe_steps` 500→4000 so subset trackers get ~8 samples per run. Staying at stage `seed_1`; next tick re-proposes with 4000 steps. Full analysis + single-point stability-ratio snapshot in `research-tick-results/verification/H_primary-eos-selective-tradeoff/seed_1/DECISION.md`.
