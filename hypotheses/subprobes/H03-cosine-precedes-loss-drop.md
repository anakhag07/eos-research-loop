---
id: H03-cosine-precedes-loss-drop
title: cosine-alignment metrics precede reductions in subset-group loss
parent: H_primary-eos-selective-tradeoff
status: ready
linked_sweeps: []
evidence: []
created: 2026-04-18
updated: 2026-04-18
source: program.md:79
---

# Cosine-alignment precedes subset-group loss improvement

## Question
Do cosine-alignment metrics (e.g. `grad_vmax_cos2` — squared cosine between
the gradient and the top Hessian eigenvector) appear to precede or predict
reductions in subset-group loss?

## Rationale
If cosine alignment rises *before* subset loss drops, alignment is a leading
indicator — useful for early stopping, for detecting when a group is about to
start learning, and for the loop's own sweep-scheduling logic. If it only
co-occurs with loss drop, it is just a symptom.

## Proposed test
For each prototype group on a run:
- detect the first step where `full_loss` enters a sustained decline (use
  `slope_in_window` over a forward rolling window)
- detect the first step where `grad_vmax_cos2` exceeds a threshold (primitive
  `crossing_step`)
Compare: is the cos² crossing reliably earlier than the loss-decline onset
across ≥ 2 sweeps?

## Signals of interest
- `group.<group>.slope.full_loss.slope` within forward windows
- first-crossing of `group.<group>.grad_vmax_cos2`

## Next sweep / infra need
Need per-group `crossing_step` applied to `grad_vmax_cos2` in
`sweep_digest.py` (small addition). Forward-window slope scan is a new
primitive (approx: `first_window_with_negative_slope`) — candidate for
Stage 5 signal proposal.

## Decision log
<!-- append-only: YYYY-MM-DD — <sweep_id> — <status change> — <note> -->
- 2026-04-18 — tick `20260418_102905` — infra-blocked: Blocked on infra: the digest does not currently extract per-group `grad_vmax_cos2` crossings nor a forward-window slope onset for `full_loss`. Both are small additions.
- 2026-04-18 — tick `20260418_104807` — infra-blocked: Blocked on infra: the digest does not currently extract per-group `grad_vmax_cos2` crossings nor a forward-window slope onset for `full_loss`. Both are small additions.
- 2026-04-18 — tick `20260418_142104` — infra-blocked: Blocked on infra: the digest does not currently extract per-group `grad_vmax_cos2` crossings nor a forward-window slope onset for `full_loss`. Both are small additions.
- 2026-04-18 — infra — unblocked: added `first_window_with_negative_slope` primitive (eos_signals schema v2) and extended `sweep_digest.py:_run_group_signals` to emit `group.<g>.cos_crossing.grad_vmax_cos2.{step,value_at_step}` (threshold 0.1) and `group.<g>.loss_decline_onset.full_loss.{step,slope,r2}` (window=10, min_r2=0.5). Next tick will have these fields available for every prototype group.
- 2026-04-18 — tick `20260418_192415` — proposed sweep `eoss-H03-cos-lead` (awaiting authorization).
- 2026-04-18 — tick `20260418_192620` — proposed sweep `eoss-H03-cos-lead` (awaiting authorization).
- 2026-04-18 — tick `20260418_201921` — proposed sweep `eoss-H03-cos-lead` (awaiting authorization).
- 2026-04-18 — tick `20260418_205846` — proposed sweep `eoss-H03-cos-lead` (awaiting authorization).
