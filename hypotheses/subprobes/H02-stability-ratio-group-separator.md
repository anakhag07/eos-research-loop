---
id: H02-stability-ratio-group-separator
title: stability_ratio separates prototype groups better than raw grad_hessian_grad or lambda_max
parent: H_primary-eos-selective-tradeoff
status: open
linked_sweeps: []
evidence: []
created: 2026-04-18
updated: 2026-04-18
source: program.md:78
---

# stability_ratio separates groups better than either numerator or denominator alone

## Question
Does subset-group `grad_hessian_grad / lambda_max` separate prototype groups
better than raw `grad_hessian_grad` or raw `lambda_max`?

## Rationale
If the ratio is just a restatement of the numerator or denominator, it adds
no explanatory power. If instead it captures the *alignment* of the gradient
with the top eigendirection in a group-specific way, it should achieve
strictly higher between-group / within-group variance than either constituent.

## Proposed test
On any sweep with ≥ 2 prototype groups logged:
- compute `group_separation.grad_hessian_grad.ratio`
- compute `group_separation.lambda_max.ratio`
- compute `group_separation.stability_ratio.ratio` (requires extending
  `sweep_digest.py` to materialize the ratio series per group and feed it to
  `group_separation`)
Verdict: supported iff stability_ratio separation > max(numerator, denominator)
separation on ≥ 2 sweeps.

## Signals of interest
- `group_separation.grad_hessian_grad.ratio`
- `group_separation.lambda_max.ratio`
- `group_separation.stability_ratio.ratio` *(not yet produced; see next-sweep note)*

## Next sweep / infra need
`sweep_digest.py` currently materializes stability_ratio only as a per-group
scalar summary. To test this hypothesis we need a `stability_ratio` time
series per group passed to `group_separation`. Small extension; worth doing
before running a dedicated sweep.

## Decision log
<!-- append-only: YYYY-MM-DD — <sweep_id> — <status change> — <note> -->
- 2026-04-18 — tick `20260418_102905` — infra-blocked: Blocked on infra: `sweep_digest.py` emits `stability_ratio` only as a per-group scalar. To compare group separation *on the ratio*, the digest needs to materialize the ratio as a time series per group and pass those DataFrames to `group_separation`.
- 2026-04-18 — tick `20260418_104807` — infra-blocked: Blocked on infra: `sweep_digest.py` emits `stability_ratio` only as a per-group scalar. To compare group separation *on the ratio*, the digest needs to materialize the ratio as a time series per group and pass those DataFrames to `group_separation`.
- 2026-04-18 — tick `20260418_142104` — infra-blocked: Blocked on infra: `sweep_digest.py` emits `stability_ratio` only as a per-group scalar. To compare group separation *on the ratio*, the digest needs to materialize the ratio as a time series per group and pass those DataFrames to `group_separation`.
- 2026-04-18 — tick `20260418_185022` — infra-blocked: Blocked on infra: `sweep_digest.py` emits `stability_ratio` only as a per-group scalar. To compare group separation *on the ratio*, the digest needs to materialize the ratio as a time series per group and pass those DataFrames to `group_separation`.
