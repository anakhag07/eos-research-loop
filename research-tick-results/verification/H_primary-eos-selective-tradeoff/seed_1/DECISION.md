---
result: fail
stage: seed_1
hypothesis: H_primary-eos-selective-tradeoff
run_id: mrjq80yn
project: eoss-Hprimary-seed_1
slurm_job: 12182636
figure: key_figure.png
date: 2026-04-18
---

# H_primary seed_1 probe — FAIL (config too short, not a scientific refutation)

## Config

- slurm job `12182636` (W&B run `mrjq80yn`) — MLP + full-GD + MSE + lr=0.05 + seed 7777
- 500-step probe, cifar10 2-class, batch 10000 (full-batch), composition
  boundary=10, inliers=10, x_outlier=5, y_outlier=5
- Wall time: 12s

## Why this fails the gate

Both necessary conditions for testing the H_primary claim are missing:

1. **EoS not entered.** `global.eos_crossing.lambda_max.step: None` — `lambda_max`
   never reached the 2/η = 40 threshold in 500 steps. Without an EoS regime
   there is no tradeoff to measure.
2. **Per-group trajectories degenerate to a single point.** Every
   `group.<g>.*` primitive returns `n: 1` because `utils.frequency.subset_metrics_rule`
   fires at step 0 and then every 512 steps for batch_size ≥ 32. The 500-step
   window only contains the step-0 log. Crossings, slope onsets, and
   cos_crossing all come back `None` by construction.

## Single-point snapshot (informational only, n=1 per group)

Even with one sample, the stability ratio ordering is suggestive:

| group                   | stability_ratio | grad_vmax_cos² |
|-------------------------|-----------------|----------------|
| `x_outlier`             | 0.192           | 3.9e-04        |
| `injected_x_outlier`    | 0.197           | 3.9e-05        |
| `injected_boundary`     | 0.461           | 1.7e-03        |
| `boundary`              | 0.462           | 3.3e-03        |
| `y_outlier`             | 0.500           | 1.2e-03        |
| `injected_y_outlier`    | 0.539           | 1.1e-03        |
| `inliers`               | 0.594           | 2.5e-03        |
| `injected_inliers`      | 0.595           | 2.8e-03        |

The x-outlier groups sit at ~3× lower curvature alignment than inliers. This
is the direction H_primary predicts, but n=1 proves nothing.

## Retune for rerun

- Bump `budget.yaml:jobs.probe_steps` from 500 → 4000. At 4000 steps with
  `subset_metrics_rule` firing every 512 steps (bs ≥ 32), each group will
  have ~8 samples — enough for `crossing_step` and
  `first_window_with_negative_slope` to return meaningful results.
- Do not change the `seed_1` stage config otherwise. Rerun and re-digest.
- If λ_max still doesn't cross 2/η at 4000 steps, the next retune is either
  (a) a longer probe, or (b) a different canonical hparam (e.g. larger lr)
  — but pick that after seeing the 4000-step trajectory.

## Stage advance: NO

Stay at `stage: seed_1`. Next tick's proposer will pick up the bumped
`probe_steps` automatically.
