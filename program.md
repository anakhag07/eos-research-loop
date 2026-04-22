# autoresearch

This is an experiment to have the LLM run queue-aware hyperparameter research over the EoSS training stack.

For the external-facing overview (setup, directory map, verification ladder), see [`README.md`](README.md). This file is the in-session operational guide.

## Scope

The goal is not generic model tuning. The goal is to find signal in the interaction between curvature and learning on prototype groups.

Primary question:
- How do curvature-related quantities interact with learning dynamics on prototype subsets?

Primary measurements of interest:
- `lambda_max`
- subset-group `grad_hessian_grad`
- subset-group stability ratio = `grad_hessian_grad / lambda_max`
- cosine similarity metrics already logged for tracked groups
- subset-group loss

The primary entry point for launching experiments is:
- `eoss_training_scripts/launch_ablation.sh`

The primary analysis surface is:
- local W&B directories produced in the project folders under `WANDB_DIR`

## Setup

At the start of a session, work with the user to confirm the study target and then do the following:

1. Read the in-scope files for operational context:
   - `program.md`
   - `eoss_training_scripts/README.md`
   - `eoss_training_scripts/launch_ablation.sh`
   - `eoss_training_scripts/train_eoss.slurm`
   - `edge-of-stochastic-stability-and-memorization/README.md`
2. Treat `bash eoss_training_scripts/launch_ablation.sh --custom ...` as the canonical experiment surface.
3. Confirm the run family naming strategy before launching anything:
   - choose a `PROJECT_NAME` that groups one hypothesis family together
   - keep the name stable across a focused sweep so local W&B outputs land in one place
4. Verify the environment assumptions:
   - Slurm is available
   - the queue may be congested or rate-limited
   - jobs may stay pending for a while
   - W&B may be offline-only, so local files are the source of truth during the run loop
5. Initialize experiment notes for the sweep using `eoss_training_scripts/experiment_manifest_template.md`.

Do not assume direct edits to `training.py` are needed. Start with launcher-level experiments first.

## Allowed Actions

What you SHOULD do by default:
- launch experiments with `launch_ablation.sh`
- use `--custom` grids to vary hyperparameters and prototype subset counts
- dry-run before submit
- submit with `--run`
- poll Slurm automatically while waiting for results
- inspect local W&B run folders and summaries as jobs finish
- generate local CSV/JSON summaries with `eoss_training_scripts/summarize_wandb_runs.py`
- propose follow-up sweeps based on observed signal

What you SHOULD NOT do by default:
- do not start by editing `training.py`
- do not add new measurements unless the required metrics are missing or ambiguously surfaced
- do not treat pending jobs as failures
- do not flood the queue with a very large unreviewed grid when a smaller targeted sweep would answer the question faster

Escalation path if the launcher surface is insufficient:
1. Inspect `eoss_training_scripts/`
2. Inspect `train_eoss.slurm`
3. Inspect core measurement or logging code in `edge-of-stochastic-stability-and-memorization/`
4. Make the smallest useful code change and update docs

## Research Objective

The objective is to identify interpretable signal in prototype-group behavior, especially around the interaction between curvature and learning.

The key decision questions for each sweep are:
- Does `lambda_max` co-vary with learning on specific prototype groups?
- Does subset-group `grad_hessian_grad / lambda_max` separate groups better than raw `grad_hessian_grad` or raw `lambda_max`?
- Do cosine-alignment metrics appear to precede or predict reductions in subset-group loss?
- Does changing learning rate, batch size, schedule, or prototype composition expose different regimes of curvature-learning interaction?

The goal is not simply the lowest global loss. Prefer experiments that clarify mechanism.

## Standard Experiment Loop

Run this loop continuously unless the user redirects you.

1. Read the current launcher capabilities and existing experiment notes.
2. Form a small, hypothesis-driven grid through `launch_ablation.sh --custom`.
3. Dry-run first and inspect the emitted `sbatch` commands.
4. Submit the jobs with `--run`.
5. Poll Slurm automatically:
   - use queue status to distinguish `PENDING`, `RUNNING`, `COMPLETED`, and failed states
   - expect queue delays and rate limits
   - if the cluster only allows a small number of jobs in flight, wait and submit follow-up batches later
6. As results land, inspect:
   - Slurm logs in `eoss_training_scripts/logs/`
   - local W&B outputs in the project directory
   - derived run summaries from `eoss_training_scripts/summarize_wandb_runs.py`
7. Compare runs using the primary measurements:
   - `lambda_max`
   - subset-group `grad_hessian_grad`
   - subset-group stability ratio = `grad_hessian_grad / lambda_max`
   - cosine similarity metrics
   - subset-group loss
8. Record the sweep decision in the manifest:
   - keep exploring this direction
   - narrow the grid
   - change one factor
   - stop pursuing the hypothesis
9. Launch the next focused sweep.

## Queue Behavior

Slurm waiting is part of the workflow.

Rules:
- Poll and wait automatically.
- Do not stop just because jobs are pending.
- Prefer smaller, targeted batches when the queue is constrained.
- When the queue clears, continue launching the next sweep without asking for confirmation.
- If jobs fail due to infrastructure issues, record that separately from scientific conclusions.

## Analysis Rules

Treat prototype groups as first-class analysis units.

For each run, look for:
- curvature increases without corresponding learning
- subset-group learning that occurs only for certain prototype classes
- stability-ratio differences across groups
- abrupt regime changes when `lambda_max` crosses a threshold or after `LMAX_SCHEDULE=drop`
- cosine-alignment changes that appear before subset-loss improvement

Prefer small grids that isolate one question at a time, for example:
- fixed prototype composition, sweep `lr`
- fixed `lr`, sweep `batch`
- fixed optimizer/schedule, sweep subset counts
- compare `train` versus `val` prototype modes
- compare `generate` versus `from:<path-or-run>` sources

## Metrics And Derived Columns

Use raw logged metrics when available, but derive stability ratios locally when needed.

Definitions:
- global stability ratio = `grad_hessian_grad / lambda_max`
- subset stability ratio = `subset_grad_hessian_grad / lambda_max`

When using derived metrics:
- verify the numerator and denominator are present in the local W&B summary or run history
- keep the raw columns alongside the ratio
- do not silently substitute one metric for another

If a needed metric is missing from summaries but appears to exist in W&B history, extend the local summarization step before editing training code.

## Output Artifacts

Maintain these artifacts during the experiment loop:

1. A sweep manifest derived from `eoss_training_scripts/experiment_manifest_template.md`
2. Local run summaries produced by:
   - `python eoss_training_scripts/summarize_wandb_runs.py --wandb-root <project-dir> --output runs.csv`
3. Short decision notes stating:
   - the hypothesis
   - the launch command
   - the important metrics
   - the current conclusion
   - the next sweep to run

Do not rely on a tiny handwritten TSV as the primary record. The local W&B outputs and the generated run summary are the source of truth.

## Non-Interactive Invocation

For loop-driven / unattended use (tmux sessions, `/loop`, future `/research-tick`),
invoke the analysis tools without any GUI dependency:

1. Environment (conda env `eoss`):
   ```bash
   source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
   conda activate eoss
   ```

2. Per-run tabular summary — already non-interactive:
   ```bash
   python eoss_training_scripts/summarize_wandb_runs.py \
       --wandb-root <project-dir> \
       --output runs.csv
   ```

3. Plots without a display — use `--headless`, which implies `--no-show`, `--save`,
   and a timestamped save-dir under `plots/<project>/<timestamp>/`:
   ```bash
   MPLBACKEND=Agg python generate_plots_wandb.py --headless
   ```
   `generate_plots_wandb.py` also auto-switches to the Agg backend when `$DISPLAY`
   is unset and `$MPLBACKEND` is not overridden, so the explicit env var is only
   belt-and-braces.

4. One-shot combined invocation is in `run-once.sh <project-dir>`:
   ```bash
   bash run-once.sh eoss-resnet-baselines
   ```
   It writes `runs.csv` to the project dir and a timestamped PNG directory under
   `plots/<project>/`.

5. Numerical signal extraction on a single run (Stage 1 surface) — emits JSON
   to stdout, suitable for consumption by the research loop:
   ```bash
   # EoS threshold crossing (skip init transient)
   python -m eos_signals extract \
       --run z7588exm --project resnet-0.05 \
       --metric lambda_max --signal crossing \
       --threshold 2/eta --eta 0.05 --start-step 100 --compact

   # Linear slope of lambda_max across a window
   python -m eos_signals extract \
       --run z7588exm --project resnet-0.05 \
       --metric lambda_max --signal slope \
       --start-step 1000 --end-step 5000 --compact

   # Stability ratio summary (grad_hessian_grad / lambda_max)
   python -m eos_signals extract \
       --run z7588exm --project resnet-0.05 \
       --signal stability_ratio \
       --grad-col grad_hessian_grad --lambda-col lambda_max --compact
   ```
   The canonical list of primitives and their inputs is in
   `eos_signals/registry.yaml`. The same primitives back the threshold / crossing
   annotations drawn by `generate_plots_wandb.py`.

6. Hypothesis registry (Stage 3 surface) — `hypotheses/` holds one markdown
   file per research hypothesis with YAML frontmatter (id, status,
   linked_sweeps, evidence, created, updated, source). Statuses: `open`,
   `supported`, `refuted`, `stalled`. The index is auto-generated:
   ```bash
   python hypotheses/rebuild_index.py  # rewrites hypotheses/INDEX.md
   ```
   Seeds: `H01-lambda-learning-covariance`, `H02-stability-ratio-group-separator`,
   `H03-cosine-precedes-loss-drop`, `H04-hparams-expose-regimes` — the 4
   open questions from the Research Objective section above.

   Append-only: when a sweep produces evidence for or against a hypothesis,
   add a line under that file's `Decision log` and update `linked_sweeps` +
   `updated` + `status` in the frontmatter, then rerun `rebuild_index.py`.

7. Sweep-level digest (Stage 2 surface) — walks a whole W&B project, runs every
   applicable primitive across every run × every detected prototype group, and
   writes both a machine-readable digest and a companion PNG bundle:
   ```bash
   python eoss_training_scripts/sweep_digest.py \
       --project resnet-0.05 \
       --samples 5000
   # → digests/resnet-0.05/<ts>/{digest.json, digest.csv, digest_plots/}
   ```
   - `digest.json` is the loop's input; flat signal keys use the naming scheme
     `global.<primitive>.<metric>.<field>` and `group.<name>.<primitive>.<metric>.<field>`.
   - `digest.csv` flattens the same data for pandas/spreadsheet use.
   - `digest_plots/<run_id>_<metric>.png` shows the same threshold + crossing
     the primitives detected, one PNG per (run, metric) pair.
   - Raise `--samples` or add `--scan` for dense sampling. Use `--run-filter`
     and `--limit` for fast iteration.

8. Autoresearch tick (Stage 4 surface) — one iteration of the research loop:
   builds the digest, scores every open hypothesis, proposes (dry-run only)
   the next sweep per hypothesis, writes a timestamped report, and appends a
   decision-log line to each hypothesis file. Driven by the `/research-tick`
   skill in `.claude/skills/research-tick/SKILL.md`.
   ```bash
   python research_tick.py --projects resnet-0.05 --samples 5000
   # → reports/<ts>/{report.md, proposals.json, digests/<project>/...}
   ```
   The tick NEVER submits sbatch. The report's `--run` command blocks are
   authorization surfaces. The user authorizes by replying `run` to the
   assistant; only then does the assistant execute the stored `--run` form.

   Proposal logic per hypothesis is hardcoded in `research_tick.py:PROPOSERS`.
   Edit that dict to change what a hypothesis asks for next. Infra-blocked
   hypotheses (currently H02, H03) emit a concrete patch sketch instead of a
   sweep command.

## Example Launcher Pattern

Use dry-run first:

```bash
bash eoss_training_scripts/launch_ablation.sh --custom \
  --models "mlp" \
  --optimizers "sgd" \
  --lrs "0.005 0.01" \
  --batches "32 128" \
  --lmax-schedule "none drop" \
  --input-prototypes-modes "val" \
  --input-prototype-sources "generate" \
  --input-boundary-counts "5 10" \
  --input-inliers-counts "5 10" \
  --input-x-outlier-counts "5" \
  --input-y-outlier-counts "5" \
  --project-name "proto-curvature-sweep-a"
```

Then submit:

```bash
bash eoss_training_scripts/launch_ablation.sh --custom \
  --models "mlp" \
  --optimizers "sgd" \
  --lrs "0.005 0.01" \
  --batches "32 128" \
  --lmax-schedule "none drop" \
  --input-prototypes-modes "val" \
  --input-prototype-sources "generate" \
  --input-boundary-counts "5 10" \
  --input-inliers-counts "5 10" \
  --input-x-outlier-counts "5" \
  --input-y-outlier-counts "5" \
  --project-name "proto-curvature-sweep-a" \
  --run
```

## When To Edit Code

Only edit code when one of these is true:
- the launcher cannot express the needed experiment
- the Slurm wrapper cannot route required metadata or flags
- the local W&B summaries do not expose the metrics needed to compare runs
- a metric of interest is not logged at all

When code changes are needed:
- keep them minimal
- update docs at the same time
- validate the wiring before returning to the sweep loop
