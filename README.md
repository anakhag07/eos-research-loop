# EoS autoresearch loop

A self-evolving research loop investigating whether **Edge of Stability**
dynamics selectively trade off performance across prototype groups
(`boundary`, `inliers`, `injected_x_outlier`, …). The loop reads a registry
of open hypotheses, digests completed W&B sweeps into per-group signals,
proposes the next experiment, writes a dated decision line on each
hypothesis, and halts for user authorization before submitting to Slurm.

The primary claim lives in
[`hypotheses/H_primary-eos-selective-tradeoff.md`](hypotheses/H_primary-eos-selective-tradeoff.md).
It advances through a verification ladder
(`seed_1 → seeds_N → archs → losses → optimizers`) gated by
`research-tick-results/budget.yaml`.

## Directory map

```
research_tick.py                    one iteration of the loop
eos_signals/                        primitive registry + implementations
  registry.yaml                     authoritative primitive list (versioned)
hypotheses/                         one file per hypothesis + auto-index
  H_primary-eos-selective-tradeoff.md
  subprobes/H0{1..4}-*.md
  INDEX.md                          auto-rendered by rebuild_index.py
research-tick-results/
  budget.yaml                       fences: queue, lr grid, ladder stages
  ticks/<ts>/report.md              per-tick proposals
  verification/<hyp>/<stage>/       tracked DECISION.md + key_figure.png
program.md                          in-session operational guide
eoss_training_scripts/              [submodule] Slurm launcher + digest
edge-of-stochastic-stability-and-memorization/  [submodule] training code
NeurIPS-2026-Draft/                 [submodule] paper draft
```

## Setup

```bash
git clone --recursive https://github.com/anakhag07/eos.git
cd eos
conda env create -f environment.yml    # or: conda create -n eoss python=3.11 && pip install -r requirements.txt
conda activate eoss
```

Assumptions: Slurm is available; W&B may be offline (local run folders are
the source of truth); submodules are cloned alongside (`--recursive`).
If you forget `--recursive`, run `git submodule update --init --recursive`.

## Running one tick

```bash
python research_tick.py --projects <wandb-project>
# or, inside Claude Code:
/research-tick <wandb-project>
```

Each tick:

1. digests `<wandb-project>` into
   `research-tick-results/ticks/<ts>/digests/<project>/` (signals + plots);
2. reads every `status: open|ready` hypothesis;
3. emits a proposal per hypothesis (sweep-ready or infra-blocked) and
   writes `ticks/<ts>/report.md` + `proposals.json`;
4. appends a decision-log line to each hypothesis file;
5. halts. Nothing is submitted to Slurm until the user explicitly approves.

## Verification ladder

`H_primary` advances through five stages defined in
[`research-tick-results/budget.yaml`](research-tick-results/budget.yaml)
under `verification_stages`. Each stage has a canonical config (model,
optimizer, loss, seeds) and a `gate: auto|visual`. A stage only advances
when `research-tick-results/verification/<hyp>/<stage>/DECISION.md`
contains `result: pass` (plus a `key_figure.png`). These two files are
the only outputs under `research-tick-results/` tracked in git — they are
the durable state of the loop.

## Hypotheses

See [`hypotheses/INDEX.md`](hypotheses/INDEX.md) for the registry
(auto-rendered by `hypotheses/rebuild_index.py`). Primary hypothesis:
[`H_primary-eos-selective-tradeoff`](hypotheses/H_primary-eos-selective-tradeoff.md).
Sub-probes: H01–H04 under `hypotheses/subprobes/`.

## Further reading

- [`program.md`](program.md) — operational guide for running the loop in a
  Claude Code session (allowed actions, launcher surface, file conventions).
- [`eos_signals/registry.yaml`](eos_signals/registry.yaml) — primitive
  definitions referenced by every `global.*` / `group.<g>.*` signal name
  in a digest.
