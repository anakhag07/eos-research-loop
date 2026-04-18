---
description: Run one iteration of the EoSS autoresearch loop — digest each active project, score hypotheses, propose (dry-run) next sweeps, write a timestamped report, and append decision notes. Never submits sbatch; awaits user "run".
---

# /research-tick

One turn of the self-evolving research loop for `~/projects/eos`. Implements
Stage 4 of `/home/anakhag/.claude/plans/can-you-plan-the-humble-floyd.md`.

## What you (the assistant) do

1. **Parse args.** Default to `--projects resnet-0.05`. If the user passed a
   different project name (e.g. `/research-tick H04-lr-sweep`), use that as
   the project to digest.

2. **Run the tick.** Execute:
   ```bash
   source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh && \
       conda activate eoss && \
       python research_tick.py --projects <project1> [<project2> ...]
   ```
   This will:
     - call `sweep_digest.py` for each project → `reports/<ts>/digests/<project>/`
     - read every open hypothesis in `hypotheses/`
     - produce `reports/<ts>/report.md` + `proposals.json`
     - append a dated decision-log line to each hypothesis file
     - rerun `hypotheses/rebuild_index.py`

3. **Read the report.** Open `reports/<ts>/report.md` and
   `reports/<ts>/proposals.json`.

4. **Summarize to the user** in under ~200 words:
   - one-line status per digested project (N runs, top surprising signals)
   - one line per hypothesis: status, and whether this tick proposed a sweep,
     flagged infra, or skipped
   - list each pending sweep proposal with its W&B project name
   - end with:
     "Reply `run` to submit <the single most ripe proposal>, or `run <Hxx>` to
     pick a specific hypothesis's sweep."

5. **Do NOT execute `--run`.** The tick report is the authorization surface.
   Wait for the user to reply "run" (or "run Hxx") on a subsequent turn. When
   they do, follow the `feedback_run_on_cue.md` memory rule: read the stored
   `run_cmd` from the most recent `proposals.json`, execute it verbatim with
   Bash (you may need to `conda activate eoss` first), and then poll `squeue`
   once to show the queued job ids.

## When to use /research-tick

- At the start of a working session to orient on what sweeps are ripe.
- After jobs complete, to re-score hypotheses against new digests.
- The user may chain it via `/loop 2h /research-tick` for autonomous cadence
  — when invoked that way, still do not submit; write the report and stop.

## Things to check before submitting on "run"

- `sacct -u $USER | head` (already allowlisted) — are we over the MaxJobsPU
  quota? If so, warn the user before firing the sweep.
- `squeue -u $USER` — any in-flight jobs for the same W&B project? If yes,
  ask before launching duplicates.

## Editing proposal logic

Each hypothesis's generator lives in `research_tick.py:PROPOSERS`. To change
what a tick proposes for a given hypothesis, edit the corresponding function
(`propose_H01`, etc.) — not this skill. Keep the generator pure / side-effect
free; the driver handles all I/O.

## Infra-blocked proposals

Some hypotheses (H02, H03) require primitive or digest extensions before a
sweep can meaningfully test them. The tick surfaces these as `kind: "infra"`
with a concrete patch sketch in `infra_note`. If the user says "do the infra"
(or similar), implement the change directly — keeping edits minimal — and
re-run the tick.
