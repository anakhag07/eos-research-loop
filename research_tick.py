#!/usr/bin/env python3
"""research_tick.py — one iteration of the EoSS research loop.

Steps:
  1. Build a sweep digest for each --project (reuses eoss_training_scripts/sweep_digest.py).
  2. Read every hypothesis in `hypotheses/` with status == open.
  3. For each open hypothesis, call its proposal generator:
       - emits a dry-run launch_ablation.sh command and a paired --run form,
       - or emits an "infra-blocked" note if a required primitive/extension is missing.
  4. Write research-tick-results/ticks/<ts>/report.md bundling digests + proposals.
  5. Append a dated decision-log line to each hypothesis's markdown file.
  6. Rerun `hypotheses/rebuild_index.py`.

The script DOES NOT submit sbatch. The assistant driving the loop reads
report.md, summarizes proposals to the user, and executes the --run command
only on explicit user authorization ("run").
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_ROOT / "eoss_training_scripts"
HYPOTHESES_DIR = REPO_ROOT / "hypotheses"
RESULTS_ROOT = REPO_ROOT / "research-tick-results"
REPORTS_DIR = RESULTS_ROOT / "ticks"
PROBES_DIR = RESULTS_ROOT / "probes"
VERIFICATION_DIR = RESULTS_ROOT / "verification"
BUDGET_FILE = RESULTS_ROOT / "budget.yaml"
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


@dataclass
class Hypothesis:
    id: str
    title: str
    status: str
    path: Path
    frontmatter: Dict[str, Any]
    body: str


@dataclass
class Proposal:
    hypothesis_id: str
    kind: str  # "sweep" | "infra" | "skip"
    project_name: str
    summary: str
    dry_run_cmd: Optional[str] = None
    run_cmd: Optional[str] = None
    infra_note: Optional[str] = None
    signals_to_watch: List[str] = field(default_factory=list)


# ---------- budget + queue guard ---------- #

@dataclass
class QueueState:
    empty: bool
    pending: List[str] = field(default_factory=list)
    running: List[str] = field(default_factory=list)


def queue_guard() -> QueueState:
    """Return the user's current slurm queue state.

    The tick's budget rule is: advance a hypothesis stage only when the queue
    is entirely empty. If anything is pending or running, we monitor and exit
    without submitting.
    """
    user = os.environ.get("USER", "")
    if not user:
        return QueueState(empty=True)
    try:
        out = subprocess.run(
            ["squeue", "-u", user, "--noheader", "--format=%i %T"],
            capture_output=True, text=True, check=False, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return QueueState(empty=True)
    pending, running = [], []
    for line in out.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        jid, state = parts[0], parts[1]
        if state == "PENDING":
            pending.append(jid)
        elif state == "RUNNING":
            running.append(jid)
    return QueueState(empty=not (pending or running), pending=pending, running=running)


def load_budget() -> Dict[str, Any]:
    """Minimal YAML-ish reader — we only consume scalar + list fields, not deep nesting."""
    if not BUDGET_FILE.exists():
        return {}
    # Defer to PyYAML if available; else lean on the frontmatter parser shape.
    try:
        import yaml  # type: ignore
        return yaml.safe_load(BUDGET_FILE.read_text(encoding="utf-8")) or {}
    except ImportError:
        # Conservative fallback: tick will treat budget as empty -> cautious defaults.
        return {}


def submit_probe(*, hypothesis_id: str, stage: str, model: str, optimizer: str,
                 lr: float, loss: str, project_name: str,
                 probe_steps: int = 500, input_proto_counts: Optional[Dict[str, int]] = None,
                 dry_run: bool = True) -> Dict[str, Any]:
    """Render a short probe-run launch command.

    The probe submits ONE job with reduced STEPS to verify that the EoS crossing
    (lambda_max >= 2/eta for sgd/fullgd, 38/eta for adam) is reachable under the
    stage's hyperparameters before committing to the full verification sweep.
    Returns {"cmd": <str>, "project": <str>}. Execution is left to the caller so
    the tick can choose to auto-submit (queue empty + under budget) or emit a
    dry-run for the user.
    """
    counts = input_proto_counts or {"boundary": 10, "inliers": 10, "x_outlier": 5, "y_outlier": 5}
    parts = [
        "bash eoss_training_scripts/launch_ablation.sh --custom",
        f'--models "{model}"',
        f'--optimizers "{optimizer}"',
        f'--lrs "{lr}"',
        '--batches "128"',
        f'--loss "{loss}"',
        '--lmax-schedule "none"',
        '--input-prototypes-modes "val"',
        '--input-prototype-sources "generate"',
        f'--input-boundary-counts "{counts.get("boundary", 10)}"',
        f'--input-inliers-counts "{counts.get("inliers", 10)}"',
        f'--input-x-outlier-counts "{counts.get("x_outlier", 5)}"',
        f'--input-y-outlier-counts "{counts.get("y_outlier", 5)}"',
        f'--steps {probe_steps}',
        f'--project-name "{project_name}"',
    ]
    if not dry_run:
        parts.append("--run")
    cmd = " \\\n    ".join(parts)
    return {
        "cmd": cmd,
        "project": project_name,
        "kind": "probe",
        "hypothesis_id": hypothesis_id,
        "stage": stage,
        "probe_steps": probe_steps,
    }


# ---------- frontmatter (minimal, no PyYAML dep) ---------- #

def _parse_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    body = m.group(1)
    rest = m.group(2)
    out: Dict[str, Any] = {}
    current_list_key: Optional[str] = None
    for raw in body.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if current_list_key and line.lstrip().startswith("- "):
            out[current_list_key].append(line.lstrip()[2:].strip())
            continue
        current_list_key = None
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if val == "[]":
            out[key] = []
        elif val == "":
            out[key] = []
            current_list_key = key
        elif val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            out[key] = [s.strip() for s in inner.split(",")] if inner else []
        else:
            out[key] = val
    return out, rest


def _load_hypotheses() -> List[Hypothesis]:
    out = []
    for p in sorted(HYPOTHESES_DIR.rglob("H*.md")):
        if p.name == "INDEX.md":
            continue
        text = p.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(text)
        out.append(Hypothesis(
            id=fm.get("id", p.stem),
            title=fm.get("title", p.stem),
            status=fm.get("status", "unknown"),
            path=p,
            frontmatter=fm,
            body=body,
        ))
    return out


# ---------- digest driver ---------- #

def _run_digest(project: str, out_root: Path, samples: int) -> Optional[Path]:
    digest_dir = out_root / "digests" / project
    digest_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "sweep_digest.py"),
        "--project", project,
        "--samples", str(samples),
        "--out-dir", str(digest_dir),
    ]
    print(f"[tick] digest: {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        return None
    print(proc.stdout.strip())
    return digest_dir


# ---------- proposal generators ---------- #

def _dry_and_run(base: List[str]) -> tuple[str, str]:
    dry = " \\\n    ".join(base)
    run = " \\\n    ".join(base + ["--run"])
    return dry, run


def propose_H01(hyp: Hypothesis, project_base: str) -> Proposal:
    """Vary prototype composition at fixed hparams."""
    project_name = f"{project_base}-H01-proto-counts"
    base = [
        "bash eoss_training_scripts/launch_ablation.sh --custom",
        '--models "resnet"',
        '--optimizers "sgd"',
        '--lrs "0.05"',
        '--batches "128"',
        '--lmax-schedule "none"',
        '--input-prototypes-modes "val"',
        '--input-prototype-sources "generate"',
        '--input-boundary-counts "5 10 20"',
        '--input-inliers-counts "5 10 20"',
        '--input-x-outlier-counts "5"',
        '--input-y-outlier-counts "5"',
        f'--project-name "{project_name}"',
    ]
    dry, run = _dry_and_run(base)
    return Proposal(
        hypothesis_id=hyp.id,
        kind="sweep",
        project_name=project_name,
        summary=(
            "Hold all hparams fixed; vary boundary/inlier counts. Then compare "
            "`group.<g>.eos_crossing.lambda_max.step` vs `group.<g>.slope.full_loss.slope` "
            "per composition to test whether lambda_max tracks group-specific learning."
        ),
        dry_run_cmd=dry,
        run_cmd=run,
        signals_to_watch=[
            "group.<group>.eos_crossing.lambda_max.step",
            "group.<group>.slope.full_loss.slope",
            "group.<group>.final.full_loss",
            "group_separation.full_loss.ratio",
        ],
    )


def propose_H02(hyp: Hypothesis, project_base: str) -> Proposal:
    return Proposal(
        hypothesis_id=hyp.id,
        kind="infra",
        project_name="",
        summary=(
            "Blocked on infra: `sweep_digest.py` emits `stability_ratio` only as a "
            "per-group scalar. To compare group separation *on the ratio*, the digest "
            "needs to materialize the ratio as a time series per group and pass those "
            "DataFrames to `group_separation`."
        ),
        infra_note=(
            "In `eoss_training_scripts/sweep_digest.py:_run_group_signals`, after the "
            "per-group `stability_ratio_summary` call, build a DataFrame of the ratio "
            "series via `primitives.stability_ratio_series`, collect them by group, and "
            "emit `group_separation.stability_ratio.ratio` alongside the existing "
            "`group_separation.full_loss.ratio` / `group_separation.lambda_max.ratio`."
        ),
        signals_to_watch=[
            "group_separation.grad_hessian_grad.ratio",
            "group_separation.lambda_max.ratio",
            "group_separation.stability_ratio.ratio  (needs infra)",
        ],
    )


def propose_H03(hyp: Hypothesis, project_base: str) -> Proposal:
    """Now that infra is in place, propose a sweep that exercises per-group
    grad_vmax_cos2 and loss_decline_onset on the canonical MLP-fullGD config.
    """
    project_name = f"{project_base}-H03-cos-lead"
    base = [
        "bash eoss_training_scripts/launch_ablation.sh --custom",
        '--models "mlp"',
        '--optimizers "fullgd"',
        '--lrs "0.05"',
        '--batches "128"',
        '--loss "mse"',
        '--lmax-schedule "none"',
        '--input-prototypes-modes "val"',
        '--input-prototype-sources "generate"',
        '--input-boundary-counts "10"',
        '--input-inliers-counts "10"',
        '--input-x-outlier-counts "5"',
        '--input-y-outlier-counts "5"',
        f'--project-name "{project_name}"',
    ]
    dry, run = _dry_and_run(base)
    return Proposal(
        hypothesis_id=hyp.id,
        kind="sweep",
        project_name=project_name,
        summary=(
            "Infra landed (eos_signals v2 + sweep_digest per-group cos_crossing / "
            "loss_decline_onset). Run MLP-fullGD-mse at the canonical composition and "
            "check whether `group.<g>.cos_crossing.grad_vmax_cos2.step` precedes "
            "`group.<g>.loss_decline_onset.full_loss.step` per prototype group."
        ),
        dry_run_cmd=dry,
        run_cmd=run,
        signals_to_watch=[
            "group.<g>.cos_crossing.grad_vmax_cos2.step",
            "group.<g>.loss_decline_onset.full_loss.step",
            "group.<g>.loss_decline_onset.full_loss.slope",
        ],
    )


def propose_H04(hyp: Hypothesis, project_base: str) -> Proposal:
    project_name = f"{project_base}-H04-lr-sweep"
    base = [
        "bash eoss_training_scripts/launch_ablation.sh --custom",
        '--models "resnet"',
        '--optimizers "sgd"',
        '--lrs "0.02 0.05 0.1"',
        '--batches "128"',
        '--lmax-schedule "none"',
        '--input-prototypes-modes "val"',
        '--input-prototype-sources "generate"',
        '--input-boundary-counts "10"',
        '--input-inliers-counts "10"',
        '--input-x-outlier-counts "5"',
        '--input-y-outlier-counts "5"',
        f'--project-name "{project_name}"',
    ]
    dry, run = _dry_and_run(base)
    return Proposal(
        hypothesis_id=hyp.id,
        kind="sweep",
        project_name=project_name,
        summary=(
            "Fix composition, sweep lr over {0.02, 0.05, 0.1}. Look for a discontinuity "
            "or reordering in `global.eos_crossing.lambda_max.step` and "
            "`global.slope.lambda_max.slope` post-crossing as lr rises."
        ),
        dry_run_cmd=dry,
        run_cmd=run,
        signals_to_watch=[
            "global.eos_crossing.lambda_max.step",
            "global.slope.lambda_max.slope",
            "group_separation.full_loss.ratio",
        ],
    )


# ---------- H_primary stage machine ---------- #

LADDER_STAGES = ["seed_1", "seeds_N", "archs", "losses", "optimizers"]


def _stage_config(budget: Dict[str, Any], stage_name: str) -> Optional[Dict[str, Any]]:
    for entry in (budget.get("verification_stages") or []):
        if entry.get("name") == stage_name:
            return entry
    return None


def _plural(key_singular: str, key_plural: str, cfg: Dict[str, Any]) -> Optional[List[Any]]:
    """Read either plural or singular variant from a stage config."""
    if key_plural in cfg:
        v = cfg[key_plural]
        return v if isinstance(v, list) else [v]
    if key_singular in cfg:
        v = cfg[key_singular]
        return v if isinstance(v, list) else [v]
    return None


def _read_stage_decision(hyp_id: str, stage_name: str) -> Optional[Dict[str, Any]]:
    """Read research-tick-results/verification/<Hxx>/<stage>/DECISION.md frontmatter.

    Expected shape:
        ---
        result: pass | fail | pending
        figure: <relative png>
        note: <one-liner>
        ---
    Returns the parsed frontmatter dict, or None if the file is absent.
    """
    p = VERIFICATION_DIR / hyp_id / stage_name / "DECISION.md"
    if not p.exists():
        return None
    try:
        fm, _ = _parse_frontmatter(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return fm or {}


def _next_stage(stage_name: str) -> Optional[str]:
    try:
        i = LADDER_STAGES.index(stage_name)
    except ValueError:
        return None
    if i + 1 >= len(LADDER_STAGES):
        return None
    return LADDER_STAGES[i + 1]


def _render_probe_for_stage(*, project_name: str, cfg: Dict[str, Any],
                            probe_steps: int) -> tuple[str, str, List[str]]:
    """Build a launch_ablation.sh command from a verification_stages entry.

    Returns (dry_cmd, run_cmd, warnings). Warnings note config elements the
    launcher can't express yet (e.g. multi-seed, multi-loss on the same run),
    so the report can flag them to the user.
    """
    models = _plural("model", "models", cfg) or ["mlp"]
    opts = _plural("optimizer", "optimizers", cfg) or ["fullgd"]
    losses = _plural("loss", "losses", cfg) or ["mse"]
    seeds = cfg.get("seeds") or [7777]

    warnings: List[str] = []
    if len(seeds) > 1:
        warnings.append(
            f"Stage requests {len(seeds)} seeds ({seeds}) but launch_ablation.sh has no "
            f"--seeds flag. Probe will run with the launcher's default seed; add --seeds "
            f"support to the launcher before completing this stage."
        )
    if len(losses) > 1:
        warnings.append(
            f"Stage requests multiple losses {losses}; launch_ablation.sh accepts a single "
            f"--loss. Probe will run with loss={losses[0]}; rerun separately for each."
        )

    parts = [
        "bash eoss_training_scripts/launch_ablation.sh --custom",
        f'--models "{" ".join(str(m) for m in models)}"',
        f'--optimizers "{" ".join(str(o) for o in opts)}"',
        '--lrs "0.05"',
        '--batches "128"',
        f'--loss "{losses[0]}"',
        '--lmax-schedule "none"',
        '--input-prototypes-modes "val"',
        '--input-prototype-sources "generate"',
        '--input-boundary-counts "10"',
        '--input-inliers-counts "10"',
        '--input-x-outlier-counts "5"',
        '--input-y-outlier-counts "5"',
        f'--steps {probe_steps}',
        f'--project-name "{project_name}"',
    ]
    dry, run = _dry_and_run(parts)
    return dry, run, warnings


def propose_H_primary(hyp: Hypothesis, project_base: str) -> Proposal:
    """Stage-aware proposer for the primary EoS-selective-tradeoff claim.

    Reads `stage:` from the hypothesis frontmatter and renders a probe for
    that stage's config (per research-tick-results/budget.yaml:verification_stages).
    If a DECISION.md for the current stage already says `result: pass`, proposes
    the *next* stage's probe instead and notes that the user should bump `stage:`.
    """
    budget = load_budget()
    current_stage = str(hyp.frontmatter.get("stage") or "seed_1").strip()
    probe_steps = int((budget.get("jobs") or {}).get("probe_steps", 500))

    decision = _read_stage_decision(hyp.id, current_stage)
    advance_note = ""
    stage_to_run = current_stage
    if decision and str(decision.get("result", "")).lower() == "pass":
        nxt = _next_stage(current_stage)
        if nxt:
            stage_to_run = nxt
            advance_note = (
                f"Stage `{current_stage}` is marked pass in "
                f"`verification/{hyp.id}/{current_stage}/DECISION.md` — bumping probe to `{stage_to_run}`. "
                f"Also update `stage:` in H_primary frontmatter."
            )
        else:
            return Proposal(
                hypothesis_id=hyp.id,
                kind="skip",
                project_name="",
                summary=(
                    f"Ladder complete — all stages {LADDER_STAGES} passed. "
                    f"H_primary ready for Phase E (draft.tex writer)."
                ),
            )

    cfg = _stage_config(budget, stage_to_run)
    if cfg is None:
        return Proposal(
            hypothesis_id=hyp.id,
            kind="skip",
            project_name="",
            summary=(
                f"No entry for stage `{stage_to_run}` in budget.yaml:verification_stages. "
                f"Add one before the tick can propose a probe."
            ),
        )

    # Budget fence: queue must be empty.
    qs = queue_guard()
    fence_note = ""
    must_be_empty = bool((budget.get("queue") or {}).get("must_be_empty", True))
    if must_be_empty and not qs.empty:
        fence_note = (
            f"\n\n**Budget fence:** queue is NOT empty ({len(qs.running)} running, "
            f"{len(qs.pending)} pending). Per `budget.yaml:queue.must_be_empty`, this "
            f"probe is proposed but not auto-submittable until the queue drains."
        )

    project_name = f"{project_base}-Hprimary-{stage_to_run}"
    dry, run, warnings = _render_probe_for_stage(
        project_name=project_name, cfg=cfg.get("config") or {}, probe_steps=probe_steps,
    )
    gate = cfg.get("gate", "visual")
    score = cfg.get("score")

    summary_lines = [
        f"H_primary stage `{stage_to_run}` probe ({probe_steps} steps) — gate: {gate}.",
    ]
    if score:
        summary_lines.append(f"Auto-score criterion: `{score}`.")
    if advance_note:
        summary_lines.append(advance_note)
    for w in warnings:
        summary_lines.append(f"⚠ {w}")
    summary_lines.append(
        "On completion, digest the resulting W&B project, pick the key per-group plot, "
        f"and save it to `verification/{hyp.id}/{stage_to_run}/key_figure.png` plus "
        f"a `DECISION.md` with `result: pass|fail`."
    )
    summary = " ".join(summary_lines) + fence_note

    return Proposal(
        hypothesis_id=hyp.id,
        kind="sweep",
        project_name=project_name,
        summary=summary,
        dry_run_cmd=dry,
        run_cmd=run,
        signals_to_watch=[
            "group.<g>.eos_crossing.lambda_max.step",
            "group.<g>.loss_decline_onset.full_loss.step",
            "group.<g>.cos_crossing.grad_vmax_cos2.step",
            "group.<g>.stability_ratio.mean",
            "group_separation.full_loss.ratio",
        ],
    )


def propose_fork(*, hypothesis_id: str, parent_run_id: str, cont_step: int,
                 lr_baseline: float, lr_exit_ratio: float = 0.1,
                 project_name: str) -> Proposal:
    """Propose a fork-intervention triplet at t*.

    Emits an infra proposal because launch_ablation.sh / train_eoss.py don't yet
    support `--cont-run-id` / `--cont-step` / `--lr-drop-at-step`. The proposal
    carries the concrete patch sketch and the target run parameters so the
    user can authorize the infra work separately.
    """
    lr_exit = lr_baseline * lr_exit_ratio
    return Proposal(
        hypothesis_id=hypothesis_id,
        kind="infra",
        project_name=project_name,
        summary=(
            f"Fork proposal: resume W&B run `{parent_run_id}` at step {cont_step} "
            f"(first global 2/η crossing), launch two siblings — baseline (lr={lr_baseline}) "
            f"and exit (lr={lr_exit}) — and compare per-group loss/curvature divergence "
            f"between forks. Requires launcher + training support for checkpoint resume "
            f"and mid-run lr drop."
        ),
        infra_note=(
            "Training-side: add `--cont-run-id` (W&B run id) and `--cont-step` (int) to\n"
            "`edge-of-stochastic-stability-and-memorization/training.py`; on load, restore\n"
            "model + optimizer state from the parent checkpoint at that step.\n"
            "Add `--lr-drop-at-step STEP:RATIO` (e.g. `T:0.1` → at step T drop lr ×0.1).\n\n"
            "Launcher-side: expose matching flags in `launch_ablation.sh` and thread them\n"
            "into the `sbatch` exports. Once wired, this proposer converts to `kind: sweep`\n"
            "emitting three sbatch jobs per fork point (shared trajectory + baseline + exit)."
        ),
        signals_to_watch=[
            "group.<g>.loss_decline_onset.full_loss.step  (per fork)",
            "group.<g>.stability_ratio.mean  (baseline vs exit)",
            "group_separation.full_loss.ratio  (diverges post-fork?)",
        ],
    )


PROPOSERS: Dict[str, Callable[[Hypothesis, str], Proposal]] = {
    "H_primary-eos-selective-tradeoff": propose_H_primary,
    "H01-lambda-learning-covariance": propose_H01,
    "H02-stability-ratio-group-separator": propose_H02,
    "H03-cosine-precedes-loss-drop": propose_H03,
    "H04-hparams-expose-regimes": propose_H04,
}


# ---------- decision-log append ---------- #

def _append_decision_log(hyp: Hypothesis, line: str) -> None:
    text = hyp.path.read_text(encoding="utf-8")
    # Insert under "## Decision log" section; keep existing content.
    if "## Decision log" not in text:
        text = text.rstrip() + "\n\n## Decision log\n"
    # Append line at end of file (keeps chronological order).
    if not text.endswith("\n"):
        text += "\n"
    text += line + "\n"
    hyp.path.write_text(text, encoding="utf-8")
    # Bump `updated` field in frontmatter.
    today = datetime.now().strftime("%Y-%m-%d")
    text = hyp.path.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^updated:\s*.*$", f"updated: {today}", text, count=1)
    hyp.path.write_text(text, encoding="utf-8")


# ---------- report renderer ---------- #

def _render_report(ts: str, projects: List[str], digest_dirs: Dict[str, Optional[Path]],
                   proposals: List[Proposal], report_dir: Path) -> Path:
    lines: List[str] = []
    lines.append(f"# Research tick — {ts}")
    lines.append("")
    lines.append(f"Projects digested: {', '.join(projects) if projects else '(none)'}")
    lines.append("")
    lines.append("## Digests")
    for proj in projects:
        d = digest_dirs.get(proj)
        if d is None:
            lines.append(f"- **{proj}**: digest FAILED (see stderr in tick log).")
            continue
        rel = d.relative_to(report_dir.parent.parent) if report_dir.parent.parent in d.parents else d
        digest_json = d / "digest.json"
        runs_in = None
        if digest_json.exists():
            try:
                data = json.loads(digest_json.read_text())
                runs_in = len(data.get("runs", []))
            except Exception:
                pass
        suffix = f" — {runs_in} runs" if runs_in is not None else ""
        lines.append(f"- **{proj}** → `{rel}`{suffix}")
    lines.append("")

    lines.append("## Proposals")
    lines.append("")
    for prop in proposals:
        lines.append(f"### {prop.hypothesis_id}")
        lines.append("")
        lines.append(prop.summary)
        lines.append("")
        if prop.signals_to_watch:
            lines.append("**Signals to watch**")
            for s in prop.signals_to_watch:
                lines.append(f"- `{s}`")
            lines.append("")
        if prop.kind == "sweep":
            lines.append(f"Proposed W&B project: `{prop.project_name}`")
            lines.append("")
            lines.append("**Dry-run (safe to inspect):**")
            lines.append("```bash")
            lines.append(prop.dry_run_cmd or "")
            lines.append("```")
            lines.append("")
            lines.append('**To authorize this sweep, reply `run` — the assistant will execute:**')
            lines.append("```bash")
            lines.append(prop.run_cmd or "")
            lines.append("```")
            lines.append("")
        elif prop.kind == "infra":
            lines.append("**Infra-blocked.** Proposed change:")
            lines.append("")
            lines.append(prop.infra_note or "")
            lines.append("")
        else:
            lines.append("_(no action this tick)_")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("Tick report written non-interactively. The `--run` forms above are NOT executed.")
    report_path = report_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Also write a machine-readable proposals.json for the driver to consume.
    proposals_path = report_dir / "proposals.json"
    proposals_path.write_text(json.dumps(
        [p.__dict__ for p in proposals],
        indent=2,
        default=str,
    ), encoding="utf-8")
    return report_path


# ---------- main ---------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one /research-tick iteration.")
    p.add_argument("--projects", nargs="+", required=True,
                   help="W&B project names to digest (e.g. resnet-0.05).")
    p.add_argument("--samples", type=int, default=5000, help="Samples passed to sweep_digest.py.")
    p.add_argument("--project-base", default="eoss",
                   help="Prefix used when generating new W&B project names for proposed sweeps.")
    p.add_argument("--out-dir", type=Path, default=None, help="Override report output directory.")
    p.add_argument("--skip-digest", action="store_true",
                   help="Skip running sweep_digest.py (use most recent digest under digests/<project>/ if any).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = args.out_dir or (REPORTS_DIR / ts)
    report_dir.mkdir(parents=True, exist_ok=True)
    print(f"[tick] report dir: {report_dir}")

    digest_dirs: Dict[str, Optional[Path]] = {}
    for proj in args.projects:
        if args.skip_digest:
            latest = sorted(
                (REPO_ROOT / "digests" / proj).glob("*/"),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            digest_dirs[proj] = latest[0] if latest else None
            if digest_dirs[proj] is None:
                print(f"[tick] WARN: --skip-digest but no existing digest for {proj}")
        else:
            digest_dirs[proj] = _run_digest(proj, report_dir, args.samples)

    hypotheses = [h for h in _load_hypotheses() if h.status in ("open", "ready")]
    print(f"[tick] {len(hypotheses)} active hypotheses (open|ready)")

    proposals: List[Proposal] = []
    for hyp in hypotheses:
        gen = PROPOSERS.get(hyp.id)
        if gen is None:
            proposals.append(Proposal(
                hypothesis_id=hyp.id, kind="skip", project_name="",
                summary=f"No proposal generator registered for {hyp.id}.",
            ))
            continue
        try:
            prop = gen(hyp, args.project_base)
        except Exception as exc:
            prop = Proposal(
                hypothesis_id=hyp.id, kind="skip", project_name="",
                summary=f"generator raised: {exc}",
            )
        proposals.append(prop)

    report_path = _render_report(ts, args.projects, digest_dirs, proposals, report_dir)
    print(f"[tick] report: {report_path}")

    today = datetime.now().strftime("%Y-%m-%d")
    for prop in proposals:
        hyp = next((h for h in hypotheses if h.id == prop.hypothesis_id), None)
        if hyp is None:
            continue
        if prop.kind == "sweep":
            line = f"- {today} — tick `{ts}` — proposed sweep `{prop.project_name}` (awaiting authorization)."
        elif prop.kind == "infra":
            line = f"- {today} — tick `{ts}` — infra-blocked: {prop.summary.splitlines()[0]}"
        else:
            line = f"- {today} — tick `{ts}` — no proposal: {prop.summary.splitlines()[0]}"
        _append_decision_log(hyp, line)

    # Rebuild index so "updated" dates propagate.
    rebuild = HYPOTHESES_DIR / "rebuild_index.py"
    if rebuild.exists():
        subprocess.run([sys.executable, str(rebuild)], cwd=REPO_ROOT, check=False)

    print(f"[tick] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
