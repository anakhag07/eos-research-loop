#!/usr/bin/env python3
"""research_tick.py — one iteration of the EoSS research loop.

Steps:
  1. Build a sweep digest for each --project (reuses eoss_training_scripts/sweep_digest.py).
  2. Read every hypothesis in `hypotheses/` with status == open.
  3. For each open hypothesis, call its proposal generator:
       - emits a dry-run launch_ablation.sh command and a paired --run form,
       - or emits an "infra-blocked" note if a required primitive/extension is missing.
  4. Write reports/<ts>/report.md bundling digests + proposals.
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
REPORTS_DIR = REPO_ROOT / "reports"
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
    for p in sorted(HYPOTHESES_DIR.glob("H*.md")):
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
    return Proposal(
        hypothesis_id=hyp.id,
        kind="infra",
        project_name="",
        summary=(
            "Blocked on infra: the digest does not currently extract per-group "
            "`grad_vmax_cos2` crossings nor a forward-window slope onset for `full_loss`. "
            "Both are small additions."
        ),
        infra_note=(
            "1) Add `grad_vmax_cos2` to `GROUP_METRIC_SUFFIXES` in sweep_digest.py and "
            "emit `group.<g>.cos_crossing.grad_vmax_cos2.step` via `crossing_step` with "
            "a threshold of 0.5 (tune later).\n"
            "2) Add a new primitive to `eos_signals.primitives`: "
            "`first_window_with_negative_slope(df, step_col, metric_col, window, slope_max)` "
            "→ returns earliest window where slope <= slope_max. Register in "
            "`registry.yaml` (schema_version bump). Apply to `group.<g>.full_loss`."
        ),
        signals_to_watch=[
            "group.<g>.cos_crossing.grad_vmax_cos2.step  (needs infra)",
            "group.<g>.loss_decline_onset.full_loss.step  (needs new primitive)",
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


PROPOSERS: Dict[str, Callable[[Hypothesis, str], Proposal]] = {
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

    hypotheses = [h for h in _load_hypotheses() if h.status == "open"]
    print(f"[tick] {len(hypotheses)} open hypotheses")

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
