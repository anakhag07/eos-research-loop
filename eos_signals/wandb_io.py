"""Thin wrapper around wandb.Api for history fetching.

Kept tiny on purpose: the primitives in `eos_signals.primitives` work on any
DataFrame. This module exists so the CLI can fetch run history without every
caller having to re-learn the wandb API.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


def _resolve_run_path(run: str, project: Optional[str], entity: Optional[str]) -> str:
    if "/" in run:
        return run
    if project is None:
        raise ValueError("run_path without slash requires --project")
    if entity:
        return f"{entity}/{project}/{run}"
    return f"{project}/{run}"


def fetch_history(
    run: str,
    metrics: Iterable[str],
    *,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    step_col: str = "_step",
    samples: int = 100_000,
    use_scan: bool = False,
) -> pd.DataFrame:
    """Fetch run history as a DataFrame with columns [step_col, *metrics].

    Uses `wandb.Api().run(...).history(...)` by default. Set use_scan=True to
    use `scan_history` for full-resolution dumps (slower, but captures every
    logged step).
    """
    import wandb

    api = wandb.Api()
    run_path = _resolve_run_path(run, project, entity)
    run_obj = api.run(run_path)

    cols = list(dict.fromkeys([step_col, *metrics]))
    if use_scan:
        rows = list(run_obj.scan_history(keys=cols))
        return pd.DataFrame(rows)
    return run_obj.history(samples=samples, keys=cols)


def fetch_run_config(run: str, *, project: Optional[str] = None, entity: Optional[str] = None) -> dict:
    """Return the run's config dict."""
    import wandb

    api = wandb.Api()
    run_obj = api.run(_resolve_run_path(run, project, entity))
    return dict(run_obj.config)
