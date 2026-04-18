"""eos_signals — numerical signal extraction primitives for EoSS runs.

Each primitive operates on a pandas DataFrame of W&B history (or a pair of
1-D arrays) and emits a small JSON-serializable dict. Primitives contain NO
matplotlib, NO wandb dependencies — the plot script and the CLI call into
them. The loop consumes their output; human reports render the same signals
as matplotlib overlays.

The canonical list of primitives is `eos_signals/registry.yaml`.
"""

from .primitives import (
    crossing_step,
    eos_threshold,
    eos_threshold_crossing,
    group_separation,
    plateau_duration,
    slope_in_window,
    stability_ratio_series,
    stability_ratio_summary,
)

__all__ = [
    "crossing_step",
    "eos_threshold",
    "eos_threshold_crossing",
    "group_separation",
    "plateau_duration",
    "slope_in_window",
    "stability_ratio_series",
    "stability_ratio_summary",
]

SCHEMA_VERSION = 1
