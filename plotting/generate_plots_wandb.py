"""
generate_plots_wandb.py

CLI equivalent of wandb_plots.ipynb. Keeps the same config variables but lets you
run from a terminal: fetch runs from Weights & Biases, produce the three plot
variants, optionally save them, and optionally suppress interactive windows.
"""

import argparse
import math
import os
from datetime import datetime
from pathlib import Path

import matplotlib

if not os.environ.get("MPLBACKEND") and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, TextBox
import matplotlib.widgets as mwidgets
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm

from eos_signals.primitives import eos_threshold as _eos_threshold_lib
from eos_signals.primitives import crossing_step as _crossing_step_lib


class RunFetchError(RuntimeError):
    """Raised when a W&B run cannot be fetched."""


class MetricFetchError(RuntimeError):
    """Raised when a metric (or its step column) cannot be fetched from a run."""


# ---------------------------- User Config ---------------------------- #
# Edit these values directly or override selected ones with CLI flags.

team_name = "edge-of-stability"

# Runs:
# - (label, run_id)
# - (label, run_id, color)
# - (label, run_id, project_name)  # same team, override project
# - (label, run_id, color, project_name)
# - (label, run_id, color, team_name, project_name) for per-run overrides.
# - (label, full_run_path[, color])
# - color is optional and can be any Matplotlib-recognized name:
#   e.g., tab:blue, tab:red, C0..C9, blue, xkcd:bright sky blue.
# If the run ref does not include '/', it defaults to team_name/project_name.
# First run is plotted in the single-run cell; all runs are used in the multi-run plots.

# Adam (Batch 128) MLP CE Cifar-10
# - Note with these: set max step size, many with outlier runs trained for a long time (2 mil. step)
# - Limit step size to 500k
project_name = "resnet-0.05"
run_specs = [
   ("Baseline, η = 0.05", "z7588exm"),
   ("Drop to η = 0.045", "qtkpwk8v"), 
   ("Drop to η = 0.025", "2psr65ik"),
   ("Drop to η = 0.005", "79o0oc8t"), 
]

# # Adam (Batch 128) MLP MSE Cifar-10
# project_name = "eoss-training-with-and-without-prototypes"
# run_specs = [
#    ("mse_adam_lr_0.0003_with_prototypes", "bbmbac7i"),
#    ("mse_adam_lr_0.0003_no_prototypes", "g4n50uha"), 
#    ("mse_adam_lr_0.0002_with_prototypes", "1nv0zzdo"),
#    ("mse_adam_lr_0.0002_no_prototypes", "0642yrr2"), 
#    ("mse_adam_lr_0.0001_with_prototypes", "t5srorff"),
#    ("mse_adam_lr_0.0001_no_prototypes", "rojsz84m"), 
# ]

# eta symbol: η

# Metrics and plotting options
step_col = "_step"
single_metric = "full_loss"
metrics = [
    "full_loss",
    # "lambda_max_precond_adam",
    "lambda_max",
    # "batch_sharpn",
    # "batch_loss",
    # # "input_space_prototypes/injected_y_outlier/injected_y_outlier/batch_sharpness",
    # "input_space_prototypes/injected_y_outlier/injected_y_outlier/full_loss",
    # "input_space_prototypes/injected_y_outlier/injected_y_outlier/lambda_max",
    # "input_space_prototypes/injected_y_outlier/injected_y_outlier/grad_hessian_grad",
    # "input_space_prototypes/injected_y_outlier/injected_y_outlier/grad_vmax_cos2",
    # "input_space_prototypes/injected_y_outlier/injected_y_outlier/grad_norm",
    # # "input_space_prototypes/injected_x_outlier/injected_x_outlier/batch_sharpness",
    # "input_space_prototypes/injected_x_outlier/injected_x_outlier/full_loss",
    # "input_space_prototypes/injected_x_outlier/injected_x_outlier/lambda_max",
    # "input_space_prototypes/injected_x_outlier/injected_x_outlier/grad_hessian_grad",
    # "input_space_prototypes/injected_x_outlier/injected_x_outlier/grad_vmax_cos2",
    # "input_space_prototypes/synthetic_x_outlier/synthetic_x_outlier/grad_norm",
    # # "input_space_prototypes/injected_boundary/injected_boundary/batch_sharpness",
    # "input_space_prototypes/injected_boundary/injected_boundary/full_loss",
    # "input_space_prototypes/injected_boundary/injected_boundary/lambda_max",
    # "input_space_prototypes/injected_boundary/injected_boundary/grad_hessian_grad",
    # "input_space_prototypes/injected_boundary/injected_boundary/grad_vmax_cos2",
    # "input_space_prototypes/synthetic_x_outlier/synthetic_x_outlier/grad_norm",
    # # "input_space_prototypes/injected_inliers/injected_inliers/batch_sharpness",
    # "input_space_prototypes/injected_inliers/injected_inliers/full_loss",
    # "input_space_prototypes/injected_inliers/injected_inliers/lambda_max",
    # "input_space_prototypes/injected_inliers/injected_inliers/grad_hessian_grad",
    # "input_space_prototypes/injected_inliers/injected_inliers/grad_vmax_cos2",
    # "input_space_prototypes/synthetic_x_outlier/synthetic_x_outlier/grad_norm",
    # # "input_space_prototypes/boundary/boundary/batch_sharpness",
    # "input_space_prototypes/boundary_points/boundary/full_loss",
    # "input_space_prototypes/boundary_points/boundary/lambda_max",
    # "input_space_prototypes/boundary_points/boundary/grad_hessian_grad",
    # "input_space_prototypes/boundary_points/boundary/grad_vmax_cos2",
    # "input_space_prototypes/boundary_points/boundary/grad_norm",
    # # "input_space_prototypes/inlier_points/inliers/batch_sharpness",
    # "input_space_prototypes/inlier_points/inliers/full_loss",
    # "input_space_prototypes/inlier_points/inliers/lambda_max",
    # "input_space_prototypes/inlier_points/inliers/grad_hessian_grad",
    # "input_space_prototypes/inlier_points/inliers/stability_ratio",
]
history_samples = 50000

log_x = False
log_y = False

# Limit how far into training to fetch (by step). Set to None to disable.
max_step_fetch = 100000

# Optional output directory for saving figures.
# Defaults to a project-local path so headless loops don't depend on an
# author-specific absolute path. Override with --save-dir.
save_dir = str(Path(__file__).resolve().parent / "plots" / project_name)

# Figure layout
grid_cols = 4

# Metric group plotting (row-wise grouped metrics)
stability_ratio_metrics = {
    "input_space_prototypes/inlier_points/inliers/stability_ratio":
        "input_space_prototypes/inlier_points/inliers/grad_hessian_grad",
    "input_space_prototypes/boundary_points/boundary/stability_ratio":
        "input_space_prototypes/boundary_points/boundary/grad_hessian_grad",
    "input_space_prototypes/injected_inliers/injected_inliers/stability_ratio":
        "input_space_prototypes/injected_inliers/injected_inliers/grad_hessian_grad",
    "input_space_prototypes/injected_boundary/injected_boundary/stability_ratio":
        "input_space_prototypes/injected_boundary/injected_boundary/grad_hessian_grad",
    "input_space_prototypes/injected_x_outlier/injected_x_outlier/stability_ratio":
        "input_space_prototypes/injected_x_outlier/injected_x_outlier/grad_hessian_grad",
    "input_space_prototypes/injected_y_outlier/injected_y_outlier/stability_ratio":
        "input_space_prototypes/injected_y_outlier/injected_y_outlier/grad_hessian_grad",
}
stability_ratio_smooth_window = 1
stability_ratio_inputs = list(stability_ratio_metrics.values()) + [
    grad_key.replace("/grad_hessian_grad", "/lambda_max")
    for grad_key in stability_ratio_metrics.values()
]

metric_groups = [
    {
        "label": "core",
        "metrics": ["full_loss", "lambda_max"],
        "legend_mode": "per-ax",
    },
    {
        "label": "full_loss (input_space_prototypes)",
        "metrics": [
            "input_space_prototypes/inlier_points/inliers/full_loss",
            "input_space_prototypes/boundary_points/boundary/full_loss",
            "input_space_prototypes/injected_inliers/injected_inliers/full_loss",
            "input_space_prototypes/injected_boundary/injected_boundary/full_loss",
            "input_space_prototypes/injected_x_outlier/injected_x_outlier/full_loss",
            "input_space_prototypes/injected_y_outlier/injected_y_outlier/full_loss",
        ],
        "legend_mode": "shared-right",
    },
    {
        "label": "lambda_max (input_space_prototypes)",
        "metrics": [
            "input_space_prototypes/inlier_points/inliers/lambda_max",
            "input_space_prototypes/boundary_points/boundary/lambda_max",
            "input_space_prototypes/injected_inliers/injected_inliers/lambda_max",
            "input_space_prototypes/injected_boundary/injected_boundary/lambda_max",
            "input_space_prototypes/injected_x_outlier/injected_x_outlier/lambda_max",
            "input_space_prototypes/injected_y_outlier/injected_y_outlier/lambda_max",
        ],
        "legend_mode": "shared-right",
    },
    {
        "label": "grad_hessian_grad (input_space_prototypes)",
        "metrics": [
            "input_space_prototypes/inlier_points/inliers/grad_hessian_grad",
            "input_space_prototypes/boundary_points/boundary/grad_hessian_grad",
            "input_space_prototypes/injected_inliers/injected_inliers/grad_hessian_grad",
            "input_space_prototypes/injected_boundary/injected_boundary/grad_hessian_grad",
            "input_space_prototypes/injected_x_outlier/injected_x_outlier/grad_hessian_grad",
            "input_space_prototypes/injected_y_outlier/injected_y_outlier/grad_hessian_grad",
        ],
        "legend_mode": "shared-right",
    },
    {
        "label": "grad_vmax_cos2 (input_space_prototypes)",
        "metrics": [
            "input_space_prototypes/inlier_points/inliers/grad_vmax_cos2",
            "input_space_prototypes/boundary_points/boundary/grad_vmax_cos2",
            "input_space_prototypes/injected_inliers/injected_inliers/grad_vmax_cos2",
            "input_space_prototypes/injected_boundary/injected_boundary/grad_vmax_cos2",
            "input_space_prototypes/injected_x_outlier/injected_x_outlier/grad_vmax_cos2",
            "input_space_prototypes/injected_y_outlier/injected_y_outlier/grad_vmax_cos2",
        ],
        "legend_mode": "shared-right",
    },
    {
        "label": "grad_norm (input_space_prototypes)",
        "metrics": [
            "input_space_prototypes/inlier_points/inliers/grad_norm",
            "input_space_prototypes/boundary_points/boundary/grad_norm",
            "input_space_prototypes/injected_inliers/injected_inliers/grad_norm",
            "input_space_prototypes/injected_boundary/injected_boundary/grad_norm",
            "input_space_prototypes/injected_x_outlier/injected_x_outlier/grad_norm",
            "input_space_prototypes/injected_y_outlier/injected_y_outlier/grad_norm",
        ],
        "legend_mode": "shared-right",
    },
    {
        "label": "stability_ratio = (gᵀHg / ||g||²) / λmax",
        "metrics": list(stability_ratio_metrics.keys()),
        "legend_mode": "shared-right",
    },
]

# Grouped figure layout: approximate square subplots
group_subplot_size = 3.4  # inches per subplot (width and height)
group_subplot_size_rect = (6.5, 4)  # match plot_metrics_separately sizing
group_left_margin = 0.08
group_ylabel_pad = 16
group_wspace = 0.13
group_plot_area_right = 0.88
group_legend_mode = "per-ax"  # per-ax | shared-right | shared-bottom
group_apply_scales = True  # apply log/min/max scaling when True

# Axis scaling
normalize_by_distance = True  # When True, scale x-axis by (learning rate * step)
interv_lr = False  # When True, use per-step lr history for distance
lr_key = "lr"  # Config key to read the learning rate from
max_x = None  # Set to a number to cap the max x-axis (after any scaling)
min_x = 0  # Set to a number to cap the min x-axis (after any scaling)
max_y = None  # Optional cap for y-axis
min_y = None  # Optional lower bound for y-axis
legend_loc = "best"  # Default legend location

# Optional friendly display names for metrics
metric_titles = {
    "full_loss" : "Full Loss",
    "lambda_max_precond_adam": "λmax (Precond Adam)",
    "lambda_max": "λmax",
    "batch_sharpn": "Batch Sharpness",
    "batch_loss": "Batch Loss",
    "input_space_prototypes/inlier_points/inliers/batch_sharpness": "Batch Sharpness (Inliers)",
    "input_space_prototypes/inlier_points/inliers/full_loss": "Full Loss (Inliers)",
    "input_space_prototypes/inlier_points/inliers/lambda_max": "λmax (Inliers)",
    "input_space_prototypes/inlier_points/inliers/grad_hessian_grad": "gᵀHg / ||g||² (Inliers)",
    "input_space_prototypes/inlier_points/inliers/grad_vmax_cos2": "cos²(g, v_max) (Inliers)",
    "input_space_prototypes/inlier_points/inliers/grad_norm": "Gradient Norm (Inliers)",
    "input_space_prototypes/inlier_points/inliers/stability_ratio": "(gᵀHg / ||g||²) / λmax (Inliers)",
    "input_space_prototypes/boundary_points/boundary/batch_sharpness": "Batch Sharpness (Boundary)",
    "input_space_prototypes/boundary_points/boundary/full_loss": "Full Loss (Boundary)",
    "input_space_prototypes/boundary_points/boundary/lambda_max": "λmax (Boundary)",
    "input_space_prototypes/boundary_points/boundary/grad_hessian_grad": "gᵀHg / ||g||² (Boundary)",
    "input_space_prototypes/boundary_points/boundary/grad_vmax_cos2": "cos²(g, v_max) (Boundary)",
    "input_space_prototypes/boundary_points/boundary/grad_norm": "Gradient Norm (Boundary)",
    "input_space_prototypes/boundary_points/boundary/stability_ratio": "(gᵀHg / ||g||²) / λmax (Boundary)",
    "input_space_prototypes/injected_inliers/injected_inliers/batch_sharpness": "Batch Sharpness (Injected Inliers)",
    "input_space_prototypes/injected_inliers/injected_inliers/full_loss": "Full Loss (Injected Inliers)",
    "input_space_prototypes/injected_inliers/injected_inliers/lambda_max": "λmax (Injected Inliers)",
    "input_space_prototypes/injected_inliers/injected_inliers/grad_hessian_grad": "gᵀHg / ||g||² (Injected Inliers)",
    "input_space_prototypes/injected_inliers/injected_inliers/grad_vmax_cos2": "cos²(g, v_max) (Injected Inliers)",
    "input_space_prototypes/injected_inliers/injected_inliers/grad_norm": "Gradient Norm (Injected Inliers)",
    "input_space_prototypes/injected_inliers/injected_inliers/stability_ratio": "(gᵀHg / ||g||²) / λmax (Injected Inliers)",
    "input_space_prototypes/injected_boundary/injected_boundary/batch_sharpness": "Batch Sharpness (Injected Boundary)",
    "input_space_prototypes/injected_boundary/injected_boundary/full_loss": "Full Loss (Injected Boundary)",
    "input_space_prototypes/injected_boundary/injected_boundary/lambda_max": "λmax (Injected Boundary)",
    "input_space_prototypes/injected_boundary/injected_boundary/grad_hessian_grad": "gᵀHg / ||g||² (Injected Boundary)",
    "input_space_prototypes/injected_boundary/injected_boundary/grad_vmax_cos2": "cos²(g, v_max) (Injected Boundary)",
    "input_space_prototypes/injected_boundary/injected_boundary/grad_norm": "Gradient Norm (Injected Boundary)",
    "input_space_prototypes/injected_boundary/injected_boundary/stability_ratio": "(gᵀHg / ||g||²) / λmax (Injected Boundary)",
    "input_space_prototypes/injected_x_outlier/injected_x_outlier/batch_sharpness": "Batch Sharpness (Injected X Outlier)",
    "input_space_prototypes/injected_x_outlier/injected_x_outlier/full_loss": "Full Loss (Injected X Outlier)",
    "input_space_prototypes/injected_x_outlier/injected_x_outlier/lambda_max": "λmax (Injected X Outlier)",
    "input_space_prototypes/injected_x_outlier/injected_x_outlier/grad_hessian_grad": "gᵀHg / ||g||² (Injected X Outlier)",
    "input_space_prototypes/injected_x_outlier/injected_x_outlier/grad_vmax_cos2": "cos²(g, v_max) (Injected X Outlier)",
    "input_space_prototypes/injected_x_outlier/injected_x_outlier/grad_norm": "Gradient Norm (Injected X Outlier)",
    "input_space_prototypes/injected_x_outlier/injected_x_outlier/stability_ratio": "(gᵀHg / ||g||²) / λmax (Injected X Outlier)",
    "input_space_prototypes/injected_y_outlier/injected_y_outlier/batch_sharpness": "Batch Sharpness (Injected Y Outlier)",
    "input_space_prototypes/injected_y_outlier/injected_y_outlier/full_loss": "Full Loss (Injected Y Outlier)",
    "input_space_prototypes/injected_y_outlier/injected_y_outlier/lambda_max": "λmax (Injected Y Outlier)",
    "input_space_prototypes/injected_y_outlier/injected_y_outlier/grad_hessian_grad": "gᵀHg / ||g||² (Injected Y Outlier)",
    "input_space_prototypes/injected_y_outlier/injected_y_outlier/grad_vmax_cos2": "cos²(g, v_max) (Injected Y Outlier)",
    "input_space_prototypes/injected_y_outlier/injected_y_outlier/grad_norm": "Gradient Norm (Injected Y Outlier)",
    "input_space_prototypes/injected_y_outlier/injected_y_outlier/stability_ratio": "(gᵀHg / ||g||²) / λmax (Injected Y Outlier)",
}

# Fetch options
fetch_method = "history"  # or "scan_history"
page_size = 500  # used only for scan_history
max_rows = history_samples  # optional cap on rows when fetching (used as samples/page cap)

# Edge-of-stability overlay options
enable_eos = False
eos_eta_override = None  # if set, use this eta instead of run.config
eos_eta_key = lr_key     # config key to read eta when override is not provided
eos_adam_override = None  # None => infer from optimizer name; True/False to force
eos_show_vertical = True
eos_show_threshold_legend = False
# ---------------------------- End of User Config ---------------------------- #


# ---------------------------- Helpers ---------------------------- #

def to_run_path(team: str, project: str, run_id_or_path: str) -> str:
    if "/" in run_id_or_path:
        return run_id_or_path
    return f"{team}/{project}/{run_id_or_path}"

run_cache = {}


def _run_label(run) -> str:
    """Best-effort identifier for a run for error messages."""
    return getattr(run, "path", None) or getattr(run, "name", None) or getattr(run, "id", None) or "<unknown run>"

def fetch_run_or_raise(api, run_path: str):
    """Fetch a W&B run with clearer error semantics."""
    try:
        return api.run(run_path)
    except Exception as e:
        raise RunFetchError(f"Could not fetch run '{run_path}': {e}") from e


def fetch_run_history(
    run,
    keys,
    step_candidates=("step", "_step", "global_step"),
    fetch_method="history",
    page_size=2000,
    max_rows=None,
    max_step_fetch=None,
    verbose=False,
):
    """
    Fetch run history using either history() or scan_history(), detect step column, and cache results.
    Returns (df, step_col).
    """
    # Build cache key
    run_id = getattr(run, "id", None) or getattr(run, "path", None) or getattr(run, "name", None)
    cache_key = (
        run_id,
        tuple(sorted(keys)) if keys else None,
        tuple(step_candidates),
        fetch_method,
        page_size,
        max_rows,
        max_step_fetch,
    )
    if cache_key in run_cache:
        return run_cache[cache_key]

    df = pd.DataFrame()
    used_method = fetch_method
    retried_without_keys = False

    effective_max_rows = max_rows
    if max_step_fetch is not None:
        if effective_max_rows is None:
            effective_max_rows = max_step_fetch
        else:
            effective_max_rows = min(effective_max_rows, max_step_fetch)

    if fetch_method == "history":
        try:
            try:
                hist = run.history(keys=keys, samples=effective_max_rows)
            except TypeError:
                hist = run.history(samples=effective_max_rows)
        except Exception as e:
            raise MetricFetchError(f"Failed to fetch history for run '{_run_label(run)}': {e}") from e
        df = pd.DataFrame(hist)
    elif fetch_method == "scan_history":
        rows = []
        try:
            for row in run.scan_history(keys=keys, page_size=page_size):
                if max_step_fetch is not None:
                    stop_for_step = False
                    for cand in step_candidates:
                        if cand in row:
                            try:
                                step_val = float(row[cand])
                            except (TypeError, ValueError):
                                step_val = None
                            if step_val is not None and step_val > max_step_fetch:
                                stop_for_step = True
                            break
                    if stop_for_step:
                        break
                rows.append(row)
                if effective_max_rows and len(rows) >= effective_max_rows:
                    break
        except Exception as e:
            raise MetricFetchError(f"Failed to scan history for run '{_run_label(run)}': {e}") from e
        df = pd.DataFrame(rows)
    else:
        raise ValueError(f"Unknown fetch_method '{fetch_method}'")

    # If key filtering returned nothing (common when requested metrics are absent), retry without the keys filter.
    if df.empty and keys:
        retried_without_keys = True
        if fetch_method == "history":
            try:
                try:
                    hist = run.history(samples=effective_max_rows)
                except TypeError:
                    hist = run.history()
            except Exception as e:
                raise MetricFetchError(
                    f"Failed to refetch history without key filter for run '{_run_label(run)}': {e}"
                ) from e
            df = pd.DataFrame(hist)
        elif fetch_method == "scan_history":
            rows = []
            try:
                for row in run.scan_history(page_size=page_size):
                    if max_step_fetch is not None:
                        stop_for_step = False
                        for cand in step_candidates:
                            if cand in row:
                                try:
                                    step_val = float(row[cand])
                                except (TypeError, ValueError):
                                    step_val = None
                                if step_val is not None and step_val > max_step_fetch:
                                    stop_for_step = True
                                break
                        if stop_for_step:
                            break
                    rows.append(row)
                    if effective_max_rows and len(rows) >= effective_max_rows:
                        break
            except Exception as e:
                raise MetricFetchError(
                    f"Failed to refetch scan_history without key filter for run '{_run_label(run)}': {e}"
                ) from e
            df = pd.DataFrame(rows)
        used_method = f"{fetch_method} (no keys)"

    if df.empty:
        raise MetricFetchError(
            f"No history rows returned for run '{_run_label(run)}' using method '{used_method}'."
        )

    # Detect step column
    step_col = None
    for cand in step_candidates:
        if cand in df.columns:
            step_col = cand
            break
    if step_col is None:
        raise MetricFetchError(
            f"Run '{_run_label(run)}' missing step column; tried {step_candidates}, columns: {list(df.columns)}"
        )

    if max_step_fetch is not None:
        step_series = pd.to_numeric(df[step_col], errors="coerce")
        df = df[step_series <= max_step_fetch].copy()
        if df.empty:
            raise MetricFetchError(
                f"Run '{_run_label(run)}' has no rows at or below step {max_step_fetch}."
            )

    # Keep requested keys that exist
    present_keys = [k for k in (keys or []) if k in df.columns and k != step_col]
    missing = [k for k in (keys or []) if k not in df.columns and k != step_col]
    df = df[[step_col] + present_keys].copy()

    if verbose:
        print(
            f"[fetch_run_history] method={used_method} page_size={page_size if fetch_method=='scan_history' else 'n/a'} "
            f"rows={len(df)} step_col={step_col}"
        )
        if missing:
            print(f"[fetch_run_history] missing keys: {missing}")
        if retried_without_keys:
            print("[fetch_run_history] initial fetch empty with keys; refetched without keys.")

    run_cache[cache_key] = (df, step_col)
    return df, step_col

def clean_metric_df(df: pd.DataFrame, step_key: str, metric: str) -> pd.DataFrame:
    if step_key not in df.columns or metric not in df.columns:
        return pd.DataFrame()
    plot_df = df[[step_key, metric]].copy()
    plot_df[step_key] = pd.to_numeric(plot_df[step_key], errors="coerce")
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    return plot_df.dropna(subset=[step_key, metric]).sort_values(step_key)


def compute_stability_ratio_df(
    df: pd.DataFrame,
    step_key: str,
    run,
    metric_name: str,
    run_label: str,
    smooth_window: int = 1,
) -> pd.DataFrame:
    if metric_name not in stability_ratio_metrics:
        return pd.DataFrame()
    grad_key = stability_ratio_metrics[metric_name]
    lambda_key = grad_key.replace("/grad_hessian_grad", "/lambda_max")

    missing = [k for k in (grad_key, lambda_key) if k not in df.columns]
    if missing:
        print(
            f"Skipping {run_label}: missing {missing} for stability ratio ({metric_name})."
        )
        return pd.DataFrame()

    plot_df = df[[step_key, grad_key, lambda_key]].copy()
    plot_df[step_key] = pd.to_numeric(plot_df[step_key], errors="coerce")
    plot_df[grad_key] = pd.to_numeric(plot_df[grad_key], errors="coerce")
    plot_df[lambda_key] = pd.to_numeric(plot_df[lambda_key], errors="coerce")
    plot_df = plot_df.dropna(subset=[step_key, grad_key, lambda_key]).sort_values(step_key)
    plot_df = plot_df[plot_df[lambda_key] > 0]
    if plot_df.empty:
        return pd.DataFrame()

    ratio = plot_df[grad_key] / plot_df[lambda_key]
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    if smooth_window and smooth_window > 1:
        ratio = ratio.rolling(window=smooth_window, min_periods=max(1, smooth_window // 2)).mean()

    plot_df[metric_name] = ratio
    return plot_df[[step_key, metric_name]].dropna(subset=[metric_name])


def compute_metric_df(
    metric: str,
    df: pd.DataFrame,
    step_key: str,
    run,
    run_label: str,
    cfg,
) -> pd.DataFrame | None:
    if metric in stability_ratio_metrics:
        return compute_stability_ratio_df(
            df,
            step_key,
            run,
            metric,
            run_label,
            smooth_window=cfg.get("stability_ratio_smooth_window", 1),
        )
    return None

def normalize_color(color):
    """Return a Matplotlib-accepted color string or None if invalid/absent."""
    if color is None:
        return None
    if isinstance(color, str):
        color = color.strip()
        # Normalize colon-separated tokens (e.g., "tab: red" -> "tab:red")
        if ":" in color:
            parts = [p.strip() for p in color.split(":")]
            color = ":".join(parts)
        try:
            mcolors.to_rgba(color)
            return color
        except ValueError:
            print(f"Warning: unknown color '{color}', falling back to default cycle.")
            return None
    return None


def is_global_lambda_metric(metric_name: str) -> bool:
    return metric_name in {"lambda_max", "lambda_max_precond_adam"}


def should_add_eos_for_metric(metric_name: str, cfg) -> bool:
    if cfg.get("enable_eos"):
        return True
    return is_global_lambda_metric(metric_name)


def is_color_value(value) -> bool:
    if value is None or not isinstance(value, str):
        return False
    try:
        mcolors.to_rgba(value.strip())
        return True
    except ValueError:
        return False


def resolve_run_spec(team: str, project: str, run_spec):
    if len(run_spec) == 2:
        label, run_ref = run_spec
        color = None
        override_team = None
        override_project = None
    elif len(run_spec) == 3:
        label, run_ref, third = run_spec
        if is_color_value(third):
            color = third
            override_project = None
        else:
            color = None
            override_project = third
        override_team = None
    elif len(run_spec) == 4:
        label, run_ref, color, override_project = run_spec
        override_team = None
    elif len(run_spec) == 5:
        label, run_ref, color, override_team, override_project = run_spec
    else:
        raise ValueError(
            "run_specs entries must be (label, run_id[, color|project]) or "
            "(label, run_id, color, project_name) or "
            "(label, run_id, color, team_name, project_name)"
        )
    color = normalize_color(color)
    run_team = override_team or team
    run_project = override_project or project
    run_path = to_run_path(run_team, run_project, run_ref)
    return label, run_path, color

def apply_log(ax, log_x_opt: bool, log_y_opt: bool):
    """Apply log scaling per axis."""
    ax.set_xscale("log" if log_x_opt else "linear")
    ax.set_yscale("log" if log_y_opt else "linear")


def infer_adam(run, default=True):
    """Infer whether run used Adam based on optimizer name; default if unknown."""
    try:
        opt = str(run.config.get("optimizer", "") or "").lower()
    except Exception:
        opt = ""
    if "adam" in opt:
        return True
    if opt:
        return False
    return default


def maybe_add_eos(
    ax,
    plot_df,
    x_col,
    run,
    step_col,
    cfg,
    *,
    color=None,
    label_prefix="",
    run_label=None,
    current_metric=None,
    enabled=None,
):
    """Add EOS overlays if enabled; no-op when data/eta unavailable."""
    if enabled is None:
        enabled = cfg.get("enable_eos")
    if not enabled:
        return

    # Per-run vertical line gating: allowlist overrides the global toggle.
    eos_vertical_allowed = cfg.get("eos_show_vertical", True)
    allowlist = cfg.get("eos_vertical_allow")
    if allowlist is not None:
        eos_vertical_allowed = run_label in allowlist

    # Per-run horizontal line gating: allowlist controls threshold line.
    eos_horizontal_allowed = True
    allowlist_h = cfg.get("eos_horizontal_allow")
    if allowlist_h is not None:
        eos_horizontal_allowed = run_label in allowlist_h

    # Pick eta
    eta = cfg.get("eos_eta_override")
    if eta is None:
        try:
            eta = float(run.config.get(cfg.get("eos_eta_key", "lr")))
        except Exception:
            eta = None
    if eta is None or eta <= 0:
        return

    # Pick driver metric series for crossing
    driver = None
    if cfg.get("eos_cross_all_metrics"):
        driver = current_metric
    elif cfg.get("eos_cross_metrics"):
        if current_metric in cfg["eos_cross_metrics"]:
            driver = current_metric
    else:
        driver_candidates = ["lambda_max", "lambda_max_precond_adam", "batch_sharpn", "batch_sharpness"]
        driver = next((m for m in driver_candidates if m in plot_df.columns), None)

    if driver is None or driver not in plot_df.columns:
        return

    y_series = plot_df[driver]
    x_series = plot_df[x_col]

    # Adam/SGD flag
    adam = cfg.get("eos_adam_override")
    if adam is None:
        adam = infer_adam(run, default=True)

    add_eos_overlays(
        ax_hline=ax if eos_horizontal_allowed else None,
        x_hline=x_series if eos_horizontal_allowed else None,
        y_for_crossing=y_series,
        eta=eta,
        adam=adam,
        color=color,
        show_threshold_legend=cfg.get("eos_show_threshold_legend", False),
        label_prefix=label_prefix,
        ax_vline=ax if eos_vertical_allowed else None,
        x_for_vline=x_series if eos_vertical_allowed else None,
        x_vline_transform=None,
    )

def prepare_step_axis(
    df: pd.DataFrame,
    run,
    step_key: str,
    normalize: bool,
    lr_key: str,
    interv_lr: bool = False,
    full_df: pd.DataFrame | None = None,
):
    """Optionally scale the x-axis by learning rate. Returns (df, x_col, x_label)."""
    if not normalize:
        friendly_label = "iteration" if step_key == "_step" else step_key
        return df, step_key, friendly_label

    try:
        lr_val = float(run.config.get(lr_key))
    except Exception:
        lr_val = None

    if interv_lr:
        source_df = full_df if full_df is not None else df
        if lr_key not in source_df.columns:
            run_name = getattr(run, "name", "<unknown run>")
            print(f"interv_lr enabled but '{lr_key}' missing for {run_name}; using constant lr.")
        else:
            lr_df = source_df[[step_key, lr_key]].copy()
            lr_df[step_key] = pd.to_numeric(lr_df[step_key], errors="coerce")
            lr_df[lr_key] = pd.to_numeric(lr_df[lr_key], errors="coerce")
            lr_df = lr_df.dropna(subset=[step_key]).sort_values(step_key)
            lr_df[lr_key] = lr_df[lr_key].ffill()
            lr_df = lr_df.dropna(subset=[lr_key])
            if not lr_df.empty:
                delta_step = lr_df[step_key].diff().fillna(0)
                lr_df["distance"] = (lr_df[lr_key] * delta_step).cumsum()
                df_scaled = df.copy()
                df_scaled = df_scaled.merge(lr_df[[step_key, "distance"]], on=step_key, how="left")
                return df_scaled, "distance", "distance (∑ η_t · Δiteration)"
            run_name = getattr(run, "name", "<unknown run>")
            print(f"interv_lr enabled but no valid '{lr_key}' points for {run_name}; using constant lr.")

    if lr_val is None or lr_val <= 0:
        run_name = getattr(run, "name", "<unknown run>")
        print(f"normalize_by_distance enabled but lr missing/invalid for {run_name}; using raw steps.")
        friendly_label = "iteration" if step_key == "_step" else step_key
        return df, step_key, friendly_label

    df_scaled = df.copy()
    new_col = "distance"
    df_scaled[new_col] = df_scaled[step_key] * lr_val
    return df_scaled, new_col, "distance (η * iteration)"


def eos_threshold(eta: float, *, adam: bool) -> float:
    """Delegates to eos_signals.primitives.eos_threshold so the loop and the
    rendering layer share one definition."""
    return _eos_threshold_lib(eta, adam=adam)


def first_crossing_x(x, y, thresh: float):
    """
    Return x at the first index where y >= thresh. Kept as a thin adaptor over
    eos_signals.primitives.crossing_step so matplotlib annotations reuse the
    same logic the signal library emits in JSON.
    """
    df = pd.DataFrame({"__x": list(x), "__y": list(y)})
    res = _crossing_step_lib(df, "__x", "__y", thresh, direction="up")
    return res["step"]


def add_eos_overlays(
    *,
    ax_hline,
    x_hline,
    y_for_crossing,
    eta: float,
    adam: bool,
    color=None,
    show_threshold_legend: bool = False,
    label_prefix: str = "",
    ax_vline=None,
    x_for_vline=None,
    x_vline_transform=None,
):
    """
    Draw horizontal EOS threshold line and optionally a vertical crossing line.
    Either axis can be disabled by passing None for ax_hline or ax_vline.
    """
    if ax_hline is None and ax_vline is None:
        return

    thresh = eos_threshold(eta, adam=adam)

    if ax_hline is not None and x_hline is not None:
        ax_hline.hlines(
            thresh,
            xmin=float(x_hline.iloc[0] if hasattr(x_hline, "iloc") else x_hline[0]),
            xmax=float(x_hline.iloc[-1] if hasattr(x_hline, "iloc") else x_hline[-1]),
            colors=color,
            linestyles="--",
            linewidth=1.5,
            label=(f"{label_prefix}{'38' if adam else '2'}/η" if show_threshold_legend else "_nolegend_"),
        )

    if ax_vline is not None and x_for_vline is not None:
        x_cross = first_crossing_x(x_for_vline, y_for_crossing, thresh)
        if x_cross is not None:
            x_v = x_vline_transform(x_cross) if x_vline_transform else x_cross
            ax_vline.axvline(
                x_v,
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
                label=(f"{label_prefix}cross @ {'38' if adam else '2'}/η"
                       if show_threshold_legend else "_nolegend_"),
            )


def attach_axis_controls(fig, ax, cfg):
    """
    Add in-window controls for log scaling and axis limits.
    Controls live in a right-hand gutter; they mutate the target ax only.
    """
    fig.subplots_adjust(right=0.78)

    # Checkbox for log toggles
    cax = fig.add_axes([0.80, 0.76, 0.18, 0.12])
    labels = ["log-x", "log-y"]
    checks = CheckButtons(cax, labels, [cfg["log_x"], cfg["log_y"]])

    def on_toggle(_label):
        logx, logy = checks.get_status()
        ax.set_xscale("log" if logx else "linear")
        ax.set_yscale("log" if logy else "linear")
        fig.canvas.draw_idle()

    checks.on_clicked(on_toggle)

    # Text boxes for limits
    box_positions = {
        "min_x": (0.80, 0.62),
        "max_x": (0.80, 0.56),
        "min_y": (0.80, 0.50),
        "max_y": (0.80, 0.44),
    }
    text_boxes = {}
    for key, (x0, y0) in box_positions.items():
        bax = fig.add_axes([x0, y0, 0.18, 0.06])
        init_val = "" if cfg.get(key) is None else str(cfg[key])
        text_boxes[key] = TextBox(bax, f"{key} ", initial=init_val)

    def apply_limits(_):
        def parse(val):
            val = val.strip()
            return None if val == "" else float(val)

        new_min_x = parse(text_boxes["min_x"].text)
        new_max_x = parse(text_boxes["max_x"].text)
        new_min_y = parse(text_boxes["min_y"].text)
        new_max_y = parse(text_boxes["max_y"].text)
        ax.set_xlim(left=new_min_x, right=new_max_x)
        ax.set_ylim(bottom=new_min_y, top=new_max_y)
        fig.canvas.draw_idle()

    for tb in text_boxes.values():
        tb.on_submit(apply_limits)

    # Legend location radio buttons
    lax = fig.add_axes([0.80, 0.30, 0.18, 0.12])
    legend_options = ["best", "upper right", "upper left", "lower left", "lower right"]
    radio = mwidgets.RadioButtons(lax, legend_options, active=legend_options.index(cfg.get("legend_loc", "best")))

    def on_legend_change(label):
        # Recreate legend at new location to avoid duplicates
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc=label)
        fig.canvas.draw_idle()

    radio.on_clicked(on_legend_change)

    # Persist widget refs on the figure to avoid garbage collection (needed for callbacks).
    if not hasattr(fig, "_axis_controls"):
        fig._axis_controls = []
    fig._axis_controls.append({"checks": checks, "text_boxes": text_boxes, "legend_radio": radio})

    # Wrap the canvas save to hide widgets when using GUI Save button.
    if not hasattr(fig, "_widgets_save_wrapped"):
        orig_print_figure = fig.canvas.print_figure

        def wrapped_print_figure(*args, **kwargs):
            _set_axis_controls_visible(fig, False)
            try:
                return orig_print_figure(*args, **kwargs)
            finally:
                _set_axis_controls_visible(fig, True)

        fig.canvas.print_figure = wrapped_print_figure
        fig._widgets_save_wrapped = True

    return checks, text_boxes


def attach_group_controls(fig, axes, cfg):
    """
    Add controls for grouped row figures.
    - log-x toggle (single checkbox)
    - shared min/max text fields (x and y)
    All controls apply to every axis in the figure.
    """
    def apply_to_all(func):
        for ax in axes:
            func(ax)

    def apply_limits(_):
        def parse(val):
            val = val.strip()
            return None if val == "" else float(val)

        new_min_x = parse(text_boxes["min_x"].text)
        new_max_x = parse(text_boxes["max_x"].text)
        new_min_y = parse(text_boxes["min_y"].text)
        new_max_y = parse(text_boxes["max_y"].text)

        def set_limits(ax):
            xscale = ax.get_xscale()
            yscale = ax.get_yscale()

            min_x = new_min_x
            max_x = new_max_x
            min_y = new_min_y
            max_y = new_max_y

            if xscale == "log":
                min_x = min_x if min_x is None or min_x > 0 else None
                max_x = max_x if max_x is None or max_x > 0 else None
            if yscale == "log":
                min_y = min_y if min_y is None or min_y > 0 else None
                max_y = max_y if max_y is None or max_y > 0 else None

            if min_x is not None or max_x is not None:
                ax.set_xlim(left=min_x, right=max_x)
            if min_y is not None or max_y is not None:
                ax.set_ylim(bottom=min_y, top=max_y)

        apply_to_all(set_limits)
        fig.canvas.draw_idle()

    # Checkbox for log-x toggle (single option)
    cx = fig.add_axes([0.08, 0.08, 0.12, 0.06])
    logx_checks = CheckButtons(cx, ["log-x"], [cfg["log_x"]])

    def on_logx(_label):
        (logx,) = logx_checks.get_status()
        def set_scale(ax):
            ax.set_xscale("log" if logx else "linear")
            ax.autoscale(enable=True, axis="x")
        apply_to_all(set_scale)
        fig.canvas.draw_idle()

    logx_checks.on_clicked(on_logx)

    # Text boxes for limits (shared)
    box_positions = {
        "min_x": (0.24, 0.08),
        "max_x": (0.40, 0.08),
        "min_y": (0.56, 0.08),
        "max_y": (0.72, 0.08),
    }
    text_boxes = {}
    for key, (x0, y0) in box_positions.items():
        bax = fig.add_axes([x0, y0, 0.14, 0.06])
        init_val = "" if cfg.get(key) is None else str(cfg[key])
        text_boxes[key] = TextBox(bax, f"{key} ", initial=init_val)

    for tb in text_boxes.values():
        tb.on_submit(apply_limits)

    # Persist widget refs on the figure to avoid garbage collection.
    if not hasattr(fig, "_axis_controls"):
        fig._axis_controls = []
    fig._axis_controls.append({"checks": logx_checks, "text_boxes": text_boxes})
    fig._group_apply_limits = apply_limits

    # Wrap the canvas save to hide widgets when using GUI Save button.
    if not hasattr(fig, "_widgets_save_wrapped"):
        orig_print_figure = fig.canvas.print_figure

        def wrapped_print_figure(*args, **kwargs):
            _set_axis_controls_visible(fig, False)
            try:
                return orig_print_figure(*args, **kwargs)
            finally:
                _set_axis_controls_visible(fig, True)

        fig.canvas.print_figure = wrapped_print_figure
        fig._widgets_save_wrapped = True

    return logx_checks, text_boxes


def attach_group_logy_toggles(fig, axes, cfg):
    """
    Add one log-y toggle per axis in the bottom gutter.
    Toggles are aligned under each subplot.
    """
    n_axes = len(axes)
    if n_axes == 0:
        return []

    toggles = []
    for ax in axes:
        pos = ax.get_position()
        box_w = min(0.14, pos.width * 0.8)
        box_h = 0.05
        x0 = pos.x0 + (pos.width - box_w) / 2
        y0 = max(0.02, pos.y0 - 0.10)
        tax = fig.add_axes([x0, y0, box_w, box_h])
        check = CheckButtons(tax, ["log-y"], [cfg["log_y"]])

        def on_toggle_factory(target_ax, target_check):
            def on_toggle(_label):
                (logy,) = target_check.get_status()
                target_ax.set_yscale("log" if logy else "linear")
                target_ax.autoscale(enable=True, axis="y")
                fig.canvas.draw_idle()
            return on_toggle

        check.on_clicked(on_toggle_factory(ax, check))
        toggles.append(check)

    if not hasattr(fig, "_axis_controls"):
        fig._axis_controls = []
    fig._axis_controls.append({"checks": toggles, "text_boxes": {}})

    # Wrap the canvas save to hide widgets when using GUI Save button.
    if not hasattr(fig, "_widgets_save_wrapped"):
        orig_print_figure = fig.canvas.print_figure

        def wrapped_print_figure(*args, **kwargs):
            _set_axis_controls_visible(fig, False)
            try:
                return orig_print_figure(*args, **kwargs)
            finally:
                _set_axis_controls_visible(fig, True)

        fig.canvas.print_figure = wrapped_print_figure
        fig._widgets_save_wrapped = True

    return toggles


def _set_axis_controls_visible(fig, visible: bool):
    """Show/hide widget axes without dropping them; used to keep saves clean."""
    if not hasattr(fig, "_axis_controls"):
        return
    for ctrl in fig._axis_controls:
        checks = ctrl.get("checks")
        if checks is not None:
            if isinstance(checks, list):
                for check in checks:
                    check.ax.set_visible(visible)
            else:
                checks.ax.set_visible(visible)
        text_boxes = ctrl.get("text_boxes", {})
        for tb in text_boxes.values():
            tb.ax.set_visible(visible)
        radio = ctrl.get("legend_radio")
        if radio is not None:
            radio.ax.set_visible(visible)

# ---------------------------- Plotters ---------------------------- #

def plot_single_run_single_metric(api, cfg, figs):
    label, run_path, color = resolve_run_spec(cfg["team_name"], cfg["project_name"], cfg["run_specs"][0])
    run = fetch_run_or_raise(api, run_path)
    try:
        df, step_col = fetch_run_history(
            run,
            keys=cfg["all_keys"],
            fetch_method=cfg["fetch_method"],
            page_size=cfg["page_size"],
            max_rows=cfg.get("max_rows"),
            max_step_fetch=cfg.get("max_step_fetch"),
        )
    except MetricFetchError as e:
        print(f"Skipping {label}: {e}")
        return
    if cfg["single_metric"] not in df.columns:
        print(
            f"Skipping {label}: missing metric '{cfg['single_metric']}'. Available columns: {list(df.columns)}"
        )
        return
    plot_df = clean_metric_df(df, step_col, cfg["single_metric"])
    if plot_df.empty:
        print(f"Skipping {label}: no data for metric '{cfg['single_metric']}' after cleaning.")
        return
    if cfg.get("max_step_fetch") is not None:
        plot_df = plot_df[plot_df[step_col] <= cfg["max_step_fetch"]]

    plot_df, x_col, x_label = prepare_step_axis(
        plot_df,
        run,
        step_col,
        cfg["normalize_by_distance"],
        cfg["lr_key"],
        interv_lr=cfg.get("interv_lr", False),
        full_df=df,
    )
    fig, ax = plt.subplots()
    line = ax.plot(plot_df[x_col], plot_df[cfg["single_metric"]], color=color)[0] if color else \
        ax.plot(plot_df[x_col], plot_df[cfg["single_metric"]])[0]
    if should_add_eos_for_metric(cfg["single_metric"], cfg):
        maybe_add_eos(
            ax,
            plot_df,
            x_col,
            run,
            step_col,
            cfg,
            color=line.get_color() if ax.lines else None,
            label_prefix="",
            run_label=label,
            current_metric=cfg["single_metric"],
            enabled=True,
        )
    if cfg["max_x"] is not None or cfg["min_x"] is not None:
        ax.set_xlim(left=cfg["min_x"], right=cfg["max_x"])
    ax.set_xlabel(x_label)
    ax.set_ylabel(cfg["single_metric"])
    title = cfg["metric_titles"].get(cfg["single_metric"], cfg["single_metric"])
    ax.set_title(f"{title} for {label}")
    apply_log(ax, cfg["log_x"], cfg["log_y"])
    if cfg.get("max_y") is not None or cfg.get("min_y") is not None:
        ax.set_ylim(bottom=cfg.get("min_y"), top=cfg.get("max_y"))
    ax.minorticks_on()
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc=cfg.get("legend_loc", "best"))
    figs.append((fig, f"{run.name or label}_{cfg['single_metric']}"))

def plot_multi_run_single_metric(api, cfg, figs):
    fig, ax = plt.subplots()
    x_label = cfg["step_col"]

    for run_spec in cfg["run_specs"]:
        label, run_path, color = resolve_run_spec(cfg["team_name"], cfg["project_name"], run_spec)
        run = fetch_run_or_raise(api, run_path)
        try:
            df, step_col = fetch_run_history(
                run,
                keys=cfg["all_keys"],
                fetch_method=cfg["fetch_method"],
                page_size=cfg["page_size"],
                max_rows=cfg.get("max_rows"),
                max_step_fetch=cfg.get("max_step_fetch"),
            )
        except MetricFetchError as e:
            print(f"Skipping {label}: {e}")
            continue
        if cfg["single_metric"] not in df.columns:
            print(
                f"Skipping {label}: missing metric '{cfg['single_metric']}'. Available columns: {list(df.columns)}"
            )
            continue
        plot_df = clean_metric_df(df, step_col, cfg["single_metric"])
        if plot_df.empty:
            print(f"Skipping {label}: no data for metric '{cfg['single_metric']}' after cleaning.")
            continue
        if cfg.get("max_step_fetch") is not None:
            plot_df = plot_df[plot_df[step_col] <= cfg["max_step_fetch"]]

        plot_df, x_col, x_label = prepare_step_axis(
            plot_df,
            run,
            step_col,
            cfg["normalize_by_distance"],
            cfg["lr_key"],
            interv_lr=cfg.get("interv_lr", False),
            full_df=df,
        )
        plot_kwargs = {"label": label}
        if color:
            plot_kwargs["color"] = color
        line = ax.plot(plot_df[x_col], plot_df[cfg["single_metric"]], **plot_kwargs)[0]
        if should_add_eos_for_metric(cfg["single_metric"], cfg):
            maybe_add_eos(
                ax,
                plot_df,
                x_col,
                run,
                step_col,
                cfg,
                color=line.get_color(),
                label_prefix=f"{label}: ",
                run_label=label,
                current_metric=cfg["single_metric"],
                enabled=True,
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(cfg["metric_titles"].get(cfg["single_metric"], cfg["single_metric"]))
    title = cfg["metric_titles"].get(cfg["single_metric"], cfg["single_metric"])
    ax.set_title(f"{title}")
    if cfg["max_x"] is not None or cfg["min_x"] is not None:
        ax.set_xlim(left=cfg["min_x"], right=cfg["max_x"])
    if cfg.get("max_y") is not None or cfg.get("min_y") is not None:
        ax.set_ylim(bottom=cfg.get("min_y"), top=cfg.get("max_y"))
    apply_log(ax, cfg["log_x"], cfg["log_y"])
    ax.minorticks_on()
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc=cfg.get("legend_loc", "best"))
    figs.append((fig, f"multi-run_{title}"))

def plot_grid_multi_metric_multi_run(api, cfg, figs):
    num_metrics = len(cfg["metrics"])
    rows = math.ceil(num_metrics / cfg["grid_cols"])
    fig, axes = plt.subplots(rows, cfg["grid_cols"], figsize=(5 * cfg["grid_cols"], 3.5 * rows))
    axes = axes.flatten()

    for metric_idx, metric in enumerate(tqdm(cfg["metrics"], desc="Plotting metrics")):
        ax = axes[metric_idx]
        x_label = cfg["step_col"]
        for run_spec in cfg["run_specs"]:
            label, run_path, color = resolve_run_spec(cfg["team_name"], cfg["project_name"], run_spec)
            run = fetch_run_or_raise(api, run_path)
            try:
                df, step_col = fetch_run_history(
                    run,
                    keys=cfg["all_keys"],
                    fetch_method=cfg["fetch_method"],
                    page_size=cfg["page_size"],
                    max_rows=cfg.get("max_rows"),
                    max_step_fetch=cfg.get("max_step_fetch"),
                )
            except MetricFetchError as e:
                print(f"Skipping {label}: {e}")
                continue
            if metric not in df.columns:
                print(f"Skipping {label}: missing metric '{metric}'. Available columns: {list(df.columns)}")
                continue
            plot_df = clean_metric_df(df, step_col, metric)
            if plot_df.empty:
                print(f"Skipping {label}: no data for metric '{metric}' after cleaning.")
                continue
            if cfg.get("max_step_fetch") is not None:
                plot_df = plot_df[plot_df[step_col] <= cfg["max_step_fetch"]]

            plot_df, x_col, x_label = prepare_step_axis(
                plot_df,
                run,
                step_col,
                cfg["normalize_by_distance"],
                cfg["lr_key"],
                interv_lr=cfg.get("interv_lr", False),
                full_df=df,
            )
            plot_kwargs = {"label": label}
            if color:
                plot_kwargs["color"] = color
            line = ax.plot(plot_df[x_col], plot_df[metric], **plot_kwargs)[0]
            if should_add_eos_for_metric(metric, cfg):
                maybe_add_eos(
                    ax,
                    plot_df,
                    x_col,
                    run,
                    step_col,
                    cfg,
                    color=line.get_color(),
                    label_prefix=f"{label}: ",
                    run_label=label,
                    current_metric=metric,
                    enabled=True,
                )

        apply_log(ax, cfg["log_x"], cfg["log_y"])
        ax.set_xlabel(x_label)
        display_name = cfg["metric_titles"].get(metric, metric)
        ax.set_ylabel(display_name)
        ax.set_title(f"{display_name}")
        if cfg["max_x"] is not None or cfg["min_x"] is not None:
            ax.set_xlim(left=cfg["min_x"], right=cfg["max_x"])
        if cfg.get("max_y") is not None or cfg.get("min_y") is not None:
            ax.set_ylim(bottom=cfg.get("min_y"), top=cfg.get("max_y"))
        ax.minorticks_on()
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc=cfg.get("legend_loc", "best"))

    for ax in axes[num_metrics:]:
        ax.set_visible(False)

    fig.tight_layout()
    figs.append((fig, f"{cfg['project_name']}_grid"))

def plot_each_metric_separately(api, cfg, figs):
    """Create one figure per metric (multi-run overlay). Easier to view side-by-side and save individually."""
    for metric in tqdm(cfg["metrics"], desc="Plotting metrics (separate)"):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        x_label = cfg["step_col"]
        for run_spec in cfg["run_specs"]:
            label, run_path, color = resolve_run_spec(cfg["team_name"], cfg["project_name"], run_spec)
            run = fetch_run_or_raise(api, run_path)
            try:
                df, step_col = fetch_run_history(
                    run,
                    keys=cfg["all_keys"],
                    fetch_method=cfg["fetch_method"],
                    page_size=cfg["page_size"],
                    max_rows=cfg.get("max_rows"),
                    max_step_fetch=cfg.get("max_step_fetch"),
                )
            except MetricFetchError as e:
                print(f"Skipping {label}: {e}")
                continue
            if metric not in df.columns:
                print(f"Skipping {label}: missing metric '{metric}'. Available columns: {list(df.columns)}")
                continue
            plot_df = clean_metric_df(df, step_col, metric)
            if plot_df.empty:
                print(f"Skipping {label}: no data for metric '{metric}' after cleaning.")
                continue
            if cfg.get("max_step_fetch") is not None:
                plot_df = plot_df[plot_df[step_col] <= cfg["max_step_fetch"]]

            plot_df, x_col, x_label = prepare_step_axis(
                plot_df,
                run,
                step_col,
                cfg["normalize_by_distance"],
                cfg["lr_key"],
                interv_lr=cfg.get("interv_lr", False),
                full_df=df,
            )
            plot_kwargs = {"label": label}
            if color:
                plot_kwargs["color"] = color
            line = ax.plot(plot_df[x_col], plot_df[metric], **plot_kwargs)[0]
            if should_add_eos_for_metric(metric, cfg):
                maybe_add_eos(
                    ax,
                    plot_df,
                    x_col,
                    run,
                    step_col,
                    cfg,
                    color=line.get_color(),
                    label_prefix=f"{label}: ",
                    run_label=label,
                    enabled=True,
                )

        apply_log(ax, cfg["log_x"], cfg["log_y"])
        ax.set_xlabel(x_label)
        display_name = cfg["metric_titles"].get(metric, metric)
        ax.set_ylabel(display_name)
        ax.set_title(f"{display_name}")
        if cfg["max_x"] is not None or cfg["min_x"] is not None:
            ax.set_xlim(left=cfg["min_x"], right=cfg["max_x"])
        if cfg.get("max_y") is not None or cfg.get("min_y") is not None:
            ax.set_ylim(bottom=cfg.get("min_y"), top=cfg.get("max_y"))
        ax.minorticks_on()
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc=cfg.get("legend_loc", "best"))

        # In-window controls for toggling log scale and limits
        fig.tight_layout()
        attach_axis_controls(fig, ax, cfg)
        figs.append((fig, f"{display_name}"))


def plot_metric_groups(api, cfg, figs):
    """
    Plot metrics grouped by row. Each row is a group; each column is a metric.
    Designed for squarish subplots and optional shared legends.
    """
    if not cfg.get("metric_groups"):
        return

    groups = cfg["metric_groups"]
    subplot_size = cfg.get("group_subplot_size", 3.4)
    rect_size = cfg.get("group_subplot_size_rect", (5.0, 3.2))
    left_margin = cfg.get("group_left_margin", 0.06)
    ylabel_pad = cfg.get("group_ylabel_pad", 24)
    group_wspace_val = cfg.get("group_wspace", 0.18)
    plot_area_right_default = cfg.get("group_plot_area_right", 0.82)

    for group in groups:
        group_label = group.get("label") or ""
        metrics_row = group.get("metrics", [])
        if not metrics_row:
            continue
        cols = len(metrics_row)
        use_rect = group_label in {
            "lambda_max (input_space_prototypes)",
            "full_loss (input_space_prototypes)",
        }
        if use_rect:
            subplot_w, subplot_h = rect_size
        else:
            subplot_w = subplot_size
            subplot_h = subplot_size
        plot_area_right = plot_area_right_default
        plot_area_bottom = 0.22
        plot_area_top = 0.94
        fig_w = (cols * subplot_w) / plot_area_right
        fig_h = subplot_h / (plot_area_top - plot_area_bottom)
        fig, axes = plt.subplots(1, cols, figsize=(fig_w, fig_h))
        if cols == 1:
            axes = [axes]

        def metric_short_title(metric_name: str) -> str:
            if "input_space_prototypes" in metric_name:
                if "/inlier_points/inliers/" in metric_name:
                    return "Inliers"
                if "/boundary_points/boundary/" in metric_name:
                    return "Boundary"
                if "/injected_inliers/injected_inliers/" in metric_name:
                    return "Injected Inliers"
                if "/injected_boundary/injected_boundary/" in metric_name:
                    return "Injected Boundary"
                if "/injected_x_outlier/injected_x_outlier/" in metric_name:
                    return "Injected X Outlier"
                if "/injected_y_outlier/injected_y_outlier/" in metric_name:
                    return "Injected Y Outlier"
            return cfg["metric_titles"].get(metric_name, metric_name)

        def group_y_label(metrics_row):
            if any("/lambda_max" in m for m in metrics_row):
                return "λmax"
            if any("/full_loss" in m for m in metrics_row):
                return "Loss"
            if any("/grad_hessian_grad" in m for m in metrics_row):
                return "gᵀHg / ||g||²"
            if any("/grad_vmax_cos2" in m for m in metrics_row):
                return "cos²(g, v_max)"
            if any("/grad_norm" in m for m in metrics_row):
                return "Gradient Norm"
            if any("/stability_ratio" in m for m in metrics_row):
                return "(gᵀHg / ||g||²) / λmax"
            return ""

        for col_idx, metric in enumerate(metrics_row):
            ax = axes[col_idx]
            x_label = cfg["step_col"]
            for run_spec in cfg["run_specs"]:
                label, run_path, color = resolve_run_spec(cfg["team_name"], cfg["project_name"], run_spec)
                run = fetch_run_or_raise(api, run_path)
                try:
                    df, step_col = fetch_run_history(
                        run,
                        keys=cfg["all_keys"],
                        fetch_method=cfg["fetch_method"],
                        page_size=cfg["page_size"],
                        max_rows=cfg.get("max_rows"),
                        max_step_fetch=cfg.get("max_step_fetch"),
                    )
                except MetricFetchError as e:
                    print(f"Skipping {label}: {e}")
                    continue
                computed_used = False
                if metric in df.columns:
                    plot_df = clean_metric_df(df, step_col, metric)
                else:
                    plot_df = compute_metric_df(metric, df, step_col, run, label, cfg)
                    if plot_df is None:
                        print(
                            f"Skipping {label}: missing metric '{metric}'. Available columns: {list(df.columns)}"
                        )
                        continue
                    computed_used = True
                if plot_df.empty:
                    if computed_used:
                        print(
                            f"Skipping {label}: no valid rows for computed metric '{metric}'."
                        )
                        continue
                    print(f"Skipping {label}: no data for metric '{metric}' after cleaning.")
                    continue
                if cfg.get("max_step_fetch") is not None:
                    plot_df = plot_df[plot_df[step_col] <= cfg["max_step_fetch"]]

                plot_df, x_col, x_label = prepare_step_axis(
                    plot_df,
                    run,
                    step_col,
                    cfg["normalize_by_distance"],
                    cfg["lr_key"],
                    interv_lr=cfg.get("interv_lr", False),
                    full_df=df,
                )
                plot_kwargs = {"label": label}
                if color:
                    plot_kwargs["color"] = color
                line = ax.plot(plot_df[x_col], plot_df[metric], **plot_kwargs)[0]
                if should_add_eos_for_metric(metric, cfg):
                    maybe_add_eos(
                        ax,
                        plot_df,
                        x_col,
                        run,
                        step_col,
                        cfg,
                        color=line.get_color(),
                        label_prefix=f"{label}: ",
                        run_label=label,
                        current_metric=metric,
                        enabled=True,
                    )

            if cfg.get("group_apply_scales", True):
                apply_log(ax, cfg["log_x"], cfg["log_y"])
                if cfg["max_x"] is not None or cfg["min_x"] is not None:
                    ax.set_xlim(left=cfg["min_x"], right=cfg["max_x"])
                if cfg.get("max_y") is not None or cfg.get("min_y") is not None:
                    ax.set_ylim(bottom=cfg.get("min_y"), top=cfg.get("max_y"))

            ax.set_xlabel(x_label)
            if col_idx == 0:
                ax.set_ylabel(group_y_label(metrics_row), labelpad=ylabel_pad)
            else:
                ax.set_ylabel("")
            ax.set_title(metric_short_title(metric), pad=6)
            ax.minorticks_on()
            ax.grid(True, alpha=0.3, which="both")

            if group.get("legend_mode", cfg.get("group_legend_mode", "per-ax")) == "per-ax":
                ax.legend(loc=cfg.get("legend_loc", "best"))

        # Shared legend per row image
        legend_mode = group.get("legend_mode", cfg.get("group_legend_mode", "per-ax"))
        if legend_mode in {"shared-right", "shared-bottom"}:
            handles = []
            labels = []
            for ax in axes:
                h, l = ax.get_legend_handles_labels()
                for handle, label in zip(h, l):
                    if label not in labels:
                        handles.append(handle)
                        labels.append(label)
            if handles:
                if legend_mode == "shared-right":
                    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(plot_area_right - 0.01, 0.5))
                else:
                    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4))

        if legend_mode == "shared-bottom":
            fig.tight_layout(rect=(left_margin, plot_area_bottom, plot_area_right, plot_area_top))
        else:
            fig.tight_layout(rect=(left_margin, plot_area_bottom, plot_area_right, plot_area_top))
        fig.subplots_adjust(wspace=group_wspace_val)

        if cfg.get("group_apply_scales", True):
            attach_group_controls(fig, axes, cfg)
            attach_group_logy_toggles(fig, axes, cfg)
        group_label = group.get("label")
        safe_label = (group_label or "group").replace("/", "-")
        figs.append((fig, f"{cfg['project_name']}_metric_group_{safe_label}"))

# ---------------------------- Main ---------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Generate W&B plots (CLI port of wandb_plots.ipynb).")
    parser.add_argument("--no-show", action="store_true", help="Do not display interactive windows.")
    parser.add_argument("--save", action="store_true", help="Save figures to save_dir (from config) or --save-dir.")
    parser.add_argument("--save-dir", type=str, default=None, help="Override save_dir for saved figures.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run non-interactively: implies --no-show, --save, and a timestamped save-dir under plots/<project>/<ts>/ unless --save-dir is given.",
    )
    parser.add_argument("--log-axes", type=str, default=None, choices=["x", "y", "both", "none"],
                        help="Override both axes at once (legacy flag).")
    parser.add_argument("--log-x", action="store_true", help="Enable log scale on x-axis.")
    parser.add_argument("--no-log-x", action="store_true", help="Disable log scale on x-axis.")
    parser.add_argument("--log-y", action="store_true", help="Enable log scale on y-axis.")
    parser.add_argument("--no-log-y", action="store_true", help="Disable log scale on y-axis.")
    parser.add_argument("--max-x", type=float, default=None, help="Override max_x.")
    parser.add_argument("--min-x", type=float, default=None, help="Override min_x.")
    parser.add_argument("--max-y", type=float, default=None, help="Override max_y.")
    parser.add_argument("--min-y", type=float, default=None, help="Override min_y.")
    parser.add_argument("--grid-cols", type=int, default=None, help="Override grid_cols.")
    parser.add_argument("--normalize", action="store_true", help="Enable normalize_by_distance regardless of config.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable normalize_by_distance regardless of config.")
    parser.add_argument("--interv-lr", action="store_true",
                        help="Use per-step lr history to compute distance when normalizing.")
    parser.add_argument("--fetch-method", type=str, choices=["history", "scan_history"], default=None,
                        help="Select fetch method for run data.")
    parser.add_argument("--page_size", type=int, default=None, help="Page size for scan_history fetch method.")
    parser.add_argument("--enable-eos", action="store_true", help="Draw EOS threshold overlays.")
    parser.add_argument("--eos-eta", type=float, default=None, help="Override eta used for EOS.")
    parser.add_argument("--eos-eta-key", type=str, default=None, help="Config key for eta when not overridden.")
    parser.add_argument("--eos-adam", action="store_true", help="Force EOS to use Adam threshold (38/eta).")
    parser.add_argument("--eos-sgd", action="store_true", help="Force EOS to use SGD threshold (2/eta).")
    parser.add_argument("--no-eos-vertical", action="store_true", help="Disable vertical crossing line.")
    parser.add_argument("--eos-threshold-legend", action="store_true", help="Show threshold/cross labels in legend.")
    parser.add_argument(
        "--plot-metric-groups",
        action="store_true",
        help="Enable grouped metric plots (row-wise groups).",
    )
    parser.add_argument(
        "--group-subplot-size",
        type=float,
        default=None,
        help="Size (inches) per grouped subplot; larger = squarer tiles.",
    )
    parser.add_argument(
        "--group-legend",
        type=str,
        choices=["per-ax", "shared-right", "shared-bottom"],
        default=None,
        help="Legend placement for grouped plots.",
    )
    parser.add_argument(
        "--no-group-scales",
        action="store_true",
        help="Disable log/min/max scaling for grouped plots.",
    )
    parser.add_argument(
        "--eos-vertical-allow",
        action="append",
        default=None,
        help="Run labels that should show EOS vertical lines; repeat flag for each label "
             "(useful with --no-eos-vertical to re-enable selected runs).",
    )
    parser.add_argument(
        "--eos-horizontal-allow",
        action="append",
        default=None,
        help="Run labels that should show EOS horizontal threshold lines; repeat flag for each label.",
    )
    parser.add_argument(
        "--eos-cross-all-metrics",
        action="store_true",
        help="Use the currently plotted metric as the EOS crossing driver (enables vertical lines for any metric).",
    )
    parser.add_argument(
        "--eos-cross-metrics",
        type=str,
        default=None,
        help="| -separated list of metric names allowed as EOS crossing drivers. "
             "Ignored if --eos-cross-all-metrics is set.",
    )
    return parser.parse_args()


def apply_overrides(cfg, args):
    if args.log_x:
        cfg["log_x"] = True
    if args.no_log_x:
        cfg["log_x"] = False
    if args.log_y:
        cfg["log_y"] = True
    if args.no_log_y:
        cfg["log_y"] = False
    if args.max_x is not None:
        cfg["max_x"] = args.max_x
    if args.min_x is not None:
        cfg["min_x"] = args.min_x
    if args.max_y is not None:
        cfg["max_y"] = args.max_y
    if args.min_y is not None:
        cfg["min_y"] = args.min_y
    if args.grid_cols is not None and args.grid_cols > 0:
        cfg["grid_cols"] = args.grid_cols
    if args.normalize:
        cfg["normalize_by_distance"] = True
    if args.no_normalize:
        cfg["normalize_by_distance"] = False
    if args.interv_lr:
        cfg["interv_lr"] = True
    if args.save_dir is not None:
        cfg["save_dir"] = args.save_dir
    elif args.headless:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg["save_dir"] = str(
            Path(__file__).resolve().parent / "research-tick-results" / "ticks" / ts
            / "plots" / cfg["project_name"]
        )
    if args.headless:
        args.no_show = True
        args.save = True
    if args.fetch_method is not None:
        cfg["fetch_method"] = args.fetch_method
    if args.page_size is not None:
        cfg["page_size"] = args.page_size
    if args.enable_eos:
        cfg["enable_eos"] = True
    if args.eos_eta is not None:
        cfg["eos_eta_override"] = args.eos_eta
    if args.eos_eta_key is not None:
        cfg["eos_eta_key"] = args.eos_eta_key
    if args.eos_adam:
        cfg["eos_adam_override"] = True
    if args.eos_sgd:
        cfg["eos_adam_override"] = False
    if args.no_eos_vertical:
        cfg["eos_show_vertical"] = False
    if args.eos_threshold_legend:
        cfg["eos_show_threshold_legend"] = True
    if args.eos_vertical_allow:
        cfg["eos_vertical_allow"] = [item.strip() for item in args.eos_vertical_allow if item.strip()]
    if args.eos_horizontal_allow:
        cfg["eos_horizontal_allow"] = [item.strip() for item in args.eos_horizontal_allow if item.strip()]
    if args.eos_cross_all_metrics:
        cfg["eos_cross_all_metrics"] = True
    if args.eos_cross_metrics:
        cfg["eos_cross_metrics"] = [m.strip() for m in args.eos_cross_metrics.split("|") if m.strip()]
    if args.group_subplot_size is not None:
        cfg["group_subplot_size"] = args.group_subplot_size
    if args.group_legend is not None:
        cfg["group_legend_mode"] = args.group_legend
    if args.no_group_scales:
        cfg["group_apply_scales"] = False
    if args.plot_metric_groups:
        cfg["plot_metric_groups"] = True
    return cfg


def save_figures(figs, save_path):
    if not save_path:
        print("Save requested but save_dir is empty; skipping save.")
        return
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    for fig, name in figs:
        safe_name = name.replace("/", "-")
        outfile = path / f"{safe_name}.png"
        _set_axis_controls_visible(fig, False)
        fig.savefig(outfile, dpi=200, bbox_inches="tight")
        _set_axis_controls_visible(fig, True)
        print(f"Saved {outfile}")


def main():
    args = parse_args()

    # Copy config into a dict for easy overrides.
    cfg = {
        "team_name": team_name,
        "project_name": project_name,
        "run_specs": run_specs,
        "step_col": step_col,
        "single_metric": single_metric,
        "metrics": metrics,
        "history_samples": history_samples,
        "log_x": log_x,
        "log_y": log_y,
        "save_dir": save_dir,
        "grid_cols": grid_cols,
        "metric_groups": metric_groups,
        "group_subplot_size": group_subplot_size,
        "group_subplot_size_rect": group_subplot_size_rect,
        "group_wspace": group_wspace,
        "group_plot_area_right": group_plot_area_right,
        "group_ylabel_pad": group_ylabel_pad,
        "group_legend_mode": group_legend_mode,
        "group_apply_scales": group_apply_scales,
        "plot_metric_groups": True,
        "normalize_by_distance": normalize_by_distance,
        "interv_lr": interv_lr,
        "lr_key": lr_key,
        "max_x": max_x,
        "min_x": min_x,
        "max_y": max_y,
        "min_y": min_y,
        "max_step_fetch": max_step_fetch,
        "all_keys": None,
        "legend_loc": legend_loc,
        "metric_titles": metric_titles,
        "stability_ratio_smooth_window": stability_ratio_smooth_window,
        "fetch_method": fetch_method,
        "page_size": page_size,
        "max_rows": max_rows,
        "enable_eos": enable_eos,
        "eos_eta_override": eos_eta_override,
        "eos_eta_key": eos_eta_key,
        "eos_adam_override": eos_adam_override,
        "eos_show_vertical": eos_show_vertical,
        "eos_show_threshold_legend": eos_show_threshold_legend,
        "eos_vertical_allow": None,
        "eos_horizontal_allow": None,
        "eos_cross_all_metrics": False,
        "eos_cross_metrics": None,
    }

    cfg = apply_overrides(cfg, args)

    # Prefer lambda_max_precond_adam in core row when present.
    metrics_set = set(cfg["metrics"])
    core_group = None
    for group in cfg.get("metric_groups", []):
        if group.get("label") == "core":
            core_group = group
            break
    if core_group is not None:
        lambda_metrics = []
        if "lambda_max_precond_adam" in metrics_set:
            lambda_metrics.append("lambda_max_precond_adam")
        if "lambda_max" in metrics_set:
            lambda_metrics.append("lambda_max")
        if lambda_metrics:
            core_group["metrics"] = ["full_loss"] + lambda_metrics

    # Build a unified key list to encourage caching: step + all metrics (unique)
    group_metrics = [
        metric
        for group in (cfg.get("metric_groups") or [])
        for metric in group.get("metrics", [])
    ]
    unique_metrics = list(
        dict.fromkeys(
            ["step", "_step"]
            + cfg["metrics"]
            + [cfg["single_metric"]]
            + group_metrics
            + stability_ratio_inputs
            + [cfg["lr_key"]]
        )
    )
    cfg["all_keys"] = unique_metrics

    # Authentication and API setup.
    os.environ["WANDB_SILENT"] = "true"
    wandb.login()  # requires API key in environment
    api = wandb.Api()

    figs = []
    # plot_single_run_single_metric(api, cfg, figs)
    # plot_multi_run_single_metric(api, cfg, figs)
    plot_each_metric_separately(api, cfg, figs)
    # plot_grid_multi_metric_multi_run(api, cfg, figs)  # Grid view disabled; per-metric figures instead
    if cfg.get("plot_metric_groups"):
        plot_metric_groups(api, cfg, figs)

    if args.save:
        save_figures(figs, cfg["save_dir"])

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
