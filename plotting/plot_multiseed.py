"""
Multi-seed Figure: EoS creates a selective learning trade-off (median + IQR across seeds).

Loads ALL runs from a W&B project, groups by seed, separates EoS (baseline)
from exit-EoS (fork), and plots median + IQR shaded bands.

Layout: 1×2
  Left:  prototype loss (boundary + input-outlier, both conditions) median ± IQR
  Right: lambda_max — individual seed traces (thin) + bold median

Usage:
    python plot_multiseed.py \
        --project entity/fork-eoss-fork-full_gd-mlp-mse-cifar10_2cls \
        --classes 1,9 --seeds 1111 1234 3333 --mode full --smooth 1
"""

import argparse
import json
import os
import pickle
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import pandas as pd
import wandb

# ── prototype registry (Okabe-Ito) ────────────────────────────────────────────
PROTOTYPE_GROUPS = [
    ("injected_inliers/injected_inliers",      "inlier",         "#E69F00"),
    ("injected_y_outlier/injected_y_outlier",  "output-outlier", "#009E73"),
    ("injected_x_outlier/injected_x_outlier",  "input-outlier",  "#0072B2"),
    ("injected_boundary/injected_boundary",    "boundary",       "#D55E00"),
]

_BLACK          = "#111111"
_GRAY           = "#999999"
LMAX_KEY        = "lambda_max"
GLOBAL_LOSS_KEY = "full_loss"
PREFIX          = "input_space_prototypes"


def _apply_rcparams():
    matplotlib.rcParams.update({
        "font.family":           "sans-serif",
        "font.sans-serif":       ["DejaVu Sans"],
        "mathtext.fontset":      "dejavusans",
        "font.size":             9,
        "axes.titlesize":        10,
        "axes.labelsize":        10,
        "xtick.labelsize":       9,
        "ytick.labelsize":       9,
        "legend.fontsize":       8,
        "legend.title_fontsize": 8,
        "lines.linewidth":       1.5,
        "xtick.direction":       "in",
        "ytick.direction":       "in",
        "xtick.major.width":     0.8,
        "ytick.major.width":     0.8,
        "xtick.major.size":      3.5,
        "ytick.major.size":      3.5,
        "axes.linewidth":        0.8,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "legend.frameon":        False,
        "legend.borderpad":      0.2,
        "legend.labelspacing":   0.3,
        "legend.handlelength":   1.5,
        "legend.handletextpad":  0.4,
        "legend.columnspacing":  0.8,
        "figure.autolayout":     False,
        "figure.dpi":            300,
        "savefig.bbox":          "tight",
        "savefig.pad_inches":    0.02,
        "savefig.dpi":           300,
    })


def ploss_key(s):
    return f"{PREFIX}/{s}/full_loss"


def pkey(s, metric):
    return f"{PREFIX}/{s}/{metric}"


# ── wandb helpers ─────────────────────────────────────────────────────────────
def _cfg_val(v):
    return v["value"] if isinstance(v, dict) and "value" in v else v


def extract_meta(run):
    cfg = run.config
    if isinstance(cfg, str):
        cfg = json.loads(cfg)
    summ = run.summary
    if isinstance(summ, str):
        summ = json.loads(summ) if summ.strip() else {}

    lr_raw = cfg.get("lr", cfg.get("learning_rate", np.nan))
    lr = float(_cfg_val(lr_raw)) if lr_raw is not np.nan else np.nan

    # fallback: parse lr from run name  e.g. "_lr0.01_"
    if np.isnan(lr):
        m = re.search(r'_lr([0-9.eE+-]+)', run.name)
        if m:
            lr = float(m.group(1))

    try:
        t_star = summ.get("t_star", None)
    except (TypeError, KeyError, AttributeError):
        t_star = None
    if t_star is not None:
        try:
            t_star = int(t_star)
        except (TypeError, ValueError):
            t_star = None

    lmax_drop = _cfg_val(cfg.get("lmax_drop", False))
    drop_mult = _cfg_val(cfg.get("lmax_drop_mult", None)) if lmax_drop else None
    lr_low = lr * drop_mult if (drop_mult is not None and not np.isnan(lr)) else None

    # fallback: parse lrdrop from run name e.g. "lrdrop0.1"
    if lr_low is None and "lrdrop" in run.name.lower():
        m = re.search(r'lrdrop([0-9.eE+-]+)', run.name, re.IGNORECASE)
        if m:
            mult = float(m.group(1))
            lr_low = lr * mult

    init_seed = _cfg_val(cfg.get("init_seed", cfg.get("seed", None)))
    # fallback: parse seed from run name e.g. "_seed1234"
    if init_seed is None:
        m = re.search(r'_seed([0-9]+)', run.name, re.IGNORECASE)
        if m:
            init_seed = int(m.group(1))

    # Extract classes from config
    classes_raw = _cfg_val(cfg.get("classes", None))
    if classes_raw is not None:
        classes = tuple(int(c) for c in classes_raw)
    else:
        classes = None

    # For fork runs: parse original high LR, drop mult, and fork step from name
    # Name pattern: fork_exit_eos_seed{S}_lr{LR_HIGH}_drop{DROP}_from_step{STEP}
    fork_step = None
    m_fork = re.search(
        r'fork_exit_eos_seed\d+_lr([0-9.eE+-]+)_drop([0-9.eE+-]+)_from_step(\d+)',
        run.name)
    if m_fork:
        fork_lr_high = float(m_fork.group(1))
        fork_drop = float(m_fork.group(2))
        fork_step = int(m_fork.group(3))
        # For fork runs, derive lr_high/lr_low from the name unconditionally:
        # cfg.lr typically stores the baseline (high) lr, not the post-drop value.
        lr = fork_lr_high
        lr_low = fork_lr_high * fork_drop
        drop_mult = fork_drop
        t_star = fork_step

    return {
        "lr_high":   lr,
        "lr_low":    lr_low,
        "t_star":    t_star,
        "drop_mult": drop_mult,
        "init_seed": init_seed,
        "classes":   classes,
        "fork_step": fork_step,
    }


def get_history(run, keys):
    cache_dir = ".wandb_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{run.id}_scan_v3.pkl")
    if os.path.exists(cache_path):
        print(f"  Loading from cache: {cache_path}")
        return pickle.load(open(cache_path, "rb"))
    print(f"  Fetching all rows from wandb via scan_history (no downsampling)...")
    rows = list(run.scan_history(keys=["_step", "step"] + keys))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    if "step" in df.columns and df["step"].notna().any():
        result = {"step": df["step"].values.astype(float)}
    elif "_step" in df.columns:
        result = {"step": df["_step"].values.astype(float)}
    else:
        return None
    for k in keys:
        result[k] = df[k].values if k in df.columns else np.full(len(df), np.nan)

    if LMAX_KEY in keys:
        lmax_rows = list(run.scan_history(keys=["_step", "step", LMAX_KEY]))
        if lmax_rows:
            lmax_df = pd.DataFrame(lmax_rows)
            step_col = "step" if "step" in lmax_df.columns and lmax_df["step"].notna().any() else "_step"
            if step_col in lmax_df.columns and LMAX_KEY in lmax_df.columns:
                result["lmax_dense_step"] = lmax_df[step_col].values.astype(float)
                result["lmax_dense"] = lmax_df[LMAX_KEY].values

    pickle.dump(result, open(cache_path, "wb"))
    print(f"  Cached → {cache_path}")
    return result


def compute_t_eff(steps, meta):
    lr_high, t_star, lr_low = meta["lr_high"], meta["t_star"], meta["lr_low"]
    if lr_low is None or t_star is None:
        return steps * lr_high
    return np.where(
        steps <= t_star,
        steps * lr_high,
        t_star * lr_high + (steps - t_star) * lr_low,
    )


def _dense_lmax(hist, meta, fallback_t_eff):
    """Return (t_eff, lmax) using dense λ_max fetch if available."""
    if "lmax_dense" in hist and "lmax_dense_step" in hist:
        return compute_t_eff(hist["lmax_dense_step"], meta), hist["lmax_dense"]
    return fallback_t_eff, hist[LMAX_KEY]


def get_t_star_eff(meta, hist, t_eff):
    """Detect t* as the first point where lambda_max >= 2/lr_high."""
    dense_t, dense_lmax = _dense_lmax(hist, meta, t_eff)
    candidates = []
    above = np.where(~np.isnan(dense_lmax) & (dense_lmax >= 2.0 / meta["lr_high"]))[0]
    if len(above):
        candidates.append(float(dense_t[above[0]]))
    if meta["t_star"] is not None:
        candidates.append(meta["t_star"] * meta["lr_high"])
    if candidates:
        val = min(candidates)
        print(f"  t*_eff = {val:.1f}")
        return val
    return 0.0


def find_t_second(rd, **_kwargs):
    """Find t** = first t_eff where lambda_max >= 2/eta_low after t*."""
    meta = rd["meta"]
    if meta["lr_low"] is None or meta["lr_low"] <= 0:
        return None
    dense_t, dense_lmax = _dense_lmax(rd["hist"], meta, rd["t_eff"])
    threshold = 2.0 / meta["lr_low"]
    above = np.where(
        ~np.isnan(dense_lmax)
        & (dense_lmax >= threshold)
        & (dense_t >= rd["t_star_eff"])
    )[0]
    if not len(above):
        return None
    return float(dense_t[above[0]])


def rolling_mean(x, w):
    if w <= 1:
        return x
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        sl = x[max(0, i - w // 2): min(len(x), i + w // 2 + 1)]
        valid = sl[~np.isnan(sl)]
        if len(valid):
            out[i] = valid.mean()
    return out


# ── interpolation / aggregation ───────────────────────────────────────────────
def interpolate_to_grid(x_vals, y_vals, grid, smooth=1):
    """
    Interpolate y_vals (defined at x_vals) onto the common grid.
    Returns array of length len(grid) with NaN where extrapolation would occur.
    """
    mask = ~np.isnan(x_vals) & ~np.isnan(y_vals)
    if mask.sum() < 2:
        return np.full(len(grid), np.nan)
    xs = x_vals[mask]
    ys = y_vals[mask]
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    if smooth > 1:
        ys = rolling_mean(ys, smooth)
    # only interpolate within the data range
    result = np.interp(grid, xs, ys, left=np.nan, right=np.nan)
    # mask grid points outside the data range
    result[grid < xs[0]]  = np.nan
    result[grid > xs[-1]] = np.nan
    return result


def aggregate_seeds(run_list, key, grid, smooth=1):
    """
    Given a list of run dicts and a data key, interpolate each run onto grid
    and return (median, p25, p75, individual_curves) arrays.
    """
    curves = []
    for rd in run_list:
        x_vals = rd["x_vals"]
        y_vals = rd["hist"].get(key, np.full(len(x_vals), np.nan))
        curves.append(interpolate_to_grid(x_vals, y_vals, grid, smooth=smooth))
    nan_arr = np.full(len(grid), np.nan)
    if not curves:
        return nan_arr, nan_arr, nan_arr, []
    arr = np.vstack(curves)  # shape: (n_seeds, n_grid)
    with np.errstate(all="ignore"):
        median = np.nanmedian(arr, axis=0)
        p25    = np.nanpercentile(arr, 25, axis=0)
        p75    = np.nanpercentile(arr, 75, axis=0)
    n_valid = np.sum(~np.isnan(arr), axis=0)
    median[n_valid == 0] = np.nan
    p25[n_valid == 0]    = np.nan
    p75[n_valid == 0]    = np.nan
    return median, p25, p75, curves


# ── vlines and span ───────────────────────────────────────────────────────────
def draw_vlines_span(axes, tstar_x, tsecond_x, label_ax):
    if tstar_x is not None and tsecond_x is not None and tsecond_x > tstar_x:
        for ax in axes:
            ax.axvspan(tstar_x, tsecond_x, alpha=0.10, color="gray", zorder=0)

    if tstar_x is not None:
        for ax in axes:
            ax.axvline(tstar_x, color=_GRAY, linewidth=0.8,
                       linestyle="--", alpha=0.6, zorder=1)
        label_ax.annotate(
            r"$t^*$",
            xy=(tstar_x, 1.0), xycoords=("data", "axes fraction"),
            xytext=(0, 0), textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
            color=_BLACK, clip_on=False,
        )

    if tsecond_x is not None:
        for ax in axes:
            ax.axvline(tsecond_x, color=_GRAY, linewidth=0.8,
                       linestyle="--", alpha=0.6, zorder=1)
        label_ax.annotate(
            r"$t^{**}$",
            xy=(tsecond_x, 1.0), xycoords=("data", "axes fraction"),
            xytext=(0, 0), textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
            color=_BLACK, clip_on=False,
        )


# ── legend ────────────────────────────────────────────────────────────────────
# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    _apply_rcparams()
    api = wandb.Api()

    mode   = args.mode
    smooth = args.smooth

    all_keys = (
        [ploss_key(s) for s, _, _ in PROTOTYPE_GROUPS]
        + [pkey(s, "grad_norm") for s, _, _ in PROTOTYPE_GROUPS]
        + [pkey(s, "grad_vmax_cos2") for s, _, _ in PROTOTYPE_GROUPS]
        + [LMAX_KEY, GLOBAL_LOSS_KEY]
    )

    # ── load ALL runs from project ─────────────────────────────────────────── #
    print(f"\nFetching all runs from project: {args.project}")
    all_runs = [r for r in api.runs(args.project) if r.state != "deleted"]
    print(f"  Found {len(all_runs)} runs")

    # ── organize by (classes, seed) to handle duplicate fork runs ───────── #
    # For each (classes, seed), keep one baseline and one fork.
    # When multiple forks exist, the last one processed wins (matches
    # plot_fork_by_class_seed.py ordering where correct t_star-based
    # forks overwrite earlier wrong-step submissions).
    class_seed_slots = {}   # (classes, seed) -> {"baseline": rd, "fork": rd}

    exclude_run_ids = set(args.exclude_run_ids) if args.exclude_run_ids else set()

    for run in all_runs:
        if run.id in exclude_run_ids:
            print(f"\n{'='*50}\nSKIP (exclude-run-ids): {run.name}  (id={run.id})")
            continue
        print(f"\n{'='*50}\nProcessing: {run.name}  (id={run.id})")
        try:
            meta = extract_meta(run)
        except Exception as e:
            print(f"  SKIP extract_meta: {e}")
            continue

        if np.isnan(meta["lr_high"]):
            print(f"  SKIP: could not determine lr")
            continue

        # Filter by classes
        if args.classes and meta.get("classes") not in args.classes:
            print(f"  SKIP: classes {meta.get('classes')} not in {args.classes}")
            continue

        # Filter by seeds
        if args.seeds and meta.get("init_seed") not in args.seeds:
            print(f"  SKIP: seed {meta.get('init_seed')} not in requested seeds")
            continue

        # Filter forks by drop_mult
        is_fork_run = (meta.get("fork_step") is not None or
                       "fork_exit" in run.name.lower() or
                       "lrdrop" in run.name.lower())
        if is_fork_run and args.drop_filter is not None:
            dm = meta.get("drop_mult")
            if dm is None or not np.isclose(float(dm), args.drop_filter, atol=1e-6):
                print(f"  SKIP fork: drop_mult={dm} != {args.drop_filter}")
                continue

        hist = get_history(run, all_keys)
        if hist is None:
            print(f"  SKIP: no history")
            continue

        t_eff = compute_t_eff(hist["step"], meta)

        # determine t* for this run
        if args.t_star is not None:
            t_star_eff = float(args.t_star)
        else:
            t_star_eff = get_t_star_eff(meta, hist, t_eff)

        x_vals   = (t_eff - t_star_eff) if mode == "post" else t_eff
        t_star_x = 0.0 if mode == "post" else t_star_eff

        is_exit = ("lrdrop" in run.name.lower() or
                   "fork_exit" in run.name.lower() or
                   meta.get("fork_step") is not None)

        rd = {
            "name":       run.name,
            "meta":       meta,
            "hist":       hist,
            "t_eff":      t_eff,
            "t_star_eff": t_star_eff,
            "x_vals":     x_vals,
            "t_star_x":   t_star_x,
            "is_exit":    is_exit,
        }

        key = (meta.get("classes"), meta.get("init_seed"))
        if key not in class_seed_slots:
            class_seed_slots[key] = {"baseline": None, "fork": None}
        if is_exit:
            class_seed_slots[key]["fork"] = rd
        else:
            class_seed_slots[key]["baseline"] = rd

        print(f"  lr_high={meta['lr_high']}  lr_low={meta['lr_low']}  "
              f"seed={meta['init_seed']}  exit_eos={is_exit}")

    # Flatten deduplicated slots into run_data
    run_data = []
    for slot in class_seed_slots.values():
        if slot["baseline"] is not None:
            run_data.append(slot["baseline"])
        if slot["fork"] is not None:
            run_data.append(slot["fork"])

    if not run_data:
        print("No runs loaded."); return

    # ── split into EoS / exit-EoS ─────────────────────────────────────────── #
    eos_runs  = [rd for rd in run_data if not rd["is_exit"]]
    exit_runs = [rd for rd in run_data if rd["is_exit"]]
    print(f"\nEoS runs: {len(eos_runs)},  exit-EoS runs: {len(exit_runs)}")

    has_exit_eos = len(exit_runs) > 0

    # ── determine t* (representative) ─────────────────────────────────────── #
    # Use the minimum (earliest) per-run t* across all runs (EoS + forks).
    if run_data:
        t_star_repr = float(min(rd["t_star_x"] for rd in run_data))
    else:
        t_star_repr = 0.0
    print(f"\nRepresentative t* = {t_star_repr:.2f}")

    # ── determine t** (representative, from exit-EoS runs) ────────────────── #
    t_second_repr = None
    if exit_runs:
        t2_vals = []
        for rd in exit_runs:
            t2 = find_t_second(rd)
            if t2 is not None:
                t2_vals.append(t2)
                print(f"  t** for {rd['name']} = {t2:.2f}")
        if t2_vals:
            t_second_repr = float(np.median(t2_vals))
            print(f"Representative t** = {t_second_repr:.2f}")

    # ── determine x_upper ─────────────────────────────────────────────────── #
    all_t_maxes = [float(np.nanmax(rd["x_vals"])) for rd in run_data]
    if args.xlim is not None:
        x_upper = args.xlim
    elif t_second_repr is not None:
        x_upper = t_second_repr + 10.0
    else:
        x_upper = t_star_repr + 500.0
    x_upper = min(x_upper, min(all_t_maxes))
    print(f"x_upper = {x_upper:.2f}")

    # ── override t*/t** if given on CLI ───────────────────────────────────── #
    if args.t_star is not None:
        t_star_repr = args.t_star
    if args.t_second is not None:
        t_second_repr = args.t_second

    # ── select prototype groups ─────────────────────────────────────────────── #
    if args.boundary_input_only:
        proto_groups = [g for g in PROTOTYPE_GROUPS
                        if g[1] in ("boundary", "input-outlier")]
    else:
        proto_groups = PROTOTYPE_GROUPS

    # ── build common t_eff grid ────────────────────────────────────────────── #
    x_min = args.xmin
    smooth_lmax = args.smooth_lmax if args.smooth_lmax is not None else smooth
    n_grid = args.n_grid
    grid = np.linspace(x_min, x_upper, n_grid)

    # ── figure layout: 1×3 default (sharpness / loss / differential);
    #    with --with-curv, add cos²·‖g‖² as 4th panel ───────────────────── #
    if args.with_curv:
        fig, (ax_lmax, ax_loss, ax_cos2grad, ax_diff) = plt.subplots(
            1, 4, figsize=(14.0, 3.0),
        )
        fig.subplots_adjust(wspace=0.42, left=0.05, right=0.94,
                            bottom=0.18, top=0.80)
    else:
        fig, (ax_lmax, ax_loss, ax_diff) = plt.subplots(
            1, 3, figsize=(10.5, 3.0),
        )
        fig.subplots_adjust(wspace=0.42, left=0.06, right=0.93,
                            bottom=0.18, top=0.80)
        ax_cos2grad = None
    ax_cos2 = None

    xlabel = (
        r"effective time since $t^*$  ($t_{\rm eff} - t^*_{\rm eff}$)"
        if mode == "post"
        else r"effective time  ($t_{\rm eff} = \eta\,t$)"
    )

    # ── helper: plot median + IQR band ────────────────────────────────────── #
    def plot_band(ax, grid, median, p25, p75, color, ls, lw=1.5,
                  alpha_line=0.9, alpha_band=0.2):
        mask = ~np.isnan(median)
        if not mask.any():
            return
        xs = grid[mask]
        ax.plot(xs, median[mask], linestyle=ls, color=color,
                linewidth=lw, alpha=alpha_line)
        ax.fill_between(xs, p25[mask], p75[mask], color=color,
                        alpha=alpha_band, linewidth=0)

    # ── helper: plot individual seed traces ───────────────────────────────── #
    def plot_traces(ax, grid, curves, color, ls, lw=0.5, alpha=0.2):
        for curve in curves:
            mask = ~np.isnan(curve)
            if not mask.any():
                continue
            ax.plot(grid[mask], curve[mask], linestyle=ls, color=color,
                    linewidth=lw, alpha=alpha)

    ls_eos  = "-"
    ls_exit = (0, (5, 4))

    # ── prototype loss panel: median (+ optional IQR) ────────────────────── #
    for suffix, label, color in proto_groups:
        key = ploss_key(suffix)
        if eos_runs:
            med, p25, p75, _ = aggregate_seeds(eos_runs, key, grid, smooth=smooth)
            if args.shade_iqr:
                plot_band(ax_loss, grid, med, p25, p75, color, ls_eos)
            else:
                m = ~np.isnan(med)
                if m.any():
                    ax_loss.plot(grid[m], med[m], linestyle=ls_eos, color=color,
                                 linewidth=1.5, alpha=0.95)
        if exit_runs:
            med, p25, p75, _ = aggregate_seeds(exit_runs, key, grid, smooth=smooth)
            if args.shade_iqr:
                plot_band(ax_loss, grid, med, p25, p75, color, ls_exit)
            else:
                m = ~np.isnan(med)
                if m.any():
                    ax_loss.plot(grid[m], med[m], linestyle=ls_exit, color=color,
                                 linewidth=1.5, alpha=0.95)

    # ── cos^2 panel: baseline only, median + IQR ─────────────────────────── #
    if ax_cos2 is not None:
        for suffix, label, color in proto_groups:
            ckey = pkey(suffix, "grad_vmax_cos2")
            if eos_runs:
                med, p25, p75, _ = aggregate_seeds(eos_runs, ckey, grid, smooth=smooth)
                plot_band(ax_cos2, grid, med, p25, p75, color, ls_eos)
        ax_cos2.set_ylim(-0.05, 1.05)

    # ── lambda_max panel: median + IQR band (uses dense λ_max if available) ── #
    def _aggregate_lmax(run_list):
        curves = []
        for rd in run_list:
            if "lmax_dense" in rd["hist"]:
                dense_t_eff = compute_t_eff(rd["hist"]["lmax_dense_step"], rd["meta"])
                xs = (dense_t_eff - rd["t_star_eff"]) if mode == "post" else dense_t_eff
                ys = rd["hist"]["lmax_dense"]
            else:
                xs = rd["x_vals"]
                ys = rd["hist"].get(LMAX_KEY, np.full(len(xs), np.nan))
            curves.append(interpolate_to_grid(xs, ys, grid, smooth=smooth_lmax))
        nan_arr = np.full(len(grid), np.nan)
        if not curves:
            return nan_arr, nan_arr, nan_arr
        arr = np.vstack(curves)
        with np.errstate(all="ignore"):
            m = np.nanmedian(arr, axis=0)
            p25 = np.nanpercentile(arr, 25, axis=0)
            p75 = np.nanpercentile(arr, 75, axis=0)
        n_valid = np.sum(~np.isnan(arr), axis=0)
        m[n_valid == 0] = np.nan
        p25[n_valid == 0] = np.nan
        p75[n_valid == 0] = np.nan
        return m, p25, p75

    if eos_runs:
        med, p25, p75 = _aggregate_lmax(eos_runs)
        plot_band(ax_lmax, grid, med, p25, p75, _BLACK, ls_eos)

    if exit_runs:
        med, p25, p75 = _aggregate_lmax(exit_runs)
        plot_band(ax_lmax, grid, med, p25, p75, _BLACK, ls_exit)

    # ── threshold lines on lmax panel with right-side labels ─────────────── #
    all_lr_high = set()
    all_lr_low  = set()
    for rd in run_data:
        all_lr_high.add(rd["meta"]["lr_high"])
        if rd["meta"]["lr_low"] is not None:
            all_lr_low.add(rd["meta"]["lr_low"])

    for lr_val in sorted(all_lr_high):
        y_val = 2.0 / lr_val
        ax_lmax.axhline(y_val, linestyle=(0, (4, 3)), color=_GRAY,
                        linewidth=0.8, alpha=0.7, zorder=0)
        ax_lmax.annotate(
            r"$2/\eta$",
            xy=(1.0, y_val), xycoords=("axes fraction", "data"),
            xytext=(5, 0), textcoords="offset points",
            ha="left", va="center", fontsize=9, color=_GRAY,
            clip_on=False,
        )

    # Collect drop_mults from exit runs so horizontal label matches the drop
    drop_mults = set()
    for rd in exit_runs:
        dm = rd["meta"].get("drop_mult")
        if dm is not None:
            drop_mults.add(float(dm))

    for lr_val in sorted(all_lr_low):
        y_val = 2.0 / lr_val
        ax_lmax.axhline(y_val, linestyle=(0, (4, 3)), color=_GRAY,
                        linewidth=0.8, alpha=0.7, zorder=0)
        # match this lr_low to its drop_mult to build the "(2/drop)/eta" label
        label_txt = r"$2/\eta_{\rm low}$"
        for dm in drop_mults:
            for rd in exit_runs:
                if (rd["meta"].get("lr_low") == lr_val
                        and rd["meta"].get("drop_mult") == dm):
                    mult = 2.0 / dm
                    label_txt = rf"${mult:.3g}/\eta$"
                    break
            else:
                continue
            break
        ax_lmax.annotate(
            label_txt,
            xy=(1.0, y_val), xycoords=("axes fraction", "data"),
            xytext=(5, 0), textcoords="offset points",
            ha="left", va="center", fontsize=9, color=_GRAY,
            clip_on=False,
        )

    # ── vlines and span ───────────────────────────────────────────────────── #
    tstar_x   = 0.0 if mode == "post" else t_star_repr
    tsecond_x = None
    if t_second_repr is not None:
        tsecond_x = (t_second_repr - t_star_repr) if mode == "post" else t_second_repr

    vline_axes = [ax_loss, ax_lmax] + ([ax_cos2] if ax_cos2 is not None else [])
    if ax_diff is not None:
        vline_axes.append(ax_diff)
    if ax_cos2grad is not None:
        vline_axes.append(ax_cos2grad)
    draw_vlines_span(vline_axes, tstar_x, tsecond_x, label_ax=ax_lmax)

    # ── cos²·‖g‖² panel: baseline + fork, all four groups ─────────────────── #
    if ax_cos2grad is not None:
        for suffix, label, color in PROTOTYPE_GROUPS:
            ckey = pkey(suffix, "grad_vmax_cos2")
            gkey = pkey(suffix, "grad_norm")
            def _median_proj(run_list):
                curves = []
                for rd in run_list:
                    xs = rd["x_vals"]
                    cos2 = rd["hist"].get(ckey, np.full(len(xs), np.nan))
                    g = rd["hist"].get(gkey, np.full(len(xs), np.nan))
                    proj = cos2 * (g ** 2)
                    curves.append(interpolate_to_grid(xs, proj, grid, smooth=smooth))
                if not curves:
                    return None
                arr = np.vstack(curves)
                with np.errstate(all="ignore"):
                    m = np.nanmedian(arr, axis=0)
                n_valid = np.sum(~np.isnan(arr), axis=0)
                m[n_valid == 0] = np.nan
                return m
            if eos_runs:
                med = _median_proj(eos_runs)
                if med is not None:
                    mask = ~np.isnan(med)
                    if mask.any():
                        ax_cos2grad.plot(grid[mask], med[mask], linestyle=ls_eos,
                                         color=color, linewidth=1.5, alpha=0.95)
            if exit_runs:
                med = _median_proj(exit_runs)
                if med is not None:
                    mask = ~np.isnan(med)
                    if mask.any():
                        ax_cos2grad.plot(grid[mask], med[mask], linestyle=ls_exit,
                                         color=color, linewidth=1.5, alpha=0.95)

    # ── differential panel: median + IQR band, all four groups ──────────── #
    diff_median_min = np.inf
    diff_median_max = -np.inf
    if ax_diff is not None:
        for suffix, label, color in PROTOTYPE_GROUPS:
            key = ploss_key(suffix)
            per_seed_curves = []
            for (cls, seed), slot in class_seed_slots.items():
                b = slot.get("baseline")
                f = slot.get("fork")
                if b is None or f is None:
                    continue
                b_curve = interpolate_to_grid(
                    b["x_vals"],
                    b["hist"].get(key, np.full(len(b["x_vals"]), np.nan)),
                    grid, smooth=smooth)
                f_curve = interpolate_to_grid(
                    f["x_vals"],
                    f["hist"].get(key, np.full(len(f["x_vals"]), np.nan)),
                    grid, smooth=smooth)
                t_fork = f["t_star_eff"]
                diff = b_curve - f_curve
                diff[grid < t_fork] = 0.0
                per_seed_curves.append(diff)
            if not per_seed_curves:
                continue
            arr = np.vstack(per_seed_curves)
            with np.errstate(all="ignore"):
                med = np.nanmedian(arr, axis=0)
                p25 = np.nanpercentile(arr, 25, axis=0)
                p75 = np.nanpercentile(arr, 75, axis=0)
            n_valid = np.sum(~np.isnan(arr), axis=0)
            med[n_valid == 0] = np.nan
            p25[n_valid == 0] = np.nan
            p75[n_valid == 0] = np.nan
            valid = ~np.isnan(med)
            if valid.any():
                diff_median_min = min(diff_median_min, float(np.nanmin(med[valid])))
                diff_median_max = max(diff_median_max, float(np.nanmax(med[valid])))
            plot_band(ax_diff, grid, med, p25, p75, color, "-")
        ax_diff.axhline(0.0, color=_GRAY, linewidth=0.7, alpha=0.6, zorder=0)

    # ── finalize loss axis (no title, white background, no grid) ─────────── #
    ax_loss.set_yscale("log")
    ax_loss.set_xlim(left=x_min, right=x_upper)
    ax_loss.set_ylabel("train loss")
    ax_loss.set_xlabel(xlabel)
    ax_loss.text(0.5, 1.11, "prototype loss", transform=ax_loss.transAxes,
                 fontsize=10, va="baseline", ha="center")
    ax_loss.set_facecolor("white")
    ax_loss.grid(False)

    if ax_diff is not None:
        ax_diff.set_xlim(left=x_min, right=x_upper)
        ax_diff.set_xlabel(xlabel)
        ax_diff.set_ylabel(r"$\Delta \ell_k$")
        ax_diff.text(0.5, 1.11, "intervention effect",
                     transform=ax_diff.transAxes,
                     fontsize=10, va="baseline", ha="center")
        ax_diff.set_facecolor("white")
        ax_diff.grid(False)
        if tstar_x is not None and tsecond_x is not None and tsecond_x > tstar_x:
            mid_x = 0.5 * (tstar_x + tsecond_x)
            if diff_median_max > 0:
                ax_diff.annotate(
                    "EoS hurts",
                    xy=(mid_x, 0.9), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=8, color="#555555",
                    clip_on=False,
                )
            if diff_median_min < 0:
                ax_diff.annotate(
                    "EoS helps",
                    xy=(mid_x, 0.04), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=8, color="#555555",
                    clip_on=False,
                )

    if ax_cos2grad is not None:
        ax_cos2grad.set_xlim(left=x_min, right=x_upper)
        ax_cos2grad.set_yscale("log")
        ax_cos2grad.set_xlabel(xlabel)
        ax_cos2grad.set_ylabel(r"$\cos^2(\nabla f_k, v_{\max})\,\|\nabla f_k\|^2$")
        ax_cos2grad.text(0.5, 1.11, r"top-eigendir. grad energy",
                         transform=ax_cos2grad.transAxes,
                         fontsize=10, va="baseline", ha="center")
        ax_cos2grad.set_facecolor("white")
        ax_cos2grad.grid(False)

    # ── finalize lmax axis (no title, white background, no grid) ─────────── #
    ax_lmax.set_xlim(left=x_min, right=x_upper)
    ax_lmax.set_ylabel("sharpness")
    ax_lmax.set_xlabel(xlabel)
    ax_lmax.text(0.5, 1.11, "sharpness", transform=ax_lmax.transAxes,
                 fontsize=10, va="baseline", ha="center")
    ax_lmax.set_facecolor("white")
    ax_lmax.grid(False)

    # ── finalize cos^2 axis ──────────────────────────────────────────────── #
    if ax_cos2 is not None:
        ax_cos2.set_xlim(left=x_min, right=x_upper)
        ax_cos2.set_ylabel(r"$\cos^2(\nabla f_k, v_{\max})$")
        ax_cos2.set_xlabel(xlabel)
        ax_cos2.set_facecolor("white")
        ax_cos2.grid(False)


    # ── top horizontal legend ────────────────────────────────────────────── #
    proto_by_label = {lbl: c for _, lbl, c in proto_groups}
    combined_handles = []
    combined_handles.append(
        mlines.Line2D([], [], linestyle="-", color=_BLACK,
                      linewidth=1.5, label=r"EoS ($\eta$ fixed)"))
    if has_exit_eos:
        dm_for_label = None
        for rd in exit_runs:
            dm = rd["meta"].get("drop_mult")
            if dm is not None:
                dm_for_label = float(dm); break
        if dm_for_label is not None and dm_for_label > 0:
            inv = 1.0 / dm_for_label
            fork_label = rf"exit EoS ($\eta\!\to\!\eta/{inv:.3g}$)"
        else:
            fork_label = r"exit EoS ($\eta\!\to\!\eta_{\rm low}$)"
        combined_handles.append(
            mlines.Line2D([], [], linestyle=(0, (5, 4)), color=_BLACK,
                          linewidth=1.5, label=fork_label))
    for lbl in ["inlier", "boundary", "output-outlier", "input-outlier"]:
        if lbl in proto_by_label:
            combined_handles.append(
                mlines.Line2D([], [], color=proto_by_label[lbl], linewidth=1.5,
                              solid_capstyle="round", label=lbl))
    leg = fig.legend(
        handles=combined_handles,
        loc="upper center", bbox_to_anchor=(0.495, 1.04),
        frameon=True, framealpha=0.95, edgecolor="0.8",
        fontsize=8,
        ncol=len(combined_handles), borderpad=0.4,
        handlelength=2.5, handletextpad=0.5, columnspacing=1.2,
        fancybox=False,
    )
    leg.get_frame().set_linewidth(0.5)

    # ── save ──────────────────────────────────────────────────────────────── #
    slug = args.project.split("/")[-1]
    # Project slug pattern: <prefix>-<optimizer>-<model>-<loss>-<dataset>
    # e.g. fork-eoss-v3proto-full_gd-mlp-mse-cifar10_2cls
    parts = slug.split("-")
    if len(parts) >= 4:
        optimizer, model, loss = parts[-4], parts[-3], parts[-2]
    else:
        optimizer, model, loss = "opt", "model", "loss"
    n_seeds = len({rd["meta"]["init_seed"] for rd in run_data
                   if rd["meta"].get("init_seed") is not None})
    cls_set = {rd["meta"].get("classes") for rd in run_data
               if rd["meta"].get("classes") is not None}
    if len(cls_set) == 1:
        c = next(iter(cls_set))
        cls_tag = f"-cls{c[0]}v{c[1]}"
    elif len(cls_set) > 1:
        cls_tag = "-" + "_".join(f"cls{c[0]}v{c[1]}" for c in sorted(cls_set))
    else:
        cls_tag = ""
    out = f"{optimizer}-{model}-{loss}-{n_seeds}seeds-multiseed{cls_tag}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.06)
    print(f"\nSaved → {out}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-seed Figure: mean ± std across seeds for EoS / exit-EoS"
    )
    parser.add_argument("--project", required=True,
                        help="W&B project path (entity/project-name)")
    parser.add_argument("--t-star",  type=float, default=None,
                        help="manually override t*_eff (in effective time units)")
    parser.add_argument("--t-second", type=float, default=None,
                        help="manually override t**_eff (in effective time units)")
    parser.add_argument("--xlim",   type=float, default=None,
                        help="upper x-axis limit (t_eff)")
    parser.add_argument("--smooth",  type=int,   default=1,
                        help="rolling-mean window (default 1 = no smoothing)")
    parser.add_argument("--smooth-lmax", type=int, default=None,
                        help="separate smoothing window for lambda_max panel (default: same as --smooth)")
    parser.add_argument("--xmin", type=float, default=0.0,
                        help="lower x-axis limit (t_eff, default: 0)")
    parser.add_argument("--t-star-override", type=float, default=None,
                        help="override t* vertical line position (t_eff units)")
    parser.add_argument("--mode",    choices=["full", "post"], default="full",
                        help="x-axis: 'full' = absolute t_eff, 'post' = relative to t*")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Class pairs to include, e.g. '1,9' '3,5'")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Seeds to include (e.g. 1111 1234 3333)")
    parser.add_argument("--exclude-run-ids", nargs="+", type=str, default=None,
                        help="W&B run IDs to exclude")
    parser.add_argument("--boundary-input-only", action="store_true", default=False,
                        help="Only plot boundary and input-outlier prototypes")
    parser.add_argument("--shade-iqr", action="store_true", default=False,
                        help="Shade IQR band on prototype loss panel")
    parser.add_argument("--with-curv", action="store_true", default=False,
                        help="Add third column with per-prototype ||grad||^2 * cos^2")
    parser.add_argument("--n-grid", type=int, default=3000,
                        help="Number of grid points for interpolation (default 3000)")
    parser.add_argument("--drop-filter", type=float, default=None,
                        help="Only include fork runs whose drop_mult equals this value")
    parser.add_argument("--tss-mult", type=float, default=20.0,
                        help="t** threshold multiplier: lambda_max >= mult/eta_high")
    parser.add_argument("--tss-eta-low", action="store_true",
                        help="Define t** as crossing of 2/eta_low instead of tss_mult/eta_high")
    args = parser.parse_args()

    # Parse class pairs
    if args.classes:
        parsed = []
        for s in args.classes:
            parts = s.split(",")
            parsed.append(tuple(int(x) for x in parts))
        args.classes = parsed

    main(args)
