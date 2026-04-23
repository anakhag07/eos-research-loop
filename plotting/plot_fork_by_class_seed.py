"""
Per-class, per-seed comparison: baseline vs forked run.

Layout: one page per class pair, rows = seeds, cols = metrics
  Col 0: Full Loss (log scale)
  Col 1: Lambda_max
  Col 2: Prototype Loss (all 4 groups)
  Col 3: cos²(grad, v_max) per prototype

Baseline = solid, fork = dashed.  t* marked with vertical line.

Usage:
    OPENBLAS_NUM_THREADS=4 conda run -n eoss python plot_fork_by_class_seed.py \
        --project shaunakwag-massachusetts-institute-of-technology/fork-eoss-fork-full_gd-mlp-mse-cifar10_2cls \
        --smooth 3

    # Restrict to specific class pairs:
    OPENBLAS_NUM_THREADS=4 conda run -n eoss python plot_fork_by_class_seed.py \
        --project shaunakwag-massachusetts-institute-of-technology/fork-eoss-fork-full_gd-mlp-mse-cifar10_2cls \
        --classes 1,9 3,5
"""

import argparse
import json
import os
import pickle
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches
import pandas as pd
import wandb

# ── prototype registry (Okabe-Ito) ────────────────────────────────────────
PROTOTYPE_GROUPS = [
    ("injected_inliers/injected_inliers",     "inlier",         "#E69F00"),
    ("injected_y_outlier/injected_y_outlier", "output-outlier", "#009E73"),
    ("injected_x_outlier/injected_x_outlier", "input-outlier",  "#0072B2"),
    ("injected_boundary/injected_boundary",   "boundary",       "#D55E00"),
]

_BLACK = "#111111"
_GRAY  = "#999999"
PREFIX = "input_space_prototypes"

def pkey(suffix, metric):
    return f"{PREFIX}/{suffix}/{metric}"

def ploss_key(s):
    return pkey(s, "full_loss")


def _apply_rcparams():
    matplotlib.rcParams.update({
        "font.family":           "sans-serif",
        "font.sans-serif":       ["DejaVu Sans"],
        "mathtext.fontset":      "dejavusans",
        "font.size":             8,
        "axes.titlesize":        9,
        "axes.labelsize":        8,
        "xtick.labelsize":       7,
        "ytick.labelsize":       7,
        "legend.fontsize":       6.5,
        "legend.title_fontsize": 7,
        "xtick.direction":       "in",
        "ytick.direction":       "in",
        "xtick.major.width":     0.5,
        "ytick.major.width":     0.5,
        "xtick.major.size":      3.0,
        "ytick.major.size":      3.0,
        "axes.linewidth":        0.5,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "legend.frameon":        False,
        "figure.autolayout":     False,
        "savefig.bbox":          "tight",
        "savefig.pad_inches":    0.02,
    })


# ── wandb helpers ──────────────────────────────────────────────────────────
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

    if lr_low is None and "lrdrop" in run.name.lower():
        m = re.search(r'lrdrop([0-9.eE+-]+)', run.name, re.IGNORECASE)
        if m:
            lr_low = lr * float(m.group(1))

    init_seed = _cfg_val(cfg.get("init_seed", cfg.get("seed", None)))
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
    fork_lr_high = None
    fork_step = None
    m_fork_lr = re.search(r'fork_exit_eos_seed\d+_lr([0-9.eE+-]+)_drop([0-9.eE+-]+)_from_step(\d+)', run.name)
    if m_fork_lr:
        fork_lr_high = float(m_fork_lr.group(1))
        fork_drop = float(m_fork_lr.group(2))
        fork_step = int(m_fork_lr.group(3))
        if lr_low is None:
            lr_low = fork_lr_high * fork_drop
            lr = fork_lr_high
            drop_mult = fork_drop
            t_star = fork_step

    return {
        "lr_high":    lr,
        "lr_low":     lr_low,
        "t_star":     t_star,
        "drop_mult":  drop_mult,
        "init_seed":  init_seed,
        "classes":    classes,
        "fork_step":  fork_step,
    }


def build_all_keys():
    keys = []
    for suffix, _, _ in PROTOTYPE_GROUPS:
        for metric in ["full_loss", "grad_vmax_cos2", "grad_norm"]:
            keys.append(pkey(suffix, metric))
    keys += ["lambda_max", "full_loss"]
    return keys


def get_history(run, keys):
    cache_dir = ".wandb_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{run.id}_fork_cls_v2.pkl")
    if os.path.exists(cache_path):
        print(f"  Loading from cache: {cache_path}")
        return pickle.load(open(cache_path, "rb"))
    print(f"  Fetching all rows via scan_history...")
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

    if "lambda_max" in keys:
        lmax_rows = list(run.scan_history(keys=["_step", "step", "lambda_max"]))
        if lmax_rows:
            lmax_df = pd.DataFrame(lmax_rows)
            step_col = "step" if "step" in lmax_df.columns and lmax_df["step"].notna().any() else "_step"
            if step_col in lmax_df.columns and "lambda_max" in lmax_df.columns:
                result["lmax_dense_step"] = lmax_df[step_col].values.astype(float)
                result["lmax_dense"] = lmax_df["lambda_max"].values

    pickle.dump(result, open(cache_path, "wb"))
    print(f"  Cached -> {cache_path}")
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
    if "lmax_dense" in hist and "lmax_dense_step" in hist:
        return compute_t_eff(hist["lmax_dense_step"], meta), hist["lmax_dense"]
    return fallback_t_eff, hist["lambda_max"]


def get_t_star_eff(meta, hist, t_eff):
    t_eff, lmax = _dense_lmax(hist, meta, t_eff)
    threshold = 2.0 / meta["lr_high"]
    valid = ~np.isnan(lmax)
    candidates = []

    # Interpolate crossing point between last-below and first-above measurements
    for i in range(1, len(lmax)):
        if valid[i] and valid[i - 1] and lmax[i - 1] < threshold <= lmax[i]:
            frac = (threshold - lmax[i - 1]) / (lmax[i] - lmax[i - 1])
            t_cross = t_eff[i - 1] + frac * (t_eff[i] - t_eff[i - 1])
            candidates.append(t_cross)
            break

    # Fallback: first measurement above threshold (no prior point to interpolate)
    if not candidates:
        above = np.where(valid & (lmax >= threshold))[0]
        if len(above):
            candidates.append(float(t_eff[above[0]]))

    if not candidates and meta["t_star"] is not None:
        candidates.append(meta["t_star"] * meta["lr_high"])

    return min(candidates) if candidates else 0.0


def get_t_star_star_eff(meta, hist, t_eff, tss_mult=20.0, tss_eta_low=False):
    """Find t** from the requested post-fork lambda_max threshold."""
    lr_high = meta.get("lr_high")
    lr_low = meta.get("lr_low")
    if lr_high is None or lr_high <= 0:
        return None

    if tss_eta_low:
        if lr_low is None or lr_low <= 0:
            return None
        threshold = 2.0 / lr_low
    else:
        threshold = float(tss_mult) / lr_high

    t_eff, lmax = _dense_lmax(hist, meta, t_eff)
    valid = ~np.isnan(lmax)

    # Interpolate crossing point between last-below and first-above measurements
    for i in range(1, len(lmax)):
        if valid[i] and valid[i - 1] and lmax[i - 1] < threshold <= lmax[i]:
            frac = (threshold - lmax[i - 1]) / (lmax[i] - lmax[i - 1])
            return float(t_eff[i - 1] + frac * (t_eff[i] - t_eff[i - 1]))

    # Fallback: first measurement above threshold
    above = np.where(valid & (lmax >= threshold))[0]
    if len(above):
        return float(t_eff[above[0]])
    return None


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


# ── main ───────────────────────────────────────────────────────────────────
def main(args):
    _apply_rcparams()
    api = wandb.Api()
    smooth = args.smooth
    all_keys = build_all_keys()

    # ── load runs ──────────────────────────────────────────────────────────
    print(f"\nFetching all runs from project: {args.project}")
    all_runs = [r for r in api.runs(args.project) if r.state != "deleted"]
    print(f"  Found {len(all_runs)} runs")

    # ── organize by (class_pair, seed) ─────────────────────────────────────
    # class_seed_data[(cls_tuple, seed)] = {"baseline": rd, "fork": rd}
    class_seed_data = {}

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
            print(f"  SKIP: no lr")
            continue

        classes = meta["classes"]
        seed = meta["init_seed"]
        if classes is None or seed is None:
            print(f"  SKIP: classes={classes}, seed={seed}")
            continue

        # Filter by excluded seeds
        if args.exclude_seeds and seed in args.exclude_seeds:
            print(f"  SKIP: seed {seed} excluded")
            continue

        # Filter by requested class pairs if specified
        if args.classes:
            if classes not in args.classes:
                print(f"  SKIP: classes {classes} not in requested {args.classes}")
                continue

        hist = get_history(run, all_keys)
        if hist is None:
            print(f"  SKIP: no history")
            continue

        # Determine if baseline or fork
        is_fork = (meta["fork_step"] is not None or
                   "fork_exit" in run.name.lower() or
                   "lrdrop" in run.name.lower())

        # Filter fork runs by drop_mult if requested
        if is_fork and args.drop_filter is not None:
            dm = meta.get("drop_mult")
            if dm is None or not np.isclose(float(dm), args.drop_filter, atol=1e-6):
                print(f"  SKIP fork: drop_mult={dm} != {args.drop_filter}")
                continue

        t_eff = compute_t_eff(hist["step"], meta)
        t_star_eff = get_t_star_eff(meta, hist, t_eff)
        if args.t_star is not None and not is_fork:
            t_star_eff = float(args.t_star)
        t_star_star_eff = None
        if is_fork:
            t_star_star_eff = get_t_star_star_eff(
                meta,
                hist,
                t_eff,
                tss_mult=args.tss_mult,
                tss_eta_low=args.tss_eta_low,
            )

        rd = {
            "name":            run.name,
            "meta":            meta,
            "hist":            hist,
            "t_eff":           t_eff,
            "t_star_eff":      t_star_eff,
            "t_star_star_eff": t_star_star_eff,
            "is_fork":         is_fork,
        }

        key = (classes, seed)
        if key not in class_seed_data:
            class_seed_data[key] = {"baseline": None, "fork": None}

        if is_fork:
            class_seed_data[key]["fork"] = rd
        else:
            class_seed_data[key]["baseline"] = rd

        tss = f"{t_star_star_eff:.2f}" if t_star_star_eff is not None else "N/A"
        print(f"  classes={classes}  seed={seed}  fork={is_fork}  "
              f"lr={meta['lr_high']}  lr_low={meta['lr_low']}  t*_eff={t_star_eff:.2f}  t**_eff={tss}")

    # ── group by class pair ────────────────────────────────────────────────
    class_pairs = sorted(set(cls for cls, _ in class_seed_data.keys()))
    print(f"\nClass pairs found: {class_pairs}")

    for cls_pair in class_pairs:
        seeds_for_cls = sorted(
            seed for (cls, seed) in class_seed_data.keys() if cls == cls_pair
        )
        n_seeds = len(seeds_for_cls)
        cls_label = f"cls {cls_pair[0]} vs {cls_pair[1]}"
        print(f"\n{'#'*60}")
        print(f"  {cls_label}: {n_seeds} seeds = {seeds_for_cls}")

        if n_seeds == 0:
            continue

        # ── compute t** per seed, then global xlim = max(t**) + 20 ────────
        MARGIN = 20.0
        t_star_effs = {}
        t_star_star_effs = {}
        for seed in seeds_for_cls:
            sd = class_seed_data[(cls_pair, seed)]
            base_rd = sd["baseline"]
            fork_rd = sd["fork"]
            # Use actual EoS crossing (lambda_max >= 2/eta); prefer the
            # smaller of baseline/fork measurements (finest crossing).
            candidates = []
            if fork_rd:
                candidates.append(fork_rd["t_star_eff"])
            if base_rd:
                candidates.append(base_rd["t_star_eff"])
            ts = min(candidates) if candidates else 0.0
            t_star_effs[seed] = ts
            tss = fork_rd["t_star_star_eff"] if fork_rd else None
            t_star_star_effs[seed] = tss

        # xlim: use --xlim if given, else max(t**) + MARGIN across seeds
        valid_tss = [v for v in t_star_star_effs.values() if v is not None]
        if args.xlim is not None:
            x_right = args.xlim
        elif valid_tss:
            x_right = max(valid_tss) + MARGIN
        else:
            x_right = max(t_star_effs.values()) + MARGIN

        print(f"  t* per seed:  {t_star_effs}")
        print(f"  t** per seed: {t_star_star_effs}")
        print(f"  xlim = {x_right:.1f}")

        # ── figure layout: n_seeds rows x 6 cols ──────────────────────────
        n_cols = 6
        fig, axes = plt.subplots(
            n_seeds, n_cols,
            figsize=(3.5 * n_cols, 2.8 * n_seeds),
            sharex="col",
            gridspec_kw={"hspace": 0.35, "wspace": 0.30},
            squeeze=False,
        )

        ls_base = "-"
        ls_fork = (0, (5, 4))
        _SHADE = "#C4B0D8"  # purple for [t*, t**] shading

        for row, seed in enumerate(seeds_for_cls):
            sd = class_seed_data[(cls_pair, seed)]
            ax_floss = axes[row, 0]
            ax_lmax  = axes[row, 1]
            ax_ploss = axes[row, 2]
            ax_cos2  = axes[row, 3]
            ax_gnorm = axes[row, 4]
            ax_curv  = axes[row, 5]

            ax_floss.set_ylabel(f"seed {seed}", fontsize=9, fontweight="bold")

            all_axes = [ax_floss, ax_lmax, ax_ploss, ax_cos2, ax_gnorm, ax_curv]

            ploss_vals = []  # collect for ylim zoom
            floss_vals = []  # collect for ylim zoom
            curv_vals  = []  # collect for ylim zoom

            for condition, ls in [("baseline", ls_base), ("fork", ls_fork)]:
                rd = sd[condition]
                if rd is None:
                    continue

                t_eff = rd["t_eff"]
                hist = rd["hist"]
                in_win_full = (t_eff >= 0) & (t_eff <= x_right)

                # Full loss
                floss = hist.get("full_loss", np.full(len(t_eff), np.nan))
                if smooth > 1:
                    floss = rolling_mean(floss, smooth)
                ax_floss.plot(t_eff, floss, linestyle=ls, color=_BLACK,
                              linewidth=1.0, alpha=0.85)
                fw = floss[in_win_full]
                fw = fw[np.isfinite(fw) & (fw > 0)]
                if len(fw):
                    floss_vals.append(fw)

                # Lambda max (dense if available)
                if "lmax_dense" in hist:
                    lmax_t = compute_t_eff(hist["lmax_dense_step"], rd["meta"])
                    lmax = hist["lmax_dense"]
                else:
                    lmax_t = t_eff
                    lmax = hist.get("lambda_max", np.full(len(t_eff), np.nan))
                ax_lmax.plot(lmax_t, lmax, linestyle=ls, color=_BLACK,
                             linewidth=1.0, alpha=0.85)

                # Prototype losses
                in_win = (t_eff >= 0) & (t_eff <= x_right)
                for suffix, plabel, color in PROTOTYPE_GROUPS:
                    key = ploss_key(suffix)
                    y = hist.get(key, np.full(len(t_eff), np.nan))
                    if smooth > 1:
                        y = rolling_mean(y, smooth)
                    ax_ploss.plot(t_eff, y, linestyle=ls, color=color,
                                 linewidth=1.0, alpha=0.85)
                    yw = y[in_win]
                    yw = yw[np.isfinite(yw) & (yw > 0)]
                    if len(yw):
                        ploss_vals.append(yw)

                # cos²(grad, v_max) per prototype
                for suffix, plabel, color in PROTOTYPE_GROUPS:
                    key = pkey(suffix, "grad_vmax_cos2")
                    y = hist.get(key, np.full(len(t_eff), np.nan))
                    if smooth > 1:
                        y = rolling_mean(y, smooth)
                    ax_cos2.plot(t_eff, y, linestyle=ls, color=color,
                                linewidth=1.0, alpha=0.85)

                # grad_norm per prototype
                for suffix, plabel, color in PROTOTYPE_GROUPS:
                    key = pkey(suffix, "grad_norm")
                    y = hist.get(key, np.full(len(t_eff), np.nan))
                    if smooth > 1:
                        y = rolling_mean(y, smooth)
                    ax_gnorm.plot(t_eff, y, linestyle=ls, color=color,
                                 linewidth=1.0, alpha=0.85)

                # ||grad||² * cos² (effective curvature contribution)
                for suffix, plabel, color in PROTOTYPE_GROUPS:
                    gn = hist.get(pkey(suffix, "grad_norm"), np.full(len(t_eff), np.nan))
                    c2 = hist.get(pkey(suffix, "grad_vmax_cos2"), np.full(len(t_eff), np.nan))
                    curv = (gn ** 2) * c2
                    if smooth > 1:
                        curv = rolling_mean(curv, smooth)
                    ax_curv.plot(t_eff, curv, linestyle=ls, color=color,
                                linewidth=1.0, alpha=0.85)
                    cw = curv[in_win_full]
                    cw = cw[np.isfinite(cw) & (cw > 0)]
                    if len(cw):
                        curv_vals.append(cw)

            # ── t* / t** markers and shading ──────────────────────────────
            ts = t_star_effs[seed]
            tss = t_star_star_effs[seed]

            base_rd = sd["baseline"]
            fork_rd = sd["fork"]

            # t* vertical line
            for ax in all_axes:
                ax.axvline(ts, color=_GRAY, lw=0.5, ls="--", alpha=0.6)

            # t** vertical line
            if tss is not None:
                for ax in all_axes:
                    ax.axvline(tss, color=_GRAY, lw=0.5, ls=":", alpha=0.6)

            # shade [t*, t**]
            if tss is not None:
                for ax in all_axes:
                    ax.axvspan(ts, tss, alpha=0.30, color=_SHADE, zorder=0)

            # 2/eta thresholds on lambda_max
            if base_rd is not None:
                lr_high = base_rd["meta"]["lr_high"]
                ax_lmax.axhline(2.0 / lr_high, ls="--", color=_BLACK, lw=0.7, alpha=0.8,
                                label=r"$2/\eta_{\rm high}$" if row == 0 else None)
            # t** threshold horizontal line follows the selected CLI rule.
            ref_rd = fork_rd if fork_rd is not None else base_rd
            if ref_rd is not None and ref_rd["meta"].get("lr_high"):
                if args.tss_eta_low:
                    lr_low = ref_rd["meta"].get("lr_low")
                    if lr_low is None or not np.isfinite(lr_low) or lr_low <= 0:
                        thresh = None
                    else:
                        thresh = 2.0 / lr_low
                    dm = ref_rd["meta"].get("drop_mult")
                    if dm:
                        mult = 2.0 / float(dm)
                        label = rf"${mult:.3g}/\eta_{{\rm high}}$"
                    else:
                        label = r"$2/\eta_{\rm low}$"
                else:
                    thresh = float(args.tss_mult) / ref_rd["meta"]["lr_high"]
                    label = rf"${args.tss_mult:.3g}/\eta_{{\rm high}}$"

                if thresh is not None:
                    ax_lmax.axhline(thresh, ls=":", color=_BLACK, lw=0.7, alpha=0.6,
                                    label=label if row == 0 else None)

            # ── styling ───────────────────────────────────────────────────
            ax_floss.set_yscale("log")
            if floss_vals:
                allf = np.concatenate(floss_vals)
                flo, fhi = np.min(allf), np.max(allf)
                ax_floss.set_ylim(flo / 1.5, fhi * 1.5)
            ax_ploss.set_yscale("log")
            if ploss_vals:
                allv = np.concatenate(ploss_vals)
                lo, hi = np.min(allv), np.max(allv)
                ax_ploss.set_ylim(lo / 1.5, hi * 1.5)
            ax_cos2.set_ylim(-0.05, 1.05)
            ax_gnorm.set_yscale("log")
            ax_curv.set_yscale("log")
            if curv_vals:
                allc = np.concatenate(curv_vals)
                clo, chi = np.min(allc), np.max(allc)
                ax_curv.set_ylim(clo / 1.5, chi * 1.5)

        # shared xlim across all subplots
        for row in range(n_seeds):
            for col in range(n_cols):
                axes[row, col].set_xlim(left=0, right=x_right)

        # Column titles
        axes[0, 0].set_title("Full Loss", fontsize=10, pad=8)
        axes[0, 1].set_title(r"$\lambda_{\max}$", fontsize=10, pad=8)
        axes[0, 2].set_title("Prototype Loss", fontsize=10, pad=8)
        axes[0, 3].set_title(r"$\cos^2(\nabla f_k, v_{\max})$", fontsize=10, pad=8)
        axes[0, 4].set_title(r"$\|\nabla f_k\|$", fontsize=10, pad=8)
        axes[0, 5].set_title(r"$\|\nabla f_k\|^2 \cdot \cos^2$", fontsize=10, pad=8)

        # x-axis label on bottom row
        for col in range(n_cols):
            axes[-1, col].set_xlabel(r"effective time  $t_{\rm eff} = \eta t$")

        # ── legend ─────────────────────────────────────────────────────────
        proto_handles = [
            mlines.Line2D([], [], color=c, lw=1.5, label=lbl)
            for _, lbl, c in PROTOTYPE_GROUPS
        ]
        cond_handles = [
            mlines.Line2D([], [], ls="-", color=_BLACK, lw=1.2, label="baseline"),
            mlines.Line2D([], [], ls=(0, (5, 4)), color=_BLACK, lw=1.2, label="fork (exit-EoS)"),
        ]
        shade_handle = [
            matplotlib.patches.Patch(facecolor=_SHADE, alpha=0.3, label=r"$[t^*, t^{**}]$"),
        ]
        fig.legend(
            handles=proto_handles + cond_handles + shade_handle,
            loc="upper center", ncol=7, fontsize=7,
            frameon=True, framealpha=0.9, edgecolor="0.85",
            bbox_to_anchor=(0.5, 1.0),
        )

        fig.suptitle(f"Baseline vs Fork  —  classes {cls_pair[0]} vs {cls_pair[1]}",
                     fontsize=12, fontweight="bold", y=1.03)

        # ── save ───────────────────────────────────────────────────────────
        # Parse optimizer/model/loss from project slug
        # e.g. fork-eoss-v3proto-full_gd-mlp-ce-cifar10_2cls
        slug_parts = args.project.split("/")[-1].split("-")
        optimizer = slug_parts[-4] if len(slug_parts) >= 4 else "opt"
        model     = slug_parts[-3] if len(slug_parts) >= 3 else "model"
        loss_name = slug_parts[-2] if len(slug_parts) >= 2 else "loss"
        cls_tag = f"cls{cls_pair[0]}v{cls_pair[1]}"
        out_dir = "/home/anakhag/projects/eos/plotting/plots"
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(
            out_dir,
            f"{optimizer}-{model}-{loss_name}-{cls_tag}_fig_fork_compare.pdf",
        )
        fig.savefig(out, bbox_inches="tight", pad_inches=0.08)
        print(f"\nSaved -> {out}")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-class, per-seed baseline vs fork comparison figure")
    parser.add_argument("--project", required=True,
                        help="W&B project path (entity/project-name)")
    parser.add_argument("--smooth", type=int, default=1,
                        help="Rolling-mean window (1 = no smoothing)")
    parser.add_argument("--xlim", type=float, default=None,
                        help="Upper x-axis limit (t_eff)")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Class pairs to include, e.g. '1,9' '3,5'")
    parser.add_argument("--exclude-seeds", type=int, nargs="+", default=None,
                        help="Seeds to exclude (e.g. 42)")
    parser.add_argument("--exclude-run-ids", type=str, nargs="+", default=None,
                        help="W&B run IDs to exclude")
    parser.add_argument("--tss-mult", type=float, default=20.0,
                        help="t** threshold multiplier: lambda_max >= mult/eta_high (default 20 for MSE, use 4 for CE)")
    parser.add_argument("--tss-eta-low", action="store_true",
                        help="Define t** as crossing of 2/eta_low (post-fork EoS bound) instead of tss_mult/eta_high")
    parser.add_argument("--t-star", type=float, default=None,
                        help="Override detected baseline t*_eff with this value")
    parser.add_argument("--drop-filter", type=float, default=None,
                        help="Only include fork runs whose drop_mult equals this value (e.g. 0.3)")
    args = parser.parse_args()

    # Parse class pairs
    if args.classes:
        parsed = []
        for s in args.classes:
            parts = s.split(",")
            parsed.append(tuple(int(x) for x in parts))
        args.classes = parsed

    main(args)
