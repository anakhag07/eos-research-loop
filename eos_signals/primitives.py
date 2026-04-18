"""Signal-extraction primitives.

Pure numerical functions over pandas DataFrames / numpy arrays.
Every function returns a JSON-serializable dict (python scalars only).
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd


def _as_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if np.isnan(v) or np.isinf(v):
        return None
    return v


def _clean_df(df: pd.DataFrame, step_col: str, metric_col: str) -> pd.DataFrame:
    cols = [step_col, metric_col]
    out = df[cols].copy()
    out[step_col] = pd.to_numeric(out[step_col], errors="coerce")
    out[metric_col] = pd.to_numeric(out[metric_col], errors="coerce")
    return out.dropna(subset=cols).sort_values(step_col).reset_index(drop=True)


def eos_threshold(eta: float, *, adam: bool = False) -> float:
    """EOS threshold: 38/eta for Adam-preconditioned lambda, 2/eta for SGD."""
    if eta is None or eta <= 0:
        raise ValueError(f"eta must be > 0, got {eta}")
    return (38.0 / eta) if adam else (2.0 / eta)


def crossing_step(
    df: pd.DataFrame,
    step_col: str,
    metric_col: str,
    threshold: float,
    *,
    direction: str = "up",
    start_step: Optional[float] = None,
    end_step: Optional[float] = None,
) -> Dict[str, object]:
    """First step at which `metric` crosses `threshold` within the window.

    direction:
      "up"   — first step where metric >= threshold
      "down" — first step where metric <= threshold
    Use start_step to skip early transient behavior (initialization spikes).
    Returns {step, value_at_step, threshold, direction, n, start_step, end_step}.
    step is None if no crossing within the window.
    """
    if direction not in ("up", "down"):
        raise ValueError("direction must be 'up' or 'down'")
    clean = _clean_df(df, step_col, metric_col)
    if start_step is not None:
        clean = clean[clean[step_col] >= start_step]
    if end_step is not None:
        clean = clean[clean[step_col] <= end_step]
    clean = clean.reset_index(drop=True)
    base = {
        "step": None, "value_at_step": None, "threshold": float(threshold),
        "direction": direction, "n": int(len(clean)),
        "start_step": _as_float(start_step), "end_step": _as_float(end_step),
    }
    if clean.empty:
        return base
    vals = clean[metric_col].to_numpy()
    mask = (vals >= threshold) if direction == "up" else (vals <= threshold)
    if not mask.any():
        return base
    idx = int(np.argmax(mask))
    base["step"] = int(clean[step_col].iloc[idx])
    base["value_at_step"] = _as_float(clean[metric_col].iloc[idx])
    return base


def eos_threshold_crossing(
    df: pd.DataFrame,
    step_col: str,
    metric_col: str,
    eta: float,
    *,
    adam: bool = False,
    start_step: Optional[float] = None,
    end_step: Optional[float] = None,
) -> Dict[str, object]:
    """First step at which `metric` crosses the EOS threshold for given eta.

    Returns crossing_step result plus {eta, adam, threshold_expr}.
    """
    thresh = eos_threshold(eta, adam=adam)
    out = crossing_step(df, step_col, metric_col, thresh, direction="up",
                        start_step=start_step, end_step=end_step)
    out["eta"] = float(eta)
    out["adam"] = bool(adam)
    out["threshold_expr"] = "38/eta" if adam else "2/eta"
    return out


def slope_in_window(
    df: pd.DataFrame,
    step_col: str,
    metric_col: str,
    start_step: Optional[float] = None,
    end_step: Optional[float] = None,
) -> Dict[str, object]:
    """Linear least-squares slope of metric vs step on [start_step, end_step].

    If start/end are None, uses min/max of the series.
    Returns {slope, intercept, r2, n, start_step, end_step}.
    """
    clean = _clean_df(df, step_col, metric_col)
    if start_step is not None:
        clean = clean[clean[step_col] >= start_step]
    if end_step is not None:
        clean = clean[clean[step_col] <= end_step]
    clean = clean.reset_index(drop=True)
    n = len(clean)
    if n < 2:
        return {
            "slope": None, "intercept": None, "r2": None, "n": int(n),
            "start_step": _as_float(start_step), "end_step": _as_float(end_step),
        }
    x = clean[step_col].to_numpy(dtype=float)
    y = clean[metric_col].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else None
    return {
        "slope": _as_float(slope),
        "intercept": _as_float(intercept),
        "r2": _as_float(r2),
        "n": int(n),
        "start_step": _as_float(x.min()),
        "end_step": _as_float(x.max()),
    }


def plateau_duration(
    df: pd.DataFrame,
    step_col: str,
    metric_col: str,
    tolerance: float,
    min_length: int = 5,
) -> Dict[str, object]:
    """Longest contiguous window where (max - min) of metric <= tolerance.

    Uses a forward two-pointer scan. Returns
    {start_step, end_step, length, mean, tolerance}. Null window if no run of
    length >= min_length satisfies the tolerance.
    """
    clean = _clean_df(df, step_col, metric_col)
    n = len(clean)
    if n == 0:
        return {"start_step": None, "end_step": None, "length": 0, "mean": None, "tolerance": tolerance}
    vals = clean[metric_col].to_numpy(dtype=float)
    steps = clean[step_col].to_numpy(dtype=float)

    best_len = 0
    best_lo = 0
    best_hi = 0
    lo = 0
    for hi in range(n):
        while vals[lo:hi + 1].max() - vals[lo:hi + 1].min() > tolerance:
            lo += 1
            if lo > hi:
                break
        cur_len = hi - lo + 1
        if cur_len > best_len:
            best_len = cur_len
            best_lo = lo
            best_hi = hi

    if best_len < min_length:
        return {"start_step": None, "end_step": None, "length": int(best_len), "mean": None, "tolerance": float(tolerance)}
    return {
        "start_step": _as_float(steps[best_lo]),
        "end_step": _as_float(steps[best_hi]),
        "length": int(best_len),
        "mean": _as_float(vals[best_lo:best_hi + 1].mean()),
        "tolerance": float(tolerance),
    }


def group_separation(
    groups: Mapping[str, pd.DataFrame],
    step_col: str,
    metric_col: str,
    start_step: Optional[float] = None,
    end_step: Optional[float] = None,
) -> Dict[str, object]:
    """Between-group vs within-group variance of a metric over an optional window.

    `groups` maps group_name -> DataFrame with at least step_col, metric_col.
    Returns {between_var, within_var, ratio, groups: {name: {n, mean}}}. A
    higher ratio means groups are more separated relative to within-group noise.
    """
    per_group: Dict[str, Dict[str, object]] = {}
    all_vals: list[float] = []
    weighted_within = 0.0
    total_n = 0
    group_means: list[tuple[int, float]] = []

    for name, df in groups.items():
        clean = _clean_df(df, step_col, metric_col)
        if start_step is not None:
            clean = clean[clean[step_col] >= start_step]
        if end_step is not None:
            clean = clean[clean[step_col] <= end_step]
        vals = clean[metric_col].to_numpy(dtype=float)
        n = int(len(vals))
        if n == 0:
            per_group[name] = {"n": 0, "mean": None, "var": None}
            continue
        mean = float(vals.mean())
        var = float(vals.var(ddof=0))
        per_group[name] = {"n": n, "mean": mean, "var": var}
        all_vals.extend(vals.tolist())
        weighted_within += var * n
        total_n += n
        group_means.append((n, mean))

    if total_n == 0 or len(group_means) < 2:
        return {"between_var": None, "within_var": None, "ratio": None, "groups": per_group}

    grand_mean = float(np.mean(all_vals))
    between = sum(n * (m - grand_mean) ** 2 for n, m in group_means) / total_n
    within = weighted_within / total_n
    ratio = (between / within) if within > 0 else None
    return {
        "between_var": _as_float(between),
        "within_var": _as_float(within),
        "ratio": _as_float(ratio),
        "groups": per_group,
    }


def stability_ratio_series(
    df: pd.DataFrame,
    step_col: str,
    grad_col: str,
    lambda_col: str,
    *,
    smooth_window: int = 1,
) -> pd.DataFrame:
    """Return a two-column DataFrame [step_col, 'stability_ratio'] = grad / lambda.

    Drops rows with non-positive lambda. Optional rolling-mean smoothing.
    Pure transform — no JSON output. Use `stability_ratio_summary` for scalars.
    """
    out = df[[step_col, grad_col, lambda_col]].copy()
    for c in (step_col, grad_col, lambda_col):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=[step_col, grad_col, lambda_col]).sort_values(step_col)
    out = out[out[lambda_col] > 0]
    if out.empty:
        return pd.DataFrame(columns=[step_col, "stability_ratio"])
    ratio = out[grad_col] / out[lambda_col]
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    if smooth_window and smooth_window > 1:
        ratio = ratio.rolling(window=smooth_window, min_periods=max(1, smooth_window // 2)).mean()
    out["stability_ratio"] = ratio
    return out[[step_col, "stability_ratio"]].dropna(subset=["stability_ratio"]).reset_index(drop=True)


def stability_ratio_summary(
    df: pd.DataFrame,
    step_col: str,
    grad_col: str,
    lambda_col: str,
    *,
    smooth_window: int = 1,
) -> Dict[str, object]:
    """Scalar summary of the stability ratio series."""
    series = stability_ratio_series(df, step_col, grad_col, lambda_col, smooth_window=smooth_window)
    if series.empty:
        return {"n": 0, "median": None, "mean": None, "max": None, "min": None, "last": None}
    r = series["stability_ratio"].to_numpy()
    return {
        "n": int(r.size),
        "median": _as_float(np.median(r)),
        "mean": _as_float(r.mean()),
        "max": _as_float(r.max()),
        "min": _as_float(r.min()),
        "last": _as_float(r[-1]),
    }
