"""eos_signals CLI.

Examples:
  python -m eos_signals extract --run z7588exm --project resnet-0.05 \
      --metric lambda_max --signal crossing --threshold 2/eta --eta 0.05

  python -m eos_signals extract --run z7588exm --project resnet-0.05 \
      --metric lambda_max --signal slope --start-step 1000

  python -m eos_signals list
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from . import primitives
from .wandb_io import fetch_history, fetch_run_config


def _parse_threshold(expr: str, eta: float | None) -> tuple[float, bool, str]:
    """Resolve a threshold expression to a numeric threshold.

    Accepts: "<number>", "2/eta" (SGD), "38/eta" (Adam).
    Returns (threshold, adam, expr).
    """
    s = expr.strip()
    if s == "2/eta":
        if eta is None:
            raise SystemExit("--threshold '2/eta' requires --eta")
        return primitives.eos_threshold(eta, adam=False), False, "2/eta"
    if s == "38/eta":
        if eta is None:
            raise SystemExit("--threshold '38/eta' requires --eta")
        return primitives.eos_threshold(eta, adam=True), True, "38/eta"
    try:
        return float(s), False, s
    except ValueError as e:
        raise SystemExit(f"invalid --threshold: {expr!r}") from e


def _cmd_list(_args) -> int:
    import importlib.resources as pkg
    from pathlib import Path

    here = Path(__file__).resolve().parent
    with (here / "registry.yaml").open() as f:
        sys.stdout.write(f.read())
    return 0


def _resolve_eta(args) -> float | None:
    if args.eta is not None:
        return args.eta
    if args.eta_from_config:
        cfg = fetch_run_config(args.run, project=args.project, entity=args.entity)
        for key in ("lr", "learning_rate", "optimizer_lr"):
            if key in cfg and cfg[key] is not None:
                return float(cfg[key])
        raise SystemExit(f"--eta-from-config: no lr/learning_rate key in run config ({sorted(cfg)[:10]}...)")
    return None


def _cmd_extract(args) -> int:
    extra_metrics = []
    if args.signal == "stability_ratio":
        if not (args.grad_col and args.lambda_col):
            raise SystemExit("--signal stability_ratio needs --grad-col and --lambda-col")
        extra_metrics = [args.grad_col, args.lambda_col]
        fetch_metric = None
    else:
        if not args.metric:
            raise SystemExit(f"--signal {args.signal} needs --metric")
        fetch_metric = args.metric

    metrics = [m for m in [fetch_metric, *extra_metrics] if m]
    df = fetch_history(
        args.run,
        metrics,
        project=args.project,
        entity=args.entity,
        step_col=args.step_col,
        samples=args.samples,
        use_scan=args.scan,
    )

    result: Dict[str, Any]
    if args.signal == "crossing":
        thresh, _, expr = _parse_threshold(args.threshold, _resolve_eta(args))
        result = primitives.crossing_step(
            df, args.step_col, args.metric, thresh,
            direction=args.direction, start_step=args.start_step, end_step=args.end_step,
        )
        result["threshold_expr"] = expr
    elif args.signal == "eos_crossing":
        eta = _resolve_eta(args)
        if eta is None:
            raise SystemExit("--signal eos_crossing requires --eta or --eta-from-config")
        result = primitives.eos_threshold_crossing(
            df, args.step_col, args.metric, eta,
            adam=args.adam, start_step=args.start_step, end_step=args.end_step,
        )
    elif args.signal == "slope":
        result = primitives.slope_in_window(df, args.step_col, args.metric, args.start_step, args.end_step)
    elif args.signal == "plateau":
        if args.tolerance is None:
            raise SystemExit("--signal plateau requires --tolerance")
        result = primitives.plateau_duration(df, args.step_col, args.metric, args.tolerance, min_length=args.min_length)
    elif args.signal == "stability_ratio":
        result = primitives.stability_ratio_summary(
            df, args.step_col, args.grad_col, args.lambda_col, smooth_window=args.smooth_window
        )
    else:
        raise SystemExit(f"unknown --signal {args.signal!r}")

    envelope = {
        "run": args.run,
        "project": args.project,
        "entity": args.entity,
        "metric": args.metric,
        "signal": args.signal,
        "result": result,
    }
    json.dump(envelope, sys.stdout, default=str, indent=None if args.compact else 2)
    sys.stdout.write("\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eos_signals", description="Extract numerical signals from W&B history.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="Print the primitive registry.")
    pl.set_defaults(func=_cmd_list)

    pe = sub.add_parser("extract", help="Fetch a run's history and run one signal primitive.")
    pe.add_argument("--run", required=True, help="W&B run id, or entity/project/run path.")
    pe.add_argument("--project", default=None, help="W&B project (if --run is bare id).")
    pe.add_argument("--entity", default=None, help="W&B entity/team.")
    pe.add_argument("--metric", default=None, help="Metric column to analyze (signals other than stability_ratio).")
    pe.add_argument("--step-col", default="_step", help="Step column in history (default: _step).")
    pe.add_argument("--samples", type=int, default=100_000, help="Max history samples to fetch.")
    pe.add_argument("--scan", action="store_true", help="Use scan_history for full-resolution fetch.")
    pe.add_argument("--compact", action="store_true", help="Emit single-line JSON.")

    pe.add_argument("--signal", required=True,
                    choices=["crossing", "eos_crossing", "slope", "plateau", "stability_ratio"])
    pe.add_argument("--threshold", default=None, help="For --signal crossing: number, '2/eta', or '38/eta'.")
    pe.add_argument("--direction", default="up", choices=["up", "down"], help="Crossing direction.")
    pe.add_argument("--eta", type=float, default=None, help="Learning rate for '2/eta' / '38/eta' or --signal eos_crossing.")
    pe.add_argument("--eta-from-config", action="store_true",
                    help="Pull eta from the run's W&B config (keys: lr, learning_rate, optimizer_lr).")
    pe.add_argument("--adam", action="store_true", help="Use Adam EOS threshold (38/eta) for eos_crossing.")
    pe.add_argument("--start-step", type=float, default=None, help="Window start for slope / plateau.")
    pe.add_argument("--end-step", type=float, default=None, help="Window end for slope / plateau.")
    pe.add_argument("--tolerance", type=float, default=None, help="Plateau: max-min tolerance.")
    pe.add_argument("--min-length", type=int, default=5, help="Plateau: minimum contiguous length.")
    pe.add_argument("--grad-col", default=None, help="grad_hessian_grad column for stability_ratio.")
    pe.add_argument("--lambda-col", default=None, help="lambda_max column for stability_ratio.")
    pe.add_argument("--smooth-window", type=int, default=1, help="Rolling-mean window for stability_ratio.")
    pe.set_defaults(func=_cmd_extract)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
