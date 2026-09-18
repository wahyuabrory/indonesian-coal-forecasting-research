from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import ConfigurationError, FEATURE_GROUPS, load_config


TARGET_CHOICES = ("ADRO.JK", "PTBA.JK", "ITMG.JK", "all")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coal-forecast",
        description="Run leak-safe daily adjusted-close return experiments.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.toml"))
    parser.add_argument("--target", choices=TARGET_CHOICES, default="all")
    parser.add_argument("--feature-group", choices=FEATURE_GROUPS, default=None)
    parser.add_argument(
        "--model",
        choices=("baseline", "xgboost", "both", "gru", "transformer", "all"),
        default="both",
        help="Evaluate baseline, XGBoost, GRU, Transformer, or a configured comparison (default: both).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Redownload all configured Yahoo snapshots instead of using the cache.",
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    targets = config.target_symbols if args.target == "all" else (args.target,)
    from .pipeline import run

    written = run(
        config,
        targets=targets,
        feature_group=args.feature_group,
        model_mode=args.model,
        refresh=args.refresh,
        output_dir=args.output_dir,
    )
    for path in written:
        print(f"Wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(_parser().parse_args(argv))
    except ModuleNotFoundError as exc:
        dependency = exc.name or "a required package"
        print(
            f"error: missing dependency {dependency!r}. Install the project with "
            "pip install -e .",
            file=sys.stderr,
        )
        return 2
    except (ConfigurationError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
