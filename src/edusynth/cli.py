"""CLI voor edusynth — dun laagje over de Python API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="edusynth",
        description="Synthetische data genereren voor Nederlandse onderwijsdatasets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # edusynth synthesize
    p_synth = sub.add_parser("synthesize", help="Genereer synthetische data.")
    p_synth.add_argument("input", type=Path, help="Pad naar het inputbestand (CSV of Parquet).")
    p_synth.add_argument("schema", type=Path, help="Pad naar het schema YAML-bestand.")
    p_synth.add_argument("output", type=Path, help="Pad voor het outputbestand (CSV of Parquet).")
    p_synth.add_argument("--rows", type=int, default=1000, help="Aantal te genereren rijen (standaard: 1000).")

    # edusynth validate
    p_val = sub.add_parser("validate", help="Vergelijk echte en synthetische data.")
    p_val.add_argument("real", type=Path, help="Pad naar het echte databestand.")
    p_val.add_argument("synthetic", type=Path, help="Pad naar het synthetische databestand.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.command == "synthesize":
        _cmd_synthesize(args)
    elif args.command == "validate":
        _cmd_validate(args)


def _cmd_synthesize(args: argparse.Namespace) -> None:
    import pandas as pd
    from edusynth.synthesize import fit, sample

    data = _read(args.input)
    model = fit(data, args.schema)
    result = sample(model, args.rows)
    _write(result, args.output)

    from rich.console import Console
    Console().print(f"[green]✓[/green] {args.rows} rijen geschreven naar {args.output}")


def _cmd_validate(args: argparse.Namespace) -> None:
    from edusynth.validate import evaluate

    real = _read(args.real)
    synth = _read(args.synthetic)
    report = evaluate(real, synth)
    report.print()
    sys.exit(0 if report.passed() else 1)


def _read(path: Path) -> "pd.DataFrame":
    import pandas as pd
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write(df: "pd.DataFrame", path: Path) -> None:
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
