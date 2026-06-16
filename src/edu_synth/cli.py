"""CLI voor edu-synth — dun laagje over de Python API."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="edu-synth",
        description="Synthetische data genereren voor Nederlandse onderwijsdatasets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # edu-synth app
    sub.add_parser("app", help="Start de Streamlit-app in de browser.")

    # edu-synth synthesize
    p_synth = sub.add_parser("synthesize", help="Genereer synthetische data (batchmodus).")
    p_synth.add_argument("input", type=Path, help="Pad naar het inputbestand (CSV of Parquet).")
    p_synth.add_argument("output", type=Path, help="Pad voor het outputbestand (CSV of Parquet).")
    p_synth.add_argument("--schema", type=Path, default=None, help="Optioneel schema YAML-bestand.")
    p_synth.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Aantal te genereren rijen (standaard: zelfde als input).",
    )
    p_synth.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed voor reproduceerbare output (zelfde seed + data → identiek).",
    )

    # edu-synth validate
    p_val = sub.add_parser("validate", help="Vergelijk echte en synthetische data.")
    p_val.add_argument("real", type=Path, help="Pad naar het echte databestand.")
    p_val.add_argument("synthetic", type=Path, help="Pad naar het synthetische databestand.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.command == "app":
        _cmd_app()
    elif args.command == "synthesize":
        _cmd_synthesize(args)
    elif args.command == "validate":
        _cmd_validate(args)


# Huisstijl + privacy als env-vars, zodat ze ook gelden bij een pip-install
# vanuit een willekeurige map (Streamlit leest .streamlit/config.toml alleen
# relatief aan de werkmap). setdefault laat de container/gebruiker overschrijven —
# bv. STREAMLIT_SERVER_HEADLESS in de devcontainer. Headless zetten we hier
# bewust niet: zo opent de browser vanzelf op een desktop.
_APP_ENV = {
    "STREAMLIT_THEME_PRIMARY_COLOR": "#3D68EC",
    "STREAMLIT_THEME_BACKGROUND_COLOR": "#FFFFFF",
    "STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR": "#F9FAFB",
    "STREAMLIT_THEME_TEXT_COLOR": "#000000",
    "STREAMLIT_THEME_FONT": "sans serif",
    "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
}


def _cmd_app() -> None:
    from streamlit.web import cli as stcli

    for key, value in _APP_ENV.items():
        os.environ.setdefault(key, value)

    app_file = str(Path(__file__).parent / "app.py")
    sys.argv = ["streamlit", "run", app_file]
    sys.exit(stcli.main())


def _cmd_synthesize(args: argparse.Namespace) -> None:
    from edu_synth.core.synthesize import fit, sample

    data = _read(args.input)
    n_rows = args.rows if args.rows is not None else len(data)
    model = fit(data, args.schema, seed=args.seed)
    result = sample(model, n_rows)
    _write(result, args.output)

    from rich.console import Console

    Console().print(f"[green]✓[/green] {n_rows} rijen geschreven naar {args.output}")


def _cmd_validate(args: argparse.Namespace) -> None:
    from edu_synth.core.validate import evaluate

    real = _read(args.real)
    synth = _read(args.synthetic)
    report = evaluate(real, synth)
    report.print()
    sys.exit(0 if report.passed() else 1)


def _read(path: Path) -> pd.DataFrame:
    import pandas as pd

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
