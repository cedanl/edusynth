"""Benchmark-harness — synthesekwaliteit over vaste SDV-demo-datasets.

Draait met een vaste seed synthese over een diverse shortlist demo-datasets en
dumpt de bestaande validatiemetrieken (per-kolom TV/Wasserstein, correlatie-delta's,
sdmetrics QualityReport, DCR/NNDR) in één overzicht. Demo-data is grond-waarheid:
afwijking is dus objectief meetbaar. Met vaste seed is de uitkomst herhaalbaar.

Geen nieuwe metrieken — puur hergebruik van ``core/validate.py`` en ``core/synthesize.py``.

Gebruik:
    uv run scripts/benchmark.py
    uv run scripts/benchmark.py --rows 2000
    uv run scripts/benchmark.py --datasets adult insurance
    uv run scripts/benchmark.py --output-dir data/benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

from edu_synth.core import validate as V
from edu_synth.core.synthesize import fit, sample

# Vaste, diverse shortlist — gekozen op variatie in kolomtypes zodat zichtbaar
# wordt welk type structureel breekt:
#   student_placements  — onderwijsdata, alle typen (datum, bool, numeriek) + correlaties
#   fake_hotel_guests   — PII/hoge cardinaliteit (e-mail, adres, creditcard) + datums
#   adult               — hoge-cardinaliteit categorieën + scheve numerieke kolommen
#   expedia_hotel_logs  — categorie-zwaar + meerdere datumkolommen
DATASETS: list[str] = [
    "student_placements",
    "fake_hotel_guests",
    "adult",
    "expedia_hotel_logs",
]

SEED = 42
DEFAULT_ROW_CAP = 5_000  # cap op trainings-/generatierijen zodat een run snel blijft
DEFAULT_OUTPUT = Path("data/benchmark")  # data/ is gitignored — lokale dump

# Getrackte baseline (níét in data/, dat is gitignored) — het ijkpunt voor --check.
BASELINE_PATH = Path(__file__).parent / "benchmark_baseline.json"

# Fideliteitsmetrieken die de regressietest bewaakt, met hun richting.
# Privacy/DCR staat bewust niet in de gate: te ruisig en niet-monotoon.
_GUARDED: dict[str, str] = {
    "mean_score": "lower",  # lager = dichter bij echte data
    "worst_score": "lower",
    "cols_failed": "lower",
    "sdmetrics_overall": "higher",  # hoger = betere kwaliteit
}

# Een verslechtering telt pas als regressie boven deze marge. Relatief (10%) vangt
# de uiteenlopende schalen (score 0.16 vs 3.5); de absolute vloer voorkomt dat ruis
# bij kleine waarden al triggert.
_REL_TOL = 0.10
_ABS_FLOOR = 0.01


def _metadata_dict(df: pd.DataFrame) -> dict:
    """Auto-detecteer metadata, hetzelfde pad als ``fit()`` zonder schema gebruikt."""
    from sdv.metadata import SingleTableMetadata

    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)
    return meta.to_dict()


def run_dataset(name: str, row_cap: int, seed: int) -> dict:
    """Synthetiseer één demo-dataset en verzamel alle validatiemetrieken.

    Retourneert een dict met een ``summary``-rij plus de losse rapporten, zodat de
    aanroeper zowel het overzicht als de detail-CSV's kan schrijven.
    """
    from sdv.datasets.demo import download_demo

    real, _ = download_demo(modality="single_table", dataset_name=name)
    if len(real) > row_cap:
        real = real.sample(row_cap, random_state=seed).reset_index(drop=True)

    # Train op de volledige tabel, maar beoordeel alleen de échte datakolommen.
    # Identifiers (sdtype "id") randomiseert SDV bewust — distributie-afstand erop
    # is betekenisloos en zou de meetlat domineren. We synthetiseren ze wel mee,
    # maar laten ze buiten de scores.
    metadata = _metadata_dict(real)
    id_cols = [c for c, info in metadata["columns"].items() if info.get("sdtype") == "id"]

    model = fit(real, seed=seed)
    synth = sample(model, len(real))

    real_eval = real.drop(columns=id_cols)
    synth_eval = synth.drop(columns=id_cols)
    eval_meta = {**metadata, "columns": {c: metadata["columns"][c] for c in real_eval.columns}}

    report = V.evaluate(real_eval, synth_eval, eval_meta)
    pairs = V.evaluate_pairs(real_eval, synth_eval)
    sdm = V.evaluate_sdmetrics(real_eval, synth_eval, eval_meta)
    priv = V.evaluate_privacy(real_eval, synth_eval)

    # Hoogste score = grootste afstand tot de echte data = slechtst gesynthetiseerde kolom.
    scored = [r for r in report.rows if "score" in r]
    worst = max(scored, key=lambda r: r["score"], default=None)

    summary = {
        "dataset": name,
        "rows": len(real),
        "cols": len(report.rows),
        "worst_col": worst["column"] if worst else "",
        "worst_score": round(worst["score"], 4) if worst else 0.0,
        "mean_score": round(sum(r["score"] for r in scored) / len(scored), 4) if scored else 0.0,
        "cols_failed": sum(1 for r in scored if not r.get("ok", True)),
        "corr_flagged": len(pairs.flagged) if pairs.available else 0,
        "corr_max_delta": pairs.flagged[0]["delta"] if pairs.available and pairs.flagged else 0.0,
        "sdmetrics_overall": sdm.overall_score if sdm.available else None,
        "privacy_risk": priv.risk_level if priv.available else "n.v.t.",
        "dcr_ratio": priv.dcr_ratio if priv.available else None,
    }
    return {"summary": summary, "report": report, "pairs": pairs, "sdm": sdm}


def _markdown_table(rows: list[dict]) -> str:
    """Render een lijst dicts als GitHub-markdown-tabel (geen extra dependency)."""
    if not rows:
        return "_(geen resultaten)_"
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append(
            "| " + " | ".join("" if row[h] is None else str(row[h]) for h in headers) + " |"
        )
    return "\n".join(lines)


def _write_details(out_dir: Path, name: str, result: dict) -> None:
    """Schrijf per dataset de detail-CSV's: per-kolom scores, correlaties, sdmetrics."""
    ds_dir = out_dir / name
    ds_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(result["report"].rows).to_csv(ds_dir / "columns.csv", index=False)

    pairs = result["pairs"]
    if pairs.available and pairs.flagged:
        pd.DataFrame(pairs.flagged).to_csv(ds_dir / "correlations.csv", index=False)

    sdm = result["sdm"]
    if sdm.available and sdm.column_shapes:
        pd.DataFrame(sdm.column_shapes).to_csv(ds_dir / "sdmetrics_shapes.csv", index=False)


def _is_regression(baseline_val: float, current_val: float, direction: str) -> bool:
    """Is *current_val* merkbaar slechter dan *baseline_val* in de gegeven richting?"""
    if direction == "lower":  # lager is beter → een stijging boven de marge is slecht
        return current_val > baseline_val * (1 + _REL_TOL) + _ABS_FLOOR
    return current_val < baseline_val * (1 - _REL_TOL) - _ABS_FLOOR  # hoger is beter


def check_against_baseline(baseline: dict, current: list[dict]) -> list[dict]:
    """Vergelijk een verse run met de baseline en geef de regressies terug.

    Retourneert per gevallen metriek een rij; een lege lijst betekent geen regressie.
    """
    base_by = {row["dataset"]: row for row in baseline.get("datasets", [])}
    regressions: list[dict] = []
    for cur in current:
        base = base_by.get(cur["dataset"])
        if base is None:
            continue  # nieuwe dataset zonder ijkpunt — niets om tegen te vergelijken
        # Stond de dataset in de baseline maar levert hij nu geen scores op, dan is de
        # run gecrasht — dat is óók een regressie, geen stille pass.
        if not any(m in cur for m in _GUARDED):
            regressions.append(
                {
                    "dataset": cur["dataset"],
                    "metric": "(run mislukt)",
                    "baseline": "ok",
                    "current": cur.get("worst_col", "ERR"),
                    "direction": "—",
                }
            )
            continue
        for metric, direction in _GUARDED.items():
            b, c = base.get(metric), cur.get(metric)
            if b is None or c is None:
                continue
            if _is_regression(b, c, direction):
                regressions.append(
                    {
                        "dataset": cur["dataset"],
                        "metric": metric,
                        "baseline": b,
                        "current": c,
                        "direction": direction,
                    }
                )
    return regressions


def _write_baseline(path: Path, summaries: list[dict], seed: int, row_cap: int) -> None:
    """Leg de huidige scores vast als ijkpunt, met versie-info voor traceerbaarheid."""
    import sdmetrics
    import sdv

    payload = {
        "seed": seed,
        "row_cap": row_cap,
        "sdv_version": sdv.__version__,
        "sdmetrics_version": sdmetrics.__version__,
        "datasets": summaries,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS, help="Demo-datasets om te draaien."
    )
    parser.add_argument(
        "--rows", type=int, default=DEFAULT_ROW_CAP, help="Max trainings-/generatierijen."
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (vast → herhaalbaar).")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Map voor de CSV-dump."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--update-baseline",
        action="store_true",
        help=f"Schrijf de scores als nieuw ijkpunt naar {BASELINE_PATH.name}.",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="Vergelijk met de baseline; exit 1 bij een regressie boven de marge.",
    )
    args = parser.parse_args()

    # SDV/sdmetrics zijn luidruchtig met deprecation-waarschuwingen — onderdruk ze
    # zodat de tabel leesbaar uit de terminal komt.
    warnings.filterwarnings("ignore")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    for name in args.datasets:
        print(f"→ {name} …", flush=True)
        try:
            result = run_dataset(name, args.rows, args.seed)
        except Exception as exc:  # één kapotte dataset mag de hele run niet stoppen
            print(f"  ✗ overgeslagen: {type(exc).__name__}: {exc}")
            summaries.append({"dataset": name, "rows": 0, "cols": 0, "worst_col": f"ERR: {exc}"})
            continue
        _write_details(args.output_dir, name, result)
        summaries.append(result["summary"])

    table = _markdown_table(summaries)
    pd.DataFrame(summaries).to_csv(args.output_dir / "summary.csv", index=False)
    md_path = args.output_dir / "summary.md"
    title = "# Benchmark synthesekwaliteit\n\n"
    subtitle = f"Seed {args.seed} · max {args.rows} rijen per dataset.\n\n"
    md_path.write_text(title + subtitle + table + "\n", encoding="utf-8")

    print("\n" + table)
    print(f"\nCSV's en summary.md geschreven naar {args.output_dir}/")

    if args.update_baseline:
        _write_baseline(BASELINE_PATH, summaries, args.seed, args.rows)
        print(f"Baseline bijgewerkt: {BASELINE_PATH}")
        return

    if args.check:
        if not BASELINE_PATH.exists():
            print(f"\nGeen baseline gevonden ({BASELINE_PATH}). Draai eerst --update-baseline.")
            sys.exit(2)
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        regressions = check_against_baseline(baseline, summaries)
        if regressions:
            print("\n✗ Regressie t.o.v. baseline:")
            print(_markdown_table(regressions))
            sys.exit(1)
        print("\n✓ Geen regressie t.o.v. baseline.")


if __name__ == "__main__":
    main()
