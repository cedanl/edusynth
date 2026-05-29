"""Validation utilities — compare real and synthetic DataFrames."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from scipy.stats import wasserstein_distance


@dataclass
class Report:
    """Validation report with per-column distances."""

    rows: list[dict] = field(default_factory=list)

    def passed(self) -> bool:
        """True if no column exceeds its threshold."""
        return all(r.get("ok", True) for r in self.rows)

    def to_dataframe(self) -> pd.DataFrame:
        return (
            pd.DataFrame(self.rows)
            .sort_values("distance", ascending=False)
            .reset_index(drop=True)
        )

    def print(self) -> None:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Validatierapport")
        table.add_column("Kolom")
        table.add_column("Type")
        table.add_column("Afstand", justify="right")
        table.add_column("Metric")
        table.add_column("OK")
        for r in sorted(self.rows, key=lambda x: x["distance"], reverse=True):
            ok = "✓" if r.get("ok", True) else "✗"
            table.add_row(r["column"], r["dtype"], f"{r['distance']:.4f}", r["metric"], ok)
        console.print(table)


def evaluate(real: pd.DataFrame, synth: pd.DataFrame) -> Report:
    """Compare *real* and *synth* column-by-column.

    Categorical columns → Total Variation distance  [0, 1]
    Numeric columns     → Wasserstein-1 distance     [0, ∞)
    """
    rows = []
    shared = set(real.columns) & set(synth.columns)

    for col in shared:
        r, s = real[col].dropna(), synth[col].dropna()
        if r.empty or s.empty:
            continue

        if pd.api.types.is_numeric_dtype(real[col]):
            dist = float(wasserstein_distance(r.to_numpy(float), s.to_numpy(float)))
            rows.append({
                "column": col, "dtype": "numeric",
                "distance": round(dist, 4), "metric": "wasserstein",
            })
        else:
            dist = _tv_distance(r, s)
            rows.append({
                "column": col, "dtype": "categorical",
                "distance": round(dist, 4), "metric": "tv", "ok": dist < 0.2,
            })

    return Report(rows=rows)


def _tv_distance(real: pd.Series, synth: pd.Series) -> float:
    real_p = real.value_counts(normalize=True)
    synth_p = synth.value_counts(normalize=True)
    cats = real_p.index.union(synth_p.index)
    return float(
        0.5 * (
            real_p.reindex(cats, fill_value=0.0) - synth_p.reindex(cats, fill_value=0.0)
        ).abs().sum()
    )
