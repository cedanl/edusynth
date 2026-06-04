"""Validation utilities — statistical and privacy validation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

# ── Statistische validatie ─────────────────────────────────────────────────────


@dataclass
class Report:
    """Per-kolom statistische afstandsmetrieken."""

    rows: list[dict] = field(default_factory=list)
    modal_flags: list[dict] = field(default_factory=list)  # {column, real_modes, synth_modes}

    def passed(self) -> bool:
        return all(r.get("ok", True) for r in self.rows)

    def to_dataframe(self) -> pd.DataFrame:
        return (
            pd.DataFrame(self.rows).sort_values("distance", ascending=False).reset_index(drop=True)
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


def _count_modes(series: pd.Series) -> int:
    from scipy.signal import find_peaks

    counts, _ = np.histogram(series.dropna(), bins=min(30, series.nunique()))
    peaks, _ = find_peaks(counts.astype(float), height=counts.max() * 0.15, distance=2)
    return max(1, len(peaks))


def evaluate(real: pd.DataFrame, synth: pd.DataFrame) -> Report:
    """Vergelijk *real* en *synth* kolom voor kolom.

    Categorisch → Total Variation afstand [0, 1]
    Numeriek    → Wasserstein-1 afstand   [0, ∞)
    """
    rows = []
    modal_flags = []
    shared = set(real.columns) & set(synth.columns)

    for col in shared:
        r, s = real[col].dropna(), synth[col].dropna()
        if r.empty or s.empty:
            continue

        if pd.api.types.is_numeric_dtype(real[col]):
            dist = float(wasserstein_distance(r.to_numpy(float), s.to_numpy(float)))
            row: dict = {
                "column": col,
                "dtype": "numeric",
                "distance": round(dist, 4),
                "metric": "wasserstein",
            }
            r_modes = _count_modes(r)
            s_modes = _count_modes(s)
            if r_modes >= 2 and s_modes < r_modes:
                row["modal_warning"] = f"{r_modes} pieken → {s_modes} pieken"
                modal_flags.append(
                    {
                        "column": col,
                        "real_modes": r_modes,
                        "synth_modes": s_modes,
                    }
                )
            rows.append(row)
        else:
            dist = _tv_distance(r, s)
            rows.append(
                {
                    "column": col,
                    "dtype": "categorical",
                    "distance": round(dist, 4),
                    "metric": "tv",
                    "ok": dist < 0.2,
                }
            )

    return Report(rows=rows, modal_flags=modal_flags)


def _tv_distance(real: pd.Series, synth: pd.Series) -> float:
    real_p = real.value_counts(normalize=True)
    synth_p = synth.value_counts(normalize=True)
    cats = real_p.index.union(synth_p.index)
    return float(
        0.5
        * (real_p.reindex(cats, fill_value=0.0) - synth_p.reindex(cats, fill_value=0.0)).abs().sum()
    )


# ── Privacyvalidatie (DCR / NNDR) ──────────────────────────────────────────────

_MAX_ROWS = 2_000  # sampling cap voor performance


@dataclass
class PrivacyReport:
    """DCR/NNDR privacyrapport gebaseerd op holdout-vergelijking.

    DCR ratio > 0.9  → laag risico  (synthetisch gedraagt zich als onbekende data)
    DCR ratio < 0.7  → hoog risico  (synthetisch zit te dicht op trainingsdata)
    """

    available: bool
    dcr_synth_median: float = 0.0
    dcr_holdout_median: float = 0.0
    dcr_ratio: float = 0.0
    nndr_median: float = 0.0
    risk_level: str = "onbekend"
    n_cols: int = 0
    reason: str = ""

    def passed(self) -> bool:
        return self.available and self.risk_level == "laag"


def evaluate_privacy(real: pd.DataFrame, synth: pd.DataFrame) -> PrivacyReport:
    """Schat het re-identificatierisico via DCR en NNDR.

    Methode (holdout-vergelijking uit discussion #3):
    1. Splits echte data in train (80%) en holdout (20%)
    2. DCR(synth  → train): hoe dicht zitten synthetische rijen bij trainingsdata?
    3. DCR(holdout → train): baseline — hoe dicht zitten onbekende echte rijen bij trainingsdata?
    4. Als DCR(synth) ≈ DCR(holdout), gedraagt synthetische data zich als een buitenstaander → goed.

    Noot: synthesizer is getraind op ALLE echte data, niet alleen train-split.
    Dit geeft een conservatieve schatting (worst-case benadering).
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import MinMaxScaler

    numeric_cols = [c for c in real.columns if pd.api.types.is_numeric_dtype(real[c])]
    if not numeric_cols:
        return PrivacyReport(available=False, reason="Geen numerieke kolommen")

    real_num = real[numeric_cols].dropna()
    synth_num = synth[numeric_cols].dropna()

    if len(real_num) < 20 or len(synth_num) < 10:
        return PrivacyReport(available=False, reason="Te weinig rijen voor analyse")

    # Normaliseer op schaal [0, 1] van echte data
    scaler = MinMaxScaler()
    real_scaled = scaler.fit_transform(real_num).astype(np.float32)
    synth_scaled = scaler.transform(synth_num).astype(np.float32)

    # Sample bij grote datasets
    rng = np.random.default_rng(42)
    if len(real_scaled) > _MAX_ROWS:
        real_scaled = real_scaled[rng.choice(len(real_scaled), _MAX_ROWS, replace=False)]
    if len(synth_scaled) > _MAX_ROWS:
        synth_scaled = synth_scaled[rng.choice(len(synth_scaled), _MAX_ROWS, replace=False)]

    # Holdout-split van echte data
    n_holdout = max(10, len(real_scaled) // 5)
    perm = rng.permutation(len(real_scaled))
    real_train = real_scaled[perm[n_holdout:]]
    real_holdout = real_scaled[perm[:n_holdout]]

    # DCR: afstand tot dichtstbijzijnde trainingsrij
    knn1 = NearestNeighbors(n_neighbors=1, metric="euclidean", algorithm="auto")
    knn1.fit(real_train)

    dcr_synth = knn1.kneighbors(synth_scaled)[0].flatten()
    dcr_holdout = knn1.kneighbors(real_holdout)[0].flatten()

    # NNDR: verhouding 1e / 2e dichtstbijzijnde buur
    n_neighbors = min(2, len(real_train))
    knn2 = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", algorithm="auto")
    knn2.fit(real_train)
    dists2 = knn2.kneighbors(synth_scaled)[0]
    if n_neighbors == 2:
        nndr = np.where(dists2[:, 1] > 0, dists2[:, 0] / dists2[:, 1], 1.0)
    else:
        nndr = np.ones(len(synth_scaled))

    med_synth = float(np.median(dcr_synth))
    med_holdout = float(np.median(dcr_holdout))
    dcr_ratio = med_synth / med_holdout if med_holdout > 0 else 1.0

    risk_level = "laag" if dcr_ratio >= 0.9 else ("matig" if dcr_ratio >= 0.7 else "hoog")

    return PrivacyReport(
        available=True,
        dcr_synth_median=round(med_synth, 4),
        dcr_holdout_median=round(med_holdout, 4),
        dcr_ratio=round(dcr_ratio, 3),
        nndr_median=round(float(np.median(nndr)), 3),
        risk_level=risk_level,
        n_cols=len(numeric_cols),
    )


# ── Bivariate correlatierapport ────────────────────────────────────────────────


@dataclass
class PairsReport:
    available: bool
    flagged: list[dict]  # {col_a, col_b, real_corr, synth_corr, delta}
    reason: str = ""


def evaluate_pairs(real: pd.DataFrame, synth: pd.DataFrame) -> PairsReport:
    """Vergelijk Pearson-correlaties tussen numerieke kolompar in echt vs. synthetisch."""
    numeric = [
        c for c in real.columns if pd.api.types.is_numeric_dtype(real[c]) and c in synth.columns
    ]
    if len(numeric) < 2:
        return PairsReport(available=False, flagged=[], reason="Minder dan 2 numerieke kolommen")

    real_corr = real[numeric].corr()
    synth_corr = synth[numeric].corr()
    flagged = []
    for i, c1 in enumerate(numeric):
        for c2 in numeric[i + 1 :]:
            rc = real_corr.loc[c1, c2]
            sc = synth_corr.loc[c1, c2]
            if pd.isna(rc) or pd.isna(sc):
                continue
            delta = abs(rc - sc)
            if delta > 0.1:
                flagged.append(
                    {
                        "col_a": c1,
                        "col_b": c2,
                        "real_corr": round(float(rc), 3),
                        "synth_corr": round(float(sc), 3),
                        "delta": round(delta, 3),
                    }
                )
    return PairsReport(
        available=True,
        flagged=sorted(flagged, key=lambda x: x["delta"], reverse=True),
    )


# ── Gebruiksaanbeveling ────────────────────────────────────────────────────────


def usage_recommendation(report: Report, priv: PrivacyReport | None = None) -> str:
    """Vertaal validatiescores naar een concrete gebruiksaanbeveling."""
    priv_risk = priv.risk_level if (priv and priv.available) else "onbekend"

    if priv_risk == "hoog":
        return (
            "Niet aanbevolen voor publicatie — hoog privacyrisico. "
            "Pas de syntheseinstellingen aan of raadpleeg uw FG."
        )

    tv_rows = [r for r in report.rows if r.get("metric") == "tv"]
    max_tv = max((r["distance"] for r in tv_rows), default=0.0)
    n_failed = sum(1 for r in tv_rows if not r.get("ok", True))

    if max_tv < 0.1 and n_failed == 0:
        return "Geschikt voor rapportages, kruistabellen en publicatie."
    if n_failed == 0:
        return (
            "Geschikt voor exploratieve analyse en trendgrafieken — "
            "controleer absolute frequenties vóór publicatie."
        )
    if n_failed <= max(1, len(tv_rows) // 3):
        return (
            "Geschikt voor patroonverkenning en interne tests — "
            "niet aanbevolen voor publicatie van statistieken."
        )
    return (
        "Alleen geschikt voor technische tests — "
        "verdeling wijkt te sterk af voor inhoudelijke analyse."
    )
