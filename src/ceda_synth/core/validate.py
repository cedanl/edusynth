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
        return pd.DataFrame(self.rows).sort_values("score", ascending=False).reset_index(drop=True)

    def print(self) -> None:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Validatierapport")
        table.add_column("Kolom")
        table.add_column("Type")
        table.add_column("Score", justify="right")  # vergelijkbaar: TV / genorm. Wasserstein
        table.add_column("Ruw", justify="right")
        table.add_column("Metric")
        table.add_column("OK")
        for r in sorted(self.rows, key=lambda x: x["score"], reverse=True):
            ok = "✓" if r.get("ok", True) else "✗"
            table.add_row(
                r["column"],
                r["dtype"],
                f"{r['score']:.4f}",
                f"{r['distance']:.4f}",
                r["metric"],
                ok,
            )
        console.print(table)


def _count_modes(series: pd.Series) -> int:
    from scipy.signal import find_peaks

    counts, _ = np.histogram(series.dropna(), bins=min(30, series.nunique()))
    peaks, _ = find_peaks(counts.astype(float), height=counts.max() * 0.15, distance=2)
    return max(1, len(peaks))


# Drempel voor het vergelijkbare `score`-veld (TV-afstand én genormaliseerde
# Wasserstein liggen na normalisatie op dezelfde [0, ~]-schaal).
_SCORE_OK = 0.2


def evaluate(real: pd.DataFrame, synth: pd.DataFrame) -> Report:
    """Vergelijk *real* en *synth* kolom voor kolom.

    Categorisch → Total Variation afstand [0, 1]
    Numeriek    → Wasserstein-1 afstand, genormaliseerd op de IQR van de echte kolom

    Elke rij krijgt een `score`: TV-afstand (categorisch) of genormaliseerde
    Wasserstein (numeriek). Doordat beide op dezelfde schaal liggen, telt elke
    kolom even zwaar mee in het eindoordeel — een `distance` van 0.3 voor een
    jaarveld is niet langer onvergelijkbaar met 0.3 voor een EC-score.
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
            score = dist / _spread(r)
            row: dict = {
                "column": col,
                "dtype": "numeric",
                "distance": round(dist, 4),
                "score": round(score, 4),
                "metric": "wasserstein",
                "ok": score < _SCORE_OK,
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
                    "score": round(dist, 4),
                    "metric": "tv",
                    "ok": dist < _SCORE_OK,
                }
            )

    return Report(rows=rows, modal_flags=modal_flags)


def _spread(real: pd.Series) -> float:
    """Schaalmaat om Wasserstein te normaliseren: IQR, met fallback std → bereik → 1.

    Een constante kolom (alle fallbacks 0) geeft 1.0 terug; de afstand blijft dan
    gelijk aan de ruwe Wasserstein (0 bij identieke data).
    """
    iqr = float(real.quantile(0.75) - real.quantile(0.25))
    if iqr > 0:
        return iqr
    std = float(real.std())
    if std > 0:
        return std
    rng = float(real.max() - real.min())
    return rng if rng > 0 else 1.0


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

# Categorische kolommen met meer dan dit aandeel unieke waarden zijn vrije tekst of
# identifiers (naam, e-mail, vrij ingevulde toelichting) — geen quasi-identifier.
# We sluiten ze uit van de afstandsberekening (one-hot zou de matrix laten exploderen)
# en waarschuwen erover in de UI, zodat de gebruiker ze handmatig beoordeelt.
_MAX_CARDINALITY_RATIO = 0.5


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
    n_numeric_cols: int = 0
    n_categorical_cols: int = 0
    excluded_cols: list[str] = field(default_factory=list)
    reason: str = ""

    def passed(self) -> bool:
        return self.available and self.risk_level == "laag"


def evaluate_privacy(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    primary_key: str | None = None,
) -> PrivacyReport:
    """Schat het re-identificatierisico via DCR en NNDR.

    Methode (holdout-vergelijking uit discussion #3):
    1. Splits echte data in train (80%) en holdout (20%)
    2. DCR(synth  → train): hoe dicht zitten synthetische rijen bij trainingsdata?
    3. DCR(holdout → train): baseline — hoe dicht zitten onbekende echte rijen bij trainingsdata?
    4. Als DCR(synth) ≈ DCR(holdout), gedraagt synthetische data zich als een buitenstaander → goed.

    Zowel numerieke als categorische kolommen tellen mee: categorische velden
    (geslacht, instellingscode, opleidingscode) zijn juist de quasi-identifiers in
    onderwijsdata. Numeriek wordt [0, 1]-geschaald, categorisch wordt one-hot
    gecodeerd (nominaal — categorieën liggen onderling even ver uit elkaar).

    Noot: synthesizer is getraind op ALLE echte data, niet alleen train-split.
    Dit geeft een conservatieve schatting (worst-case benadering).
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

    shared = [c for c in real.columns if c in synth.columns]
    if primary_key and primary_key in shared:
        shared.remove(primary_key)

    numeric_cols = [c for c in shared if pd.api.types.is_numeric_dtype(real[c])]
    cat_candidates = [c for c in shared if c not in numeric_cols]

    # Identifier-/vrije-tekstkolommen (bijna uniek) zijn geen quasi-identifier en
    # zouden de one-hot matrix laten exploderen — uitsluiten en erover waarschuwen.
    n_real = len(real)
    cat_cols, excluded_cols = [], []
    for c in cat_candidates:
        ratio = real[c].nunique(dropna=True) / n_real if n_real else 1.0
        (excluded_cols if ratio > _MAX_CARDINALITY_RATIO else cat_cols).append(c)

    if not numeric_cols and not cat_cols:
        return PrivacyReport(
            available=False,
            excluded_cols=excluded_cols,
            reason="Geen bruikbare kolommen voor afstandsberekening",
        )

    use_cols = numeric_cols + cat_cols
    real_use = real[use_cols].dropna()
    synth_use = synth[use_cols].dropna()

    if len(real_use) < 20 or len(synth_use) < 10:
        return PrivacyReport(
            available=False,
            excluded_cols=excluded_cols,
            reason="Te weinig rijen voor analyse",
        )

    # Bouw één feature-matrix: [0, 1]-geschaalde numeriek + one-hot categorisch.
    # Encoder/scaler op echte data fitten, daarna identiek op synth toepassen.
    real_blocks, synth_blocks = [], []
    if numeric_cols:
        scaler = MinMaxScaler()
        real_blocks.append(scaler.fit_transform(real_use[numeric_cols]))
        synth_blocks.append(scaler.transform(synth_use[numeric_cols]))
    if cat_cols:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        real_blocks.append(enc.fit_transform(real_use[cat_cols].astype(str)))
        synth_blocks.append(enc.transform(synth_use[cat_cols].astype(str)))

    real_scaled = np.hstack(real_blocks).astype(np.float32)
    synth_scaled = np.hstack(synth_blocks).astype(np.float32)

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
        # Bij categorische exact-matches is de 2e buur vaak ook op afstand 0 (0/0):
        # ratio dan op 1.0 zetten (synth gedraagt zich als een willekeurige buur).
        second = dists2[:, 1]
        nndr = np.divide(dists2[:, 0], second, out=np.ones_like(second), where=second > 0)
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
        n_cols=len(use_cols),
        n_numeric_cols=len(numeric_cols),
        n_categorical_cols=len(cat_cols),
        excluded_cols=excluded_cols,
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

    # Alle kolommen tellen mee — numeriek (genorm. Wasserstein) net zo goed als
    # categorisch (TV). Een volledig verkeerd gesynthetiseerde numerieke kolom
    # beïnvloedt het eindoordeel nu wél.
    scored = [r for r in report.rows if "score" in r]
    max_score = max((r["score"] for r in scored), default=0.0)
    n_failed = sum(1 for r in scored if not r.get("ok", True))

    if max_score < 0.1 and n_failed == 0:
        return "Geschikt voor rapportages, kruistabellen en publicatie."
    if n_failed == 0:
        return (
            "Geschikt voor exploratieve analyse en trendgrafieken — "
            "controleer absolute frequenties vóór publicatie."
        )
    if n_failed <= max(1, len(scored) // 3):
        return (
            "Geschikt voor patroonverkenning en interne tests — "
            "niet aanbevolen voor publicatie van statistieken."
        )
    return (
        "Alleen geschikt voor technische tests — "
        "verdeling wijkt te sterk af voor inhoudelijke analyse."
    )
