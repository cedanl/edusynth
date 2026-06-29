"""Validation utilities — statistical and privacy validation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def _is_numeric(series: pd.Series) -> bool:
    """Echt numeriek: een getalkolom, maar geen boolean.

    pandas rekent ``bool`` tot de numerieke dtypes, maar een ja/nee-kolom hoort
    als categorie vergeleken te worden (TV-afstand), niet via Wasserstein — anders
    crasht de IQR-berekening op booleans (``numpy boolean subtract``).
    """
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


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


def _as_timestamp(series: pd.Series, fmt: str | None) -> pd.Series:
    """Datumkolom naar nanoseconden sinds epoch, zodat Wasserstein erop werkt.

    Onparseerbare waarden worden NaT en vallen weg. Het ns-getal is groot, maar de
    score normaliseert op de IQR (ook in ns), dus schaalvrij.
    """
    dt = (
        pd.to_datetime(series, format=fmt, errors="coerce")
        if fmt
        else pd.to_datetime(series, errors="coerce")
    )
    return dt.dropna().astype("int64")


def evaluate(real: pd.DataFrame, synth: pd.DataFrame, metadata: dict | None = None) -> Report:
    """Vergelijk *real* en *synth* kolom voor kolom.

    Categorisch → Total Variation afstand [0, 1]
    Numeriek    → Wasserstein-1 afstand, genormaliseerd op de IQR van de echte kolom
    Datum       → naar timestamp, daarna als numeriek (zie *metadata*)

    Elke rij krijgt een `score`: TV-afstand (categorisch) of genormaliseerde
    Wasserstein (numeriek). Doordat beide op dezelfde schaal liggen, telt elke
    kolom even zwaar mee in het eindoordeel — een `distance` van 0.3 voor een
    jaarveld is niet langer onvergelijkbaar met 0.3 voor een EC-score.

    *metadata* is de SDV-metadata-dict. Zonder metadata wordt het type uit de
    pandas-dtype afgeleid (oud gedrag). Mét metadata worden datumkolommen via het
    Wasserstein-pad gescoord (niet als hoog-cardinale categorie, wat vals alarm
    gaf) en worden id-kolommen overgeslagen — een id heeft geen verdeling om te
    bewaren.
    """
    columns_meta = (metadata or {}).get("columns", {})
    rows = []
    modal_flags = []
    shared = set(real.columns) & set(synth.columns)

    for col in shared:
        sdtype = columns_meta.get(col, {}).get("sdtype")
        if sdtype == "id":
            continue

        r, s = real[col].dropna(), synth[col].dropna()
        if r.empty or s.empty:
            continue

        is_datetime = sdtype == "datetime"
        if is_datetime:
            fmt = columns_meta.get(col, {}).get("datetime_format")
            r, s = _as_timestamp(r, fmt), _as_timestamp(s, fmt)
            if r.empty or s.empty:
                continue

        # Numeriek pad alleen als de kolom in zowel echt als synthetisch numeriek
        # is (datums zijn na conversie int64). SDV kan een kolom onderweg omzetten
        # (bv. int-stadscode → stadsnamen) — dan zou Wasserstein op strings crashen
        # en valt de kolom terug op de TV-afstand.
        if is_datetime or (_is_numeric(real[col]) and _is_numeric(s)):
            dist = float(wasserstein_distance(r.to_numpy(float), s.to_numpy(float)))
            score = dist / _spread(r)
            row: dict = {
                "column": col,
                "dtype": "datetime" if is_datetime else "numeric",
                "distance": round(dist, 4),
                "score": round(score, 4),
                "metric": "wasserstein",
                "ok": score < _SCORE_OK,
            }
            # Multimodaliteit is een vorm-signaal voor echte meetwaarden; op
            # timestamps niet zinvol, dus alleen voor niet-datum numeriek.
            if not is_datetime:
                r_modes = _count_modes(r)
                s_modes = _count_modes(s)
                if r_modes >= 2 and s_modes < r_modes:
                    row["modal_warning"] = f"{r_modes} pieken → {s_modes} pieken"
                    modal_flags.append(
                        {"column": col, "real_modes": r_modes, "synth_modes": s_modes}
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

    # Numeriek alleen als de kolom in beide tabellen numeriek is; anders crasht de
    # MinMaxScaler op tekst (zie _is_numeric). De rest gaat one-hot als categorie.
    numeric_cols = [c for c in shared if _is_numeric(real[c]) and _is_numeric(synth[c])]
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
    # Alleen kolommen die in beide tabellen numeriek zijn — een kolom die SDV naar
    # tekst omzette zou ``.corr()`` laten crashen (zie _is_numeric).
    numeric = [
        c
        for c in real.columns
        if c in synth.columns and _is_numeric(real[c]) and _is_numeric(synth[c])
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


# Een omgeklapt verband (teken draait om op een betekenisvol verband) is de
# zwaarste correlatiefout: wie op dat verband een analyse bouwt, trekt een
# omgekeerde conclusie. Een grote delta zonder tekenomslag is ook ernstig; milde
# afwijkingen blijven "matig".
_CORR_SIGNIFICANT = 0.15  # |corr| in de echte data waaronder een tekenomslag ruis is
_CORR_LARGE_DELTA = 0.3  # delta hierboven → ernstige afwijking, ook zonder tekenomslag


def correlation_risk(pairs: PairsReport) -> str:
    """Vertaal de geflagde correlatieparen naar 'laag' | 'matig' | 'hoog'.

    - ``hoog``: een tekenomslag op een betekenisvol verband (positief wordt
      negatief of omgekeerd) of een grote delta;
    - ``matig``: wel geflagde paren, maar mild;
    - ``laag``: geen geflagde paren, of correlatie niet berekenbaar (< 2
      numerieke kolommen) — dat laatste mag het oordeel niet verlagen.
    """
    if not pairs.available or not pairs.flagged:
        return "laag"
    for p in pairs.flagged:
        rc, sc = p["real_corr"], p["synth_corr"]
        flipped = rc * sc < 0 and abs(rc) >= _CORR_SIGNIFICANT
        if flipped or p["delta"] > _CORR_LARGE_DELTA:
            return "hoog"
    return "matig"


# ── sdmetrics QualityReport ──────────────────────────────────────────────────────

_SDMETRICS_MAX_ROWS = 5_000  # cap zodat de report responsive blijft bij grote sets


@dataclass
class SDMetricsReport:
    """sdmetrics QualityReport — column shapes + column pair trends.

    Vult de eigen TV/Wasserstein-snelvalidatie aan met aanvullende sdmetrics-
    metrieken. Column Pair Trends dekt categorisch × categorisch
    (ContingencySimilarity) en categorisch × numeriek — verbanden die de
    Pearson-only `evaluate_pairs` mist.
    """

    available: bool
    overall_score: float | None = None
    column_shapes: list[dict] = field(default_factory=list)
    column_pair_trends: list[dict] = field(default_factory=list)
    reason: str = ""


def _subsample(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    return df if len(df) <= cap else df.sample(cap, random_state=42)


def _round_records(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    for col in df.select_dtypes("number").columns:
        df[col] = df[col].round(4)
    return df.to_dict("records")


def evaluate_sdmetrics(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    metadata_dict: dict | None,
) -> SDMetricsReport:
    """Bereken de sdmetrics QualityReport voor *real* vs. *synth*.

    Kolomparen met een zwakke samenhang in de echte data (onder de sdmetrics-
    associatiedrempel) krijgen score NaN en blijven buiten beschouwing — dat is
    geen fout. `get_score()` middelt met nanmean, dus de overall-score blijft
    geldig zolang Column Shapes berekend kon worden.
    """
    if not metadata_dict or not metadata_dict.get("columns"):
        return SDMetricsReport(available=False, reason="Geen metadata beschikbaar")

    # sdmetrics vereist dat data en metadata exact dezelfde kolommen dekken.
    cols = [c for c in real.columns if c in synth.columns and c in metadata_dict["columns"]]
    if not cols:
        return SDMetricsReport(available=False, reason="Geen gedeelde kolommen met metadata")

    meta = {**metadata_dict, "columns": {c: metadata_dict["columns"][c] for c in cols}}
    real_s = _subsample(real[cols], _SDMETRICS_MAX_ROWS)
    synth_s = _subsample(synth[cols], _SDMETRICS_MAX_ROWS)

    try:
        from sdmetrics.reports.single_table import QualityReport

        report = QualityReport()
        report.generate(real_s, synth_s, meta, verbose=False)
        overall = float(report.get_score())
        shapes = _round_records(report.get_details("Column Shapes"))
        pairs = _round_records(report.get_details("Column Pair Trends"))
    except Exception as exc:
        return SDMetricsReport(available=False, reason=str(exc))

    return SDMetricsReport(
        available=True,
        overall_score=None if pd.isna(overall) else round(overall, 4),
        column_shapes=shapes,
        column_pair_trends=pairs,
    )


# ── Gebruiksaanbeveling ────────────────────────────────────────────────────────

# De drempelwaarden hieronder zijn een operationele vuistregel, niet ontleend aan
# een vastgestelde norm. Het oordeel beschrijft daarom statistische kwaliteit en
# bruikbaarheid in neutrale termen — de uiteindelijke afweging blijft aan de
# gebruiker; zie de disclaimer.
RECOMMENDATION_DISCLAIMER = (
    "Dit oordeel is een operationele vuistregel op basis van afstandsmetrieken "
    "(TV, genormaliseerde Wasserstein), geen vastgestelde norm. "
    "Beoordeel zelf of de kwaliteit volstaat voor het beoogde gebruik."
)


def usage_recommendation(
    report: Report,
    priv: PrivacyReport | None = None,
    pairs: PairsReport | None = None,
) -> str:
    """Vertaal validatiescores naar een neutrale kwaliteits- en bruikbaarheidsindicatie.

    Beschrijft de statistische gelijkenis met de echte data; claimt bewust geen
    vastgestelde geschiktheid (zie :data:`RECOMMENDATION_DISCLAIMER`). Als *pairs*
    is meegegeven, weegt het correlatiebehoud mee — een omgeklapt verband krijgt
    een expliciete waarschuwing.
    """
    priv_risk = priv.risk_level if (priv and priv.available) else "onbekend"

    if priv_risk == "hoog":
        return (
            "Niet aanbevolen — hoog privacyrisico. "
            "Pas de syntheseinstellingen aan of raadpleeg uw FG."
        )

    # Alle kolommen tellen mee — numeriek (genorm. Wasserstein) net zo goed als
    # categorisch (TV). Een volledig verkeerd gesynthetiseerde numerieke kolom
    # beïnvloedt het eindoordeel nu wél.
    scored = [r for r in report.rows if "score" in r]
    max_score = max((r["score"] for r in scored), default=0.0)
    n_failed = sum(1 for r in scored if not r.get("ok", True))

    if max_score < 0.1 and n_failed == 0:
        base = "Hoge statistische kwaliteit — alle kolomverdelingen liggen dicht bij de echte data."
    elif n_failed == 0:
        base = (
            "Goede statistische kwaliteit — verdelingen volgen de echte data met "
            "kleine afwijkingen. Controleer absolute frequenties vóór extern gebruik."
        )
    elif n_failed <= max(1, len(scored) // 3):
        base = (
            "Matige statistische kwaliteit — enkele kolommen wijken merkbaar af. "
            "Bruikbaar voor patroonverkenning en interne tests."
        )
    else:
        base = (
            "Lage statistische kwaliteit — meerdere kolommen wijken sterk af. "
            "Alleen geschikt voor technische tests."
        )

    corr_risk = correlation_risk(pairs) if pairs is not None else "laag"
    if corr_risk == "hoog":
        base += (
            " Let op: een of meer verbanden tussen kolommen zijn omgeklapt of wijken "
            "sterk af — controleer de correlaties vóór analyses die op samenhang leunen."
        )
    elif corr_risk == "matig":
        base += (
            " Enkele verbanden tussen kolommen wijken af; zie de correlaties in het "
            "validatierapport."
        )
    return base


# ── Verbeteradvies ─────────────────────────────────────────────────────────────
# Bij een matig/onvoldoende oordeel: vertaal de gemeten signalen naar concrete
# acties, gekoppeld aan de slechtst scorende kolommen. Elk advies wijst naar een
# knop in de app, zodat de gebruiker zelf kan bijsturen.
_SMALL_DATASET = 500  # onder dit rij-aantal heeft de synthesizer weinig houvast
_HIGH_CARDINALITY = 50  # categorische kolom met meer unieke waarden → kandidaat om te schrappen
_MAX_ADVICE = 4  # niet overladen; alleen de belangrijkste punten


def improvement_advice(
    report: Report,
    real: pd.DataFrame,
    priv: PrivacyReport | None = None,
    numerical_distributions: dict[str, str] | None = None,
    pairs: PairsReport | None = None,
) -> list[str]:
    """Geef concrete verbeteradviezen, gericht op de slechtst scorende kolommen.

    Leeg bij goede kwaliteit. Elk advies koppelt een gemeten signaal (verkeerd
    kolomtype, verloren pieken, hoge cardinaliteit, te weinig rijen, privacyrisico,
    omgeklapte correlatie) aan een actie in de app. Markdown-opmaak, bedoeld als
    bulletlijst.

    *numerical_distributions* is de actieve per-kolom verdeling, zodat een advies
    niet aanraadt wat al aanstaat (bv. ``gaussian_kde``). *pairs* voegt advies toe
    over afwijkende verbanden tussen kolommen.
    """
    from edu_synth.core.synthesize import infer_column_hints

    advice: list[str] = []

    if priv is not None and priv.available and priv.risk_level == "hoog":
        advice.append(
            "**Privacy** — hoog risico op herleidbaarheid. Markeer identifiers "
            "(studentnummer, e-mail) als type *ID*, of genereer minder rijen."
        )

    hints = {h.name: h for h in infer_column_hints(real) if h.has_suggestion}
    modal_cols = {f["column"] for f in report.modal_flags}
    scored = [r for r in report.rows if "score" in r]
    failing = sorted(
        (r for r in scored if not r.get("ok", True)), key=lambda r: r["score"], reverse=True
    )

    for r in failing:
        if len(advice) >= _MAX_ADVICE:
            break
        col = r["column"]
        if col in hints:
            suggestion = {"categorical": "categorisch", "datetime": "een datum"}.get(
                hints[col].suggested_sdtype, hints[col].suggested_sdtype
            )
            advice.append(
                f"**{col}** lijkt {suggestion} maar wordt anders behandeld — pas het type "
                "aan onder *Kolomtypes aanpassen*."
            )
        elif col in modal_cols:
            if (numerical_distributions or {}).get(col) == "gaussian_kde":
                advice.append(
                    f"**{col}** is multimodaal en vervlakt ondanks de al actieve "
                    "*gaussian_kde*-verdeling. Met weinig rijen blijft dit lastig — meer data "
                    "helpt, of accepteer de afwijking."
                )
            else:
                advice.append(
                    f"**{col}** heeft meerdere pieken die in de synthese vervlakken. Probeer een "
                    "andere verdeling (bijv. *gaussian_kde*) onder *Verdelingen*."
                )
        elif r["dtype"] == "categorical" and real[col].nunique() > _HIGH_CARDINALITY:
            advice.append(
                f"**{col}** heeft veel unieke waarden ({real[col].nunique()}). Overweeg de "
                "kolom weg te laten of waarden te groeperen."
            )

    if pairs is not None and len(advice) < _MAX_ADVICE and correlation_risk(pairs) != "laag":
        worst = pairs.flagged[0]  # gesorteerd op delta
        flipped = worst["real_corr"] * worst["synth_corr"] < 0
        kind = "is omgeklapt" if flipped else "wijkt af"
        advice.append(
            f"Het verband tussen **{worst['col_a']}** en **{worst['col_b']}** {kind} "
            f"(echt {worst['real_corr']:+.2f}, synthetisch {worst['synth_corr']:+.2f}). "
            "Correlaties zijn lastig direct te sturen — controleer of je analyses op dit "
            "verband leunen; meer trainingsdata helpt soms."
        )

    if len(real) < _SMALL_DATASET and len(advice) < _MAX_ADVICE:
        advice.append(
            f"De dataset is klein ({len(real)} rijen). Onder {_SMALL_DATASET} rijen heeft de "
            "synthesizer weinig houvast; meer data geeft stabielere resultaten."
        )

    return advice[:_MAX_ADVICE]


# ── Validatierapport-export ──────────────────────────────────────────────────────


def build_validation_report(
    *,
    report: Report,
    priv: PrivacyReport,
    sdm: SDMetricsReport,
    recommendation: str,
    synthesizer: str,
    n_training_rows: int,
    n_generated_rows: int,
    sdv_version: str,
    generated_at: str,
    random_seed: int | None = None,
    intended_use: str | None = None,
) -> dict:
    """Stel een machine-leesbaar validatierapport samen voor latere verantwoording.

    Bundelt alle scores die anders alleen in de UI zichtbaar zijn (per-kolom
    afstanden, sdmetrics, DCR/NNDR) plus de synthese-parameters, zodat een
    download het hele oordeel reproduceerbaar vastlegt.
    """
    privacy = {"available": priv.available}
    if priv.available:
        privacy.update(
            {
                "dcr_ratio": priv.dcr_ratio,
                "nndr_median": priv.nndr_median,
                "risk_level": priv.risk_level,
                "n_cols": priv.n_cols,
            }
        )
    else:
        privacy["reason"] = priv.reason

    sdmetrics: dict = {"available": sdm.available}
    if sdm.available:
        sdmetrics.update(
            {
                "overall_score": sdm.overall_score,
                "column_shapes": sdm.column_shapes,
                "column_pair_trends": sdm.column_pair_trends,
            }
        )
    else:
        sdmetrics["reason"] = sdm.reason

    return {
        "generated_at": generated_at,
        "sdv_version": sdv_version,
        "synthesizer": synthesizer,
        "n_training_rows": n_training_rows,
        "n_generated_rows": n_generated_rows,
        "random_seed": random_seed,
        "intended_use": intended_use,
        "column_stats": [dict(row) for row in report.rows],
        "sdmetrics": sdmetrics,
        "privacy": privacy,
        "usage_recommendation": recommendation,
        "disclaimer": RECOMMENDATION_DISCLAIMER,
    }
