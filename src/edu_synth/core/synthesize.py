"""Core synthesis functions — fit, sample, column hint detection."""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sdv.cag import FixedCombinations, Inequality
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

_DTYPE_TO_SDTYPE: dict[str, str] = {
    "categorical": "categorical",
    "integer": "numerical",
    "float": "numerical",
    "string": "id",
    "date": "datetime",
}

# Patroon → strftime-formaat, zodat een herkende datumkolom het juiste
# datetime_format meekrijgt richting SDV (Nederlandse onderwijsdata gebruikt vaak
# YYYYMMDD, het DUO-formaat).
_DATE_FORMATS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^\d{8}$"), "%Y%m%d"),  # YYYYMMDD
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), "%Y-%m-%d"),  # YYYY-MM-DD
    (re.compile(r"^\d{2}-\d{2}-\d{4}$"), "%d-%m-%Y"),  # DD-MM-YYYY
]
_DATE_PATTERNS = [pattern for pattern, _ in _DATE_FORMATS]
_DEFAULT_DATETIME_FORMAT = "%Y-%m-%d"


# ── Kolomtype-hints ────────────────────────────────────────────────────────────


@dataclass
class ColumnHint:
    """Suggestie voor een mogelijk verkeerd gedetecteerd kolomtype."""

    name: str
    detected_sdtype: str
    suggested_sdtype: str
    reason: str
    confidence: float  # 0 = alleen waarschuwing; >0 = type-suggestie

    @property
    def has_suggestion(self) -> bool:
        return self.suggested_sdtype != self.detected_sdtype and self.confidence > 0


def infer_column_hints(df: pd.DataFrame) -> list[ColumnHint]:
    """Detecteer potentieel verkeerde kolomtypes en geef correctiesuggesties.

    Heuristieken (generiek — geen hardcoded kolomnamen):
    - Integer ≤ 15 unieke waarden en max ≤ 100  → waarschijnlijk categorisch
    - Integer in bereik 1000–9999               → mogelijke postcode/code
    - String met datumspatroon                  → datetime
    - Kolom met > 30% missende waarden          → waarschuwing
    """
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)

    hints: list[ColumnHint] = []
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue

        detected = meta.columns.get(col, {}).get("sdtype", "categorical")
        miss_pct = df[col].isna().mean()

        if pd.api.types.is_integer_dtype(series):
            n_unique = int(series.nunique())
            min_val, max_val = int(series.min()), int(series.max())

            # Numeriek maar weinig unieke kleine waarden → waarschijnlijk categorische code
            if detected == "numerical" and n_unique <= 15 and max_val <= 100:
                hints.append(
                    ColumnHint(
                        name=col,
                        detected_sdtype="numerical",
                        suggested_sdtype="categorical",
                        reason=f"{n_unique} unieke waarden ≤ 100 — mogelijk een code of klasse",
                        confidence=0.9 if n_unique <= 5 else 0.65,
                    )
                )
                continue

            # Postcode-bereik: ook triggeren als SDV het als id detecteert
            if detected in ("numerical", "id") and 1000 <= min_val and max_val <= 9999:
                hints.append(
                    ColumnHint(
                        name=col,
                        detected_sdtype=detected,
                        suggested_sdtype="categorical",
                        reason="Waarden 1000–9999 — mogelijke postcode of ID-code",
                        confidence=0.7,
                    )
                )
                continue

        if detected in ("categorical", "id") and pd.api.types.is_object_dtype(series):
            sample = series.head(20).tolist()
            fmt = _detect_date_format(sample)
            if fmt is not None:
                hints.append(
                    ColumnHint(
                        name=col,
                        detected_sdtype=detected,
                        suggested_sdtype="datetime",
                        reason=f"Patroon lijkt op een datum (formaat {fmt})",
                        confidence=0.8,
                    )
                )
                continue

        if miss_pct > 0.3:
            hints.append(
                ColumnHint(
                    name=col,
                    detected_sdtype=detected,
                    suggested_sdtype=detected,
                    reason=f"{miss_pct:.0%} missende waarden — controleer of dit structureel is",
                    confidence=0.0,
                )
            )

    return hints


def _detect_date_format(values: list) -> str | None:
    """Geef het strftime-formaat als ≥70% van de stringsample één patroon volgt."""
    sample = [v for v in values[:10] if isinstance(v, str)]
    if len(sample) < 3:
        return None
    for pattern, fmt in _DATE_FORMATS:
        matches = sum(1 for v in sample if pattern.match(v.strip()))
        if matches >= len(sample) * 0.7:
            return fmt
    return None


def _looks_like_date(values: list) -> bool:
    return _detect_date_format(values) is not None


def detect_datetime_format(series: pd.Series) -> str | None:
    """Detecteer het strftime-formaat van een (string) datumkolom, of None.

    Gebruikt door de app om datumkolommen het juiste ``datetime_format`` mee te
    geven zonder schema — de app kent immers geen schemabestand.
    """
    return _detect_date_format(series.dropna().head(20).tolist())


# ── Per-kolom distributie-aanbeveling ────────────────────────────────────────────
# GaussianCopula modelleert elke numerieke kolom met een marginale verdeling. De
# default 'norm' faalt hard op scheve of zero-inflated kolommen (bv. capital-gain:
# 92% nullen): de gefitte normaal trekt dan onmogelijke waarden en de afstand tot
# de echte data loopt op. 'gaussian_kde' volgt de empirische vorm en lost dat op,
# tegen wat extra rekentijd. We zetten KDE daarom gericht op de kolommen die het
# nodig hebben — globaal toepassen is traag en geheugenintensief.
_KDE = "gaussian_kde"
_SKEW_THRESHOLD = 2.0  # |scheefheid| hierboven → een marginale normaal past slecht
_MODE_FREQ_THRESHOLD = 0.5  # één waarde domineert → zero-inflated/multimodaal
_MIN_UNIQUE_FOR_KDE = 20  # te weinig unieke waarden → feitelijk discreet, KDE zinloos

# SDV-GaussianCopula marginale verdelingen, in oplopende complexiteit.
DISTRIBUTION_CHOICES: list[str] = ["norm", "beta", "truncnorm", "uniform", "gamma", _KDE]


def is_skewed(series: pd.Series) -> bool:
    """Is *series* zo scheef/zero-inflated dat een marginale normaal slecht past?

    True bij hoge absolute scheefheid óf wanneer één waarde de kolom domineert,
    mits er genoeg unieke waarden zijn om KDE zinvol te maken (anders is de kolom
    feitelijk discreet/categorisch).
    """
    s = series.dropna()
    if s.nunique() < _MIN_UNIQUE_FOR_KDE:
        return False
    mode_freq = float(s.value_counts(normalize=True).iloc[0])
    return abs(float(s.skew())) >= _SKEW_THRESHOLD or mode_freq >= _MODE_FREQ_THRESHOLD


def recommend_numerical_distributions(
    df: pd.DataFrame, numerical_columns: list[str]
) -> dict[str, str]:
    """Kies per scheve/zero-inflated numerieke kolom ``gaussian_kde`` als marginale.

    De overige kolommen blijven op de SDV-default ('norm') en komen niet in het
    resultaat. ``numerical_columns`` bepaalt welke kolommen als numeriek gelden,
    zodat als categorisch getypeerde codes buiten beschouwing blijven.
    """
    return {
        col: _KDE
        for col in numerical_columns
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and is_skewed(df[col])
    }


def set_seed(seed: int) -> None:
    """Maak synthese reproduceerbaar door numpy/random te seeden vóór ``fit``.

    SDV 1.17 biedt geen seed-parameter in de constructor; GaussianCopula en PAR
    gebruiken de globale numpy-randomstate. Een ``np.random.seed()`` vóór ``fit()``
    levert daarom bij gelijke seed + data identieke synthetische output.
    """
    random.seed(seed)
    np.random.seed(seed)


def fit(
    data: pd.DataFrame,
    schema_path: Path | None = None,
    seed: int | None = None,
    numerical_distributions: dict[str, str] | None = None,
) -> GaussianCopulaSynthesizer:
    """Train a synthesizer on *data*.

    Parameters
    ----------
    data:
        Real dataset to learn from.
    schema_path:
        Path to a YAML schema file. If omitted, SDV auto-detects column types.
    seed:
        Optional random seed. When set, makes the generated output reproducible
        for identical input data.
    numerical_distributions:
        Per-column marginal distribution for GaussianCopula. When ``None`` (the
        default), skewed/zero-inflated columns are detected automatically and get
        ``gaussian_kde``; the rest keep SDV's default. Pass an explicit dict to
        override, or ``{}`` to disable the auto-detection entirely. A
        ``distribution`` field per column in the YAML schema takes precedence.

    Returns
    -------
    Fitted SDV synthesizer — pass to :func:`sample` to generate rows.
    """
    schema: dict | None = None
    if schema_path is not None:
        schema = _load_schema(schema_path)
        metadata = _build_metadata(schema)
    else:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)

    if numerical_distributions is None:
        num_cols = [c for c, info in metadata.columns.items() if info.get("sdtype") == "numerical"]
        numerical_distributions = recommend_numerical_distributions(data, num_cols)
        if schema is not None:  # expliciete schema-keuze overschrijft de aanbeveling
            numerical_distributions = {**numerical_distributions, **_schema_distributions(schema)}

    if seed is not None:
        set_seed(seed)
    synthesizer = GaussianCopulaSynthesizer(
        metadata, numerical_distributions=numerical_distributions or None
    )

    if schema is not None:
        constraints = _build_constraints(schema)
        if constraints:
            try:
                synthesizer.add_constraints(constraints=constraints)
            except Exception as exc:  # SDV-validatie: conflicterende/ongeldige regels
                raise ValueError(
                    f"Constraints uit het schema konden niet worden toegepast: {exc}. "
                    "Controleer of de kolommen bestaan en de regels niet botsen met de data."
                ) from exc

    synthesizer.fit(data)
    return synthesizer


def sample(model: Any, n_rows: int) -> pd.DataFrame:
    """Generate *n_rows* synthetic rows from a fitted *model*."""
    return model.sample(num_rows=n_rows)


# ── Interne helpers ────────────────────────────────────────────────────────────


def _load_schema(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _schema_distributions(schema: dict) -> dict[str, str]:
    """Lees een optioneel ``distribution``-veld per kolom uit het YAML-schema."""
    return {
        name: col["distribution"]
        for name, col in schema.get("columns", {}).items()
        if col.get("distribution")
    }


def _build_constraints(schema: dict) -> list:
    """Bouw SDV-cag-constraints uit het optionele ``constraints``-blok in het schema."""
    return build_constraints(schema.get("constraints", []))


def build_constraints(rules: list[dict]) -> list:
    """Vertaal rule-dicts naar SDV-cag-constraints.

    Cross-kolom-regels die SDV niet uit de data afleidt. Ondersteund:

    - ``inequality`` — ``low ≤ high`` (``strict: true`` voor strikt ``<``,
      standaard ``false`` zodat gelijk mag).
    - ``fixed_combinations`` — alleen in de data voorkomende combinaties van de
      opgegeven categorische kolommen.

    Dezelfde rule-vorm als het ``constraints``-blok in het YAML-schema, zodat de
    app (point-and-click) en de CLI (schema) één vertaalpad delen. Een onbekend
    ``type`` of ontbrekende sleutel levert een ``ValueError`` op.
    """
    constraints: list = []
    for i, rule in enumerate(rules, start=1):
        rule_type = rule.get("type")
        if rule_type == "inequality":
            try:
                low, high = rule["low"], rule["high"]
            except KeyError as exc:
                raise ValueError(
                    f"Constraint {i} (inequality) mist een verplichte sleutel: {exc}. "
                    "Vereist: 'low' en 'high'."
                ) from exc
            constraints.append(
                Inequality(
                    low_column_name=low,
                    high_column_name=high,
                    strict_boundaries=bool(rule.get("strict", False)),
                )
            )
        elif rule_type == "fixed_combinations":
            try:
                columns = rule["columns"]
            except KeyError as exc:
                raise ValueError(
                    f"Constraint {i} (fixed_combinations) mist verplichte sleutel: {exc}. "
                    "Vereist: 'columns'."
                ) from exc
            constraints.append(FixedCombinations(column_names=list(columns)))
        else:
            raise ValueError(
                f"Constraint {i} heeft onbekend type {rule_type!r}. "
                "Ondersteund: 'inequality', 'fixed_combinations'."
            )
    return constraints


def _build_metadata(schema: dict) -> SingleTableMetadata:
    metadata = SingleTableMetadata()
    for col_name, col in schema.get("columns", {}).items():
        sdtype = _DTYPE_TO_SDTYPE.get(col.get("dtype", "categorical"), "categorical")
        kwargs: dict[str, Any] = {"sdtype": sdtype}
        if col.get("role") == "primary_key":
            metadata.add_column(col_name, sdtype="id")
            metadata.set_primary_key(col_name)
            continue
        if sdtype == "datetime":
            # Zonder expliciet formaat valt SDV terug op ISO 8601 en faalt op
            # DUO-datums (YYYYMMDD). Schema mag het formaat overschrijven.
            kwargs["datetime_format"] = col.get("datetime_format", _DEFAULT_DATETIME_FORMAT)
        metadata.add_column(col_name, **kwargs)
    return metadata


# ── Sequentieel / longitudinaal ──────────────────────────────────────────────────

_SEQ_INDEX_NAME_HINTS = ("jaar", "year", "datum", "date", "periode", "tijd", "maand", "kwartaal")
_SEQ_KEY_NAME_HINTS = ("id", "nummer", "sleutel", "key", "student", "instelling", "pgn", "bsn")


def infer_sequence_columns(df: pd.DataFrame) -> tuple[bool, str | None, str | None]:
    """Raad of *df* longitudinaal is en welke kolommen sequence key/index zijn.

    Heuristiek (voor goede defaults — de gebruiker kan altijd corrigeren):
    - sequence index = de **tijdkolom**: numeriek/datum, met de minste unieke
      waarden (de tijdstappen), bij voorkeur een tijd-naam (jaar, datum, …).
    - sequence key = de **entiteit**: een kolom die zich herhaalt (≥ 2 rijen per
      waarde) met meer unieke waarden dan de index, bij voorkeur een ID-naam.

    Retourneert ``(lijkt_longitudinaal, seq_key, seq_index)``.
    """
    n = len(df)
    counts = {c: df[c].nunique(dropna=True) for c in df.columns}
    repeating = {c: u for c, u in counts.items() if 1 < u < n and n / u >= 2}

    index_candidates = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        or pd.api.types.is_datetime64_any_dtype(df[c])
        or detect_datetime_format(df[c]) is not None
    ]

    def _index_rank(col: str) -> tuple[int, int]:
        named = 0 if any(h in col.lower() for h in _SEQ_INDEX_NAME_HINTS) else 1
        return (named, counts[col])  # tijd-naam eerst, dan minste tijdstappen

    seq_index = min(index_candidates, key=_index_rank) if index_candidates else None

    def _key_rank(col: str) -> tuple[int, int]:
        named = 0 if any(h in col.lower() for h in _SEQ_KEY_NAME_HINTS) else 1
        return (named, -counts[col])  # ID-naam eerst, dan meeste entiteiten

    key_candidates = [c for c in repeating if c != seq_index]
    seq_key = min(key_candidates, key=_key_rank) if key_candidates else None

    return (seq_key is not None and seq_index is not None), seq_key, seq_index


def build_sequential_metadata(df: pd.DataFrame, seq_key: str, seq_index: str) -> Any:
    """Bouw SDV-metadata voor longitudinale data uit een geüploade tabel.

    SDV eist dat de sequence index ``numerical`` of ``datetime`` is, anders crasht
    ``set_sequence_index``. We forceren daarom het juiste type op key en index.
    """
    from sdv.metadata import Metadata

    metadata = Metadata.detect_from_dataframe(df, table_name="data")
    metadata.update_column(seq_key, sdtype="id", table_name="data")

    index_col = df[seq_index]
    if pd.api.types.is_numeric_dtype(index_col):
        metadata.update_column(seq_index, sdtype="numerical", table_name="data")
    elif pd.api.types.is_datetime64_any_dtype(index_col):
        metadata.update_column(seq_index, sdtype="datetime", table_name="data")
    elif (fmt := detect_datetime_format(index_col)) is not None:
        metadata.update_column(seq_index, sdtype="datetime", datetime_format=fmt, table_name="data")
    else:
        metadata.update_column(seq_index, sdtype="numerical", table_name="data")

    metadata.set_sequence_key(seq_key, table_name="data")
    metadata.set_sequence_index(seq_index, table_name="data")
    return metadata


# ── Lichte sequentiële synthesizer (wide + GaussianCopula) ──────────────────────
#
# PAR (deep learning) is op CPU minutenlang en matig op kleine onderwijsdatasets.
# In plaats daarvan zetten we de longitudinale data plat (één rij per entiteit,
# kolommen ``feature__tN`` per tijdstap) en laten we de bestaande GaussianCopula die
# leren — inclusief de cross-tijd-correlaties (dus doorstroomkansen) en een expliciete
# reekslengte. Bij sampling reconstrueren we het originele long-format terug.
#
# Twee reconstructie-regels houden de reeksen geldig:
#   1. De reekslengte komt uit een meegemodelleerde ``__seq_len__``-kolom, niet uit
#      het (ruizige) NaN-patroon — zo blijft de lengteverdeling kloppen.
#   2. Een reeks stopt bij een *terminale* staat (een categorie die in de echte data
#      nooit een opvolger heeft, bv. gediplomeerd/uitgestroomd) — zo ontstaan geen
#      onmogelijke paden (actieve staat ná een eindstaat).

_SEQ_LEN_COL = "__seq_len__"


@dataclass
class SequentialCopulaModel:
    """Gefitte lichte sequentiële synthesizer. Geef door aan :func:`sample_sequential`."""

    copula: GaussianCopulaSynthesizer
    seq_key: str
    seq_index: str
    original_columns: list[str]
    feature_cols: list[str]
    feature_dtypes: dict[str, Any]
    index_levels: list  # geordende originele tijd-waarden; positie t → index_levels[t-1]
    index_dtype: Any
    terminal: dict[str, set]  # per categorische kolom: waarden die een reeks beëindigen
    fallback: dict[str, Any]  # per kolom: waarde als forward-fill niets heeft (mode/mediaan)
    max_len: int


def _learn_terminal_states(
    df: pd.DataFrame, seq_key: str, seq_index: str, cat_cols: list[str]
) -> dict[str, set]:
    """Leer per categorische kolom welke waarden *terminaal* zijn.

    Een waarde is terminaal als ze in de echte data nooit een opvolgende rij binnen
    dezelfde entiteit heeft — precies het gedrag van een absorberende staat
    (gediplomeerd, uitgestroomd). Zo hoeven we die staten niet hard te coderen.
    """
    ordered = df.sort_values([seq_key, seq_index])
    terminal: dict[str, set] = {}
    for col in cat_cols:
        seen: set = set()
        with_successor: set = set()
        for _, g in ordered.groupby(seq_key, sort=False):
            vals = g[col].tolist()
            for v in vals:
                if pd.notna(v):
                    seen.add(v)
            for v in vals[:-1]:  # alles behalve de laatste rij heeft een opvolger
                if pd.notna(v):
                    with_successor.add(v)
        terminal[col] = seen - with_successor
    return terminal


def _to_wide(
    df: pd.DataFrame,
    seq_key: str,
    seq_index: str,
    feature_cols: list[str],
    pos: dict,
    max_len: int,
) -> pd.DataFrame:
    """long → wide: één rij per entiteit, kolommen ``feature__tN`` + ``__seq_len__``."""
    records = []
    for _, g in df.groupby(seq_key, sort=False):
        g = g.sort_values(seq_index)
        row: dict = {}
        for _, r in g.iterrows():
            t = pos[r[seq_index]]
            for feat in feature_cols:
                row[f"{feat}__t{t}"] = r[feat]
        row[_SEQ_LEN_COL] = len(g)
        records.append(row)

    columns = [f"{feat}__t{t}" for t in range(1, max_len + 1) for feat in feature_cols]
    wide = pd.DataFrame(records).reindex(columns=[*columns, _SEQ_LEN_COL])
    # Numerieke features: forceer numeriek zodat SDV ze als 'numerical' detecteert
    # (de pivot met gemengde NaN maakt er anders object van).
    for feat in feature_cols:
        if pd.api.types.is_numeric_dtype(df[feat]):
            for t in range(1, max_len + 1):
                wide[f"{feat}__t{t}"] = pd.to_numeric(wide[f"{feat}__t{t}"], errors="coerce")
    return wide


def fit_sequential(
    df: pd.DataFrame, seq_key: str, seq_index: str, seed: int | None = None
) -> SequentialCopulaModel:
    """Train de lichte sequentiële synthesizer op longitudinale *df* (long-format).

    *seq_key* is de entiteit (bv. studentnummer), *seq_index* de tijd-as (bv.
    studiejaar). Fit en sampling draaien in seconden op CPU — geschikt voor lokale
    onderwijs-apparatuur zonder GPU.
    """
    feature_cols = [c for c in df.columns if c not in (seq_key, seq_index)]
    index_levels = sorted(df[seq_index].dropna().unique().tolist())
    pos = {lvl: i + 1 for i, lvl in enumerate(index_levels)}
    max_len = len(index_levels)

    # Generieke vormchecks: blokkeer alleen data die deze aanpak echt niet aankan,
    # ongeacht de dataset. (1) Zonder herhaalde entiteiten of tijdstappen is het
    # niet longitudinaal. (2) Wordt de wide-tabel breder dan het aantal entiteiten
    # (features × tijdstappen ≥ entiteiten), dan is de correlatiematrix onderbepaald
    # en levert de synthese onbetrouwbare verbanden — beter weigeren dan misleiden.
    n_entities = df[seq_key].nunique()
    if max_len < 2 or n_entities < 2:
        raise ValueError(
            "Deze data is niet longitudinaal genoeg: er zijn te weinig tijdstappen "
            f"({max_len}) of entiteiten ({n_entities}). Kies een dataset met meerdere "
            "rijen per entiteit over de tijd."
        )
    n_wide_cols = len(feature_cols) * max_len
    if n_wide_cols >= n_entities:
        raise ValueError(
            f"Te veel kolommen voor te weinig entiteiten: {len(feature_cols)} kolommen × "
            f"{max_len} tijdstappen = {n_wide_cols} dimensies bij {n_entities} entiteiten. "
            "De verbanden worden dan onbetrouwbaar. Laat kolommen weg, gebruik een dataset "
            "met meer entiteiten, of kies onder 'Synthesizer kiezen' de PAR-synthesizer — "
            "die verwerkt lange reeksen direct zonder deze beperking."
        )

    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    terminal = _learn_terminal_states(df, seq_key, seq_index, cat_cols)

    fallback: dict[str, Any] = {}
    for feat in feature_cols:
        col = df[feat].dropna()
        if col.empty:
            fallback[feat] = None
        elif pd.api.types.is_numeric_dtype(df[feat]):
            fallback[feat] = col.median()
        else:
            fallback[feat] = col.mode().iloc[0]

    wide = _to_wide(df, seq_key, seq_index, feature_cols, pos, max_len)
    copula = fit(wide, seed=seed)

    return SequentialCopulaModel(
        copula=copula,
        seq_key=seq_key,
        seq_index=seq_index,
        original_columns=list(df.columns),
        feature_cols=feature_cols,
        feature_dtypes={c: df[c].dtype for c in feature_cols},
        index_levels=index_levels,
        index_dtype=df[seq_index].dtype,
        terminal=terminal,
        fallback=fallback,
        max_len=max_len,
    )


def _is_missing(val: Any) -> bool:
    return val is None or (isinstance(val, float) and pd.isna(val))


def _coerce_like(s: pd.Series, dtype: Any) -> pd.Series:
    """Zet *s* strak in *dtype* terug, zonder gemengde types over te houden.

    Numerieke doel-dtype → altijd numeriek (zodat de kolom net als de echte data als
    'numeriek' herkend wordt en niet half object blijft). Niet-numeriek → exact het
    echte dtype; lukt dat niet, dan uniform string (één type i.p.v. int/str-mix, wat
    downstream het samenvoegen van verdelingen laat crashen).
    """
    if pd.api.types.is_numeric_dtype(dtype):
        num = pd.to_numeric(s, errors="coerce")
        if pd.api.types.is_integer_dtype(dtype) and not num.isna().any():
            return num.astype(dtype)
        return num
    try:
        return s.astype(dtype)
    except (ValueError, TypeError):
        return s.astype(str)


def _first_terminal_pos(row: pd.Series, model: SequentialCopulaModel) -> int | None:
    """Eerste tijdstap (1-based) waarop een gesampelde staat terminaal is, of ``None``."""
    for t in range(1, model.max_len + 1):
        for feat, terms in model.terminal.items():
            if not terms:
                continue
            val = row.get(f"{feat}__t{t}")
            if not _is_missing(val) and val in terms:
                return t
    return None


def sample_sequential(model: SequentialCopulaModel, n_sequences: int) -> pd.DataFrame:
    """Genereer *n_sequences* synthetische reeksen, terug in het originele long-format."""
    synth_wide = sample(model.copula, n_sequences)
    rows: list[dict] = []

    for new_id, (_, r) in enumerate(synth_wide.iterrows(), start=1):
        # Bepaal de reekslengte. Een eindstaat (gediplomeerd/uitgestroomd) is leidend:
        # de reeks stopt daar. Komt er geen eindstaat voor, dan is de reeks gecensureerd
        # (bv. nog ingeschreven) en gebruiken we de meegemodelleerde ``__seq_len__``.
        terminal_pos = _first_terminal_pos(r, model)
        if terminal_pos is not None:
            k = terminal_pos
        else:
            raw_len = r.get(_SEQ_LEN_COL)
            k = model.max_len if _is_missing(raw_len) else int(round(float(raw_len)))
        k = max(1, min(k, model.max_len))

        last: dict[str, Any] = {feat: None for feat in model.feature_cols}
        for t in range(1, k + 1):
            record = {model.seq_key: new_id, model.seq_index: model.index_levels[t - 1]}
            for feat in model.feature_cols:
                val = r.get(f"{feat}__t{t}")
                if _is_missing(val):  # gat → draag laatst bekende (of fallback) door
                    val = last[feat] if not _is_missing(last[feat]) else model.fallback[feat]
                last[feat] = val
                record[feat] = val
            rows.append(record)

    out = pd.DataFrame(rows)
    for feat, dtype in model.feature_dtypes.items():
        out[feat] = _coerce_like(out[feat], dtype)
    out[model.seq_index] = _coerce_like(out[model.seq_index], model.index_dtype)
    return out.reindex(columns=model.original_columns)


# ── PAR (deep learning) — optionele zwaardere synthesizer ────────────────────────
#
# PAR is SDV's neurale sequentiële synthesizer (LSTM). Structureel trager dan de
# lichte copula (op CPU minuten i.p.v. seconden), maar kan complexere temporele
# patronen leren. We bieden 'm als bewuste keuze naast fit_sequential; de copula
# blijft de aanbevolen default (zie issue #77).


@contextmanager
def _par_progress(callback: Callable[[float], None] | None):
    """Rapporteer PAR-trainingsvoortgang per epoch via *callback* (fractie 0–1).

    PAR (deepecho) heeft geen callback-API; de enige voortgangsbron is de interne
    ``tqdm`` over de epochs. We vervangen die tijdelijk door een shim die per epoch
    ``callback(voltooide_epoch / totaal)`` aanroept en de originele ``tqdm`` daarna
    weer terugzet. Zonder callback doet dit niets (geen patch).
    """
    if callback is None:
        yield
        return

    import deepecho.models.par as parmod

    original_tqdm = parmod.tqdm

    class _ProgressTqdm:
        def __init__(self, iterable=None, **_kwargs):
            self._items = list(iterable) if iterable is not None else []
            self._total = len(self._items) or 1

        def __iter__(self):
            for i, item in enumerate(self._items, start=1):
                callback(i / self._total)
                yield item

        def set_description(self, *_args, **_kwargs):
            pass

    parmod.tqdm = _ProgressTqdm
    try:
        yield
    finally:
        parmod.tqdm = original_tqdm


def fit_par(
    df: pd.DataFrame,
    seq_key: str,
    seq_index: str,
    epochs: int = 128,
    seed: int | None = None,
    progress: Callable[[float], None] | None = None,
) -> Any:
    """Train SDV's ``PARSynthesizer`` (deep learning) op longitudinale *df*.

    Zwaarder dan :func:`fit_sequential` (LSTM op CPU) maar kan complexere temporele
    patronen leren. *progress* is een optionele callback die per epoch de voltooide
    fractie (0–1) krijgt — de app koppelt die aan een voortgangsbalk.
    """
    from sdv.sequential import PARSynthesizer

    metadata = build_sequential_metadata(df, seq_key, seq_index)
    if seed is not None:
        set_seed(seed)
    synthesizer = PARSynthesizer(metadata, epochs=epochs, verbose=False)
    with _par_progress(progress):
        synthesizer.fit(df)
    return synthesizer


def sample_par(model: Any, n_sequences: int) -> pd.DataFrame:
    """Genereer *n_sequences* synthetische reeksen met een gefitte ``PARSynthesizer``."""
    return model.sample(num_sequences=n_sequences)
