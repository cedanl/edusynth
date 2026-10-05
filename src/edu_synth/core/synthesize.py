"""Core synthesis functions — fit, sample, column hint detection."""

from __future__ import annotations

import math
import random
import re
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
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
# kolommen ``feature__tN`` per stap) en laten we de bestaande GaussianCopula die
# leren — inclusief de cross-tijd-correlaties (dus doorstroomkansen) en een expliciete
# reekslengte. Bij sampling reconstrueren we het originele long-format terug.
#
# Stappen zijn relatief: stap 1 is de eerste waarneming van een entiteit, niet het
# eerste tijdniveau van de dataset. Het startmoment (``__seq_start__``, positie in de
# geordende tijdniveaus) en de afstand tot de vorige stap (``__dt__tN``) worden
# meegemodelleerd, zodat gespreide instroom en onderbrekingen behouden blijven.
#
# Twee reconstructie-regels houden de reeksen geldig:
#   1. De reekslengte komt uit een meegemodelleerde ``__seq_len__``-kolom, niet uit
#      het (ruizige) NaN-patroon — zo blijft de lengteverdeling kloppen.
#   2. Een reeks stopt bij een *terminale* staat (een waarde die in de echte data
#      nooit een opvolger heeft, bv. gediplomeerd/uitgestroomd of een 0/1-eindvlag,
#      in een tekst- of getalkolom met weinig verschillende waarden) — zo ontstaan geen
#      onmogelijke paden (actieve staat ná een eindstaat).

_SEQ_LEN_COL = "__seq_len__"
# Een geleerde reeksregel geldt alleen als die in (vrijwel) alle echte gevallen opgaat.
_RULE_COVERAGE = 0.99
# Kans waaronder een patroon niet meer als toeval geldt.
_CHANCE_LEVEL = 0.01
_SEQ_START_COL = "__seq_start__"
_DT_PREFIX = "__dt__t"
_AXIS_COL = "__axis__"
_DAY = pd.Timedelta(1, unit="D")
_EPOCH = pd.Timestamp(0)


@dataclass
class SequentialCopulaModel:
    """Gefitte lichte sequentiële synthesizer. Geef door aan :func:`sample_sequential`."""

    copula: GaussianCopulaSynthesizer
    seq_key: str
    seq_index: str
    original_columns: list[str]
    feature_cols: list[str]
    feature_dtypes: dict[str, Any]
    index_kind: str  # 'numeric', 'datetime' of 'label'
    index_format: str | None  # strftime-formaat bij een datum als tekst
    index_levels: list  # geordende originele tijd-waarden
    level_axis: list[float]  # tijd-as-positie per niveau, zelfde volgorde
    index_dtype: Any
    dt_mode: float  # meest voorkomende stapafstand, voor een ontbrekende gesamplede
    dt_min: float
    dt_max: float
    dt_integer: bool  # echte stapafstanden zijn gehele getallen → afronden
    terminal: dict[str, set]  # per kolom: waarden die een reeks beëindigen
    patterns: SequencePatterns  # kolommen met een vaste relatie tot de reeks
    fallback: dict[str, Any]  # per kolom: waarde als forward-fill niets heeft (mode/mediaan)
    max_len: int  # langste reeks in de echte data


def _detect_index_kind(index: pd.Series) -> tuple[str, str | None]:
    """Soort tijd-as: ``numeric``, ``datetime`` of ``label``, plus het datumformaat."""
    if pd.api.types.is_numeric_dtype(index):
        return "numeric", None
    if pd.api.types.is_datetime64_any_dtype(index):
        return "datetime", None
    if (fmt := detect_datetime_format(index)) is not None:
        return "datetime", fmt
    return "label", None


def _to_axis(index: pd.Series, kind: str, fmt: str | None, labels: list) -> pd.Series:
    """Zet tijd-waarden om naar een numerieke as waarop afstanden zinvol zijn.

    Jaartal of getal: de waarde zelf. Datum: dagen. Tijdlabel (bv. ``2022-2023``):
    positie in de gesorteerde labels, dus de afstand telt in niveaus.
    """
    if kind == "numeric":
        return index.astype(float)
    if kind == "datetime":
        return (pd.to_datetime(index, format=fmt) - _EPOCH) / _DAY
    return index.map({lvl: i + 1 for i, lvl in enumerate(labels)}).astype(float)


def to_time_axis(index: pd.Series, like: pd.Series) -> pd.Series:
    """Zet *index* op de numerieke tijd-as die bij de tijdkolom *like* hoort.

    Soort, datumformaat en labelvolgorde komen uit *like* (de echte data), zodat
    echte en synthetische tijdkolommen op dezelfde as vergelijkbaar zijn.
    """
    kind, fmt = _detect_index_kind(like)
    labels = sorted(like.dropna().unique().tolist()) if kind == "label" else []
    return _to_axis(index, kind, fmt, labels)


def _from_axis(value: float, model: SequentialCopulaModel) -> Any:
    """Inverse van :func:`_to_axis` voor één gesamplede as-waarde."""
    if model.index_kind == "numeric":
        return value
    if model.index_kind == "datetime":
        ts = _EPOCH + value * _DAY
        return ts.strftime(model.index_format) if model.index_format else ts
    return model.index_levels[round(value) - 1]


def _is_discrete(values: pd.Series) -> bool:
    """Heeft de kolom weinig verschillende waarden? Continue getalkolommen niet."""
    return not (pd.api.types.is_numeric_dtype(values) and values.nunique() >= _MIN_UNIQUE_FOR_KDE)


def _values_confined_to(values: pd.Series, at_position: pd.Series) -> set:
    """Waarden die (vrijwel) alleen op de rijen van *at_position* voorkomen.

    Een waarde moet vaak genoeg voorkomen: staat ze toevallig een paar keer op zo'n
    rij, dan is dat geen regel. De minimale frequentie volgt uit de kans dat een
    willekeurige rij op die positie staat.
    """
    p_position = at_position.mean()
    if p_position >= 1:
        return set()
    min_support = math.ceil(math.log(_CHANCE_LEVEL) / math.log(p_position))
    counts = values.value_counts()
    elsewhere = values[~at_position].value_counts().reindex(counts.index, fill_value=0)
    keep = (counts >= min_support) & (elsewhere <= (1 - _RULE_COVERAGE) * counts)
    return set(counts.index[keep])


def detect_terminal_states(
    df: pd.DataFrame, seq_key: str, seq_index: str, columns: list[str]
) -> dict[str, set]:
    """Leer per kolom welke waarden een reeks beëindigen (*terminaal* zijn).

    Een waarde is terminaal als ze in de echte data (vrijwel) nooit een opvolgende
    rij binnen dezelfde entiteit heeft: het gedrag van een absorberende staat
    (gediplomeerd, uitgestroomd, een 0/1-eindvlag). Zo hoeven we die staten niet
    hard te coderen. Elke kolom met weinig verschillende waarden doet mee, ongeacht
    het datatype; continue numerieke kolommen niet.
    """
    ordered = df.sort_values([seq_key, seq_index])
    is_last = ordered.groupby(seq_key, sort=False).cumcount(ascending=False) == 0
    return {
        col: _values_confined_to(ordered[col], is_last)
        for col in columns
        if _is_discrete(ordered[col])
    }


@dataclass
class SequencePatterns:
    """Vaste relaties tussen kolommen en de reeks, geleerd uit de echte data."""

    constant: list[str] = field(default_factory=list)
    # kolom → ("step", c): +c per rij, of ("time", c): +c per eenheid op de tijd-as
    counters: dict[str, tuple[str, float]] = field(default_factory=dict)
    # kolom → (waarden alleen op de eerste rij, vervanging op latere rijen,
    #          vaste waarde voor de eerste rij of None)
    first_only: dict[str, tuple[set, Any, Any]] = field(default_factory=dict)


def _holds_per_entity(ok: pd.Series, entity: pd.Series) -> bool:
    """Geldt *ok* op alle rijen van (vrijwel) elke entiteit?"""
    return ok.groupby(entity).all().mean() >= _RULE_COVERAGE


def _detect_counter(
    values: pd.Series, dt: pd.Series, entity: pd.Series
) -> tuple[str, float] | None:
    """Loopt de kolom per rij (``step``) of per tijdseenheid (``time``) vast op?"""
    diff = values.groupby(entity).diff()
    has_prev = dt.notna()
    diff, dt, entity = diff[has_prev], dt[has_prev], entity[has_prev]
    for kind, unit in (("step", 1.0), ("time", dt)):
        rate = (diff / unit).mode()
        if rate.empty or rate.iloc[0] == 0:
            continue
        c = float(rate.iloc[0])
        if _holds_per_entity(pd.Series(np.isclose(diff, c * unit), index=diff.index), entity):
            return kind, c
    return None


def detect_sequence_patterns(
    df: pd.DataFrame, seq_key: str, seq_index: str, columns: list[str]
) -> SequencePatterns:
    """Leer welke kolommen een vaste relatie met de reeks hebben.

    Vier generieke patronen, elk alleen als het bij (vrijwel) alle entiteiten geldt:
    constant per entiteit, een teller die per rij of per tijdseenheid met een vaste
    waarde oploopt, en een waarde die alleen op de eerste rij voorkomt. Het vierde
    patroon, een waarde die alleen op de laatste rij voorkomt, is een eindstaat (zie
    :func:`detect_terminal_states`). Tellers per tijdseenheid vragen een numerieke
    *seq_index*.
    """
    ordered = df.sort_values([seq_key, seq_index])
    entity = ordered[seq_key]
    by_entity = ordered.groupby(seq_key, sort=False)
    is_first = by_entity.cumcount() == 0
    multi = by_entity[seq_key].transform("size") > 1
    dt = by_entity[seq_index].diff() if pd.api.types.is_numeric_dtype(ordered[seq_index]) else None
    patterns = SequencePatterns()
    if not multi.any():
        return patterns

    for col in columns:
        values = ordered[col]
        first = by_entity[col].transform("first")
        same_as_first = values.eq(first) | (values.isna() & first.isna())
        if _holds_per_entity(same_as_first[multi], entity[multi]):
            patterns.constant.append(col)
            continue
        if pd.api.types.is_numeric_dtype(values):
            step_dt = dt if dt is not None else pd.Series(1.0, index=values.index).where(~is_first)
            counter = _detect_counter(values, step_dt, entity)
            if counter is not None:
                patterns.counters[col] = counter
                continue
        if not _is_discrete(values):
            continue
        first_values = _values_confined_to(values, is_first)
        if first_values:
            later = values[~is_first & ~values.isin(first_values)]
            first_fill = values[is_first].mode()
            always = values[is_first].isin(first_values).mean() >= _RULE_COVERAGE
            patterns.first_only[col] = (
                first_values,
                later.mode().iloc[0] if not later.empty else None,
                first_fill.iloc[0] if always and not first_fill.empty else None,
            )
    return patterns


def _apply_sequence_patterns(
    out: pd.DataFrame, seq_key: str, axis_col: str, patterns: SequencePatterns
) -> pd.DataFrame:
    """Leid kolommen met een vast patroon opnieuw af uit de gesamplede reeks."""
    by_entity = out.groupby(seq_key, sort=False)
    step = by_entity.cumcount()
    is_first = step == 0
    for col in patterns.constant:
        out[col] = by_entity[col].transform("first")
    for col, (kind, c) in patterns.counters.items():
        start = pd.to_numeric(out[col], errors="coerce").groupby(out[seq_key]).transform("first")
        offset = step if kind == "step" else out[axis_col] - by_entity[axis_col].transform("first")
        out[col] = start + c * offset
    for col, (first_values, replacement, first_fill) in patterns.first_only.items():
        if replacement is not None:
            out.loc[~is_first & out[col].isin(first_values), col] = replacement
        if first_fill is not None:
            out.loc[is_first & ~out[col].isin(first_values), col] = first_fill
    return out


def _to_wide(ordered: pd.DataFrame, seq_key: str, feature_cols: list[str]) -> pd.DataFrame:
    """long → wide: één rij per entiteit met ``feature__tN``, ``__dt__tN``,
    ``__seq_start__`` en ``__seq_len__``.

    *ordered* is per entiteit op de tijd-as gesorteerd en heeft de hulpkolommen
    ``__step__`` (relatieve stap), ``__dt__`` (afstand tot vorige stap) en ``__start__``.
    """
    long = ordered.set_index([seq_key, "__step__"])
    wide_feat = long[feature_cols].unstack("__step__")
    wide_feat.columns = [f"{feat}__t{t}" for feat, t in wide_feat.columns]
    wide_dt = long["__dt__"].unstack("__step__").drop(columns=1)
    wide_dt.columns = [f"{_DT_PREFIX}{t}" for t in wide_dt.columns]

    max_len = int(ordered["__step__"].max())
    columns = [f"{feat}__t{t}" for t in range(1, max_len + 1) for feat in feature_cols]
    columns += [f"{_DT_PREFIX}{t}" for t in range(2, max_len + 1)]
    by_entity = ordered.groupby(seq_key, sort=False)
    wide = pd.concat([wide_feat, wide_dt], axis=1)
    wide[_SEQ_START_COL] = by_entity["__start__"].first()
    wide[_SEQ_LEN_COL] = by_entity.size()
    wide = wide.reindex(columns=[*columns, _SEQ_START_COL, _SEQ_LEN_COL]).reset_index(drop=True)
    # Numerieke features: forceer numeriek zodat SDV ze als 'numerical' detecteert
    # (de pivot met gemengde NaN maakt er anders object van).
    for feat in feature_cols:
        if pd.api.types.is_numeric_dtype(ordered[feat]):
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
    df = df.dropna(subset=[seq_index])

    n_dupes = int(df.duplicated([seq_key, seq_index]).sum())
    if n_dupes:
        raise ValueError(
            f"De data bevat {n_dupes} keer meerdere rijen voor dezelfde entiteit op hetzelfde "
            f"tijdstip ('{seq_key}' × '{seq_index}'). Kies een fijnere tijdkolom, of voeg "
            "die rijen eerst samen tot één rij per entiteit per tijdstip."
        )

    kind, fmt = _detect_index_kind(df[seq_index])
    labels = sorted(df[seq_index].unique().tolist()) if kind == "label" else []
    axis = _to_axis(df[seq_index], kind, fmt, labels)
    level_axis = sorted(axis.unique().tolist())
    levels = df[seq_index].groupby(axis).first()  # één originele waarde per as-positie

    ordered = df.assign(**{_AXIS_COL: axis}).sort_values([seq_key, _AXIS_COL])
    by_entity = ordered.groupby(seq_key, sort=False)
    ordered["__step__"] = by_entity.cumcount() + 1
    ordered["__dt__"] = by_entity[_AXIS_COL].diff()
    level_pos = {a: i + 1 for i, a in enumerate(level_axis)}
    ordered["__start__"] = by_entity[_AXIS_COL].transform("first").map(level_pos)
    max_len = int(ordered["__step__"].max())

    # Generieke vormchecks: blokkeer alleen data die deze aanpak echt niet aankan,
    # ongeacht de dataset. (1) Zonder herhaalde entiteiten of tijdstappen is het
    # niet longitudinaal. (2) Wordt de wide-tabel breder dan het aantal entiteiten
    # (features × stappen ≥ entiteiten), dan is de correlatiematrix onderbepaald
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

    terminal = detect_terminal_states(ordered, seq_key, _AXIS_COL, feature_cols)
    patterns = detect_sequence_patterns(ordered, seq_key, _AXIS_COL, feature_cols)

    fallback: dict[str, Any] = {}
    for feat in feature_cols:
        col = df[feat].dropna()
        if col.empty:
            fallback[feat] = None
        elif pd.api.types.is_numeric_dtype(df[feat]):
            fallback[feat] = col.median()
        else:
            fallback[feat] = col.mode().iloc[0]

    dts = ordered["__dt__"].dropna()
    wide = _to_wide(ordered, seq_key, feature_cols)
    copula = fit(wide, seed=seed)

    return SequentialCopulaModel(
        copula=copula,
        seq_key=seq_key,
        seq_index=seq_index,
        original_columns=list(df.columns),
        feature_cols=feature_cols,
        feature_dtypes={c: df[c].dtype for c in feature_cols},
        index_kind=kind,
        index_format=fmt,
        index_levels=[levels[a] for a in level_axis],
        level_axis=level_axis,
        index_dtype=df[seq_index].dtype,
        dt_mode=float(dts.mode().iloc[0]),
        dt_min=float(dts.min()),
        dt_max=float(dts.max()),
        dt_integer=bool(np.allclose(dts, dts.round())),
        terminal=terminal,
        patterns=patterns,
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
    """Eerste stap (1-based) waarop een gesampelde staat terminaal is, of ``None``."""
    for t in range(1, model.max_len + 1):
        for feat, terms in model.terminal.items():
            if not terms:
                continue
            val = row.get(f"{feat}__t{t}")
            if not _is_missing(val) and val in terms:
                return t
    return None


def _resolve_step(raw: Any, model: SequentialCopulaModel) -> float:
    """Gesamplede stapafstand, binnen het echte bereik; ontbreekt die, de meest voorkomende."""
    dt = model.dt_mode if _is_missing(raw) else float(raw)
    if model.dt_integer:
        dt = round(dt)
    return min(max(dt, model.dt_min), model.dt_max)


def sample_sequential(model: SequentialCopulaModel, n_sequences: int) -> pd.DataFrame:
    """Genereer *n_sequences* synthetische reeksen, terug in het originele long-format."""
    synth_wide = sample(model.copula, n_sequences)
    rows: list[dict] = []
    last_level = model.level_axis[-1]

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

        raw_start = r.get(_SEQ_START_COL)
        start = 1 if _is_missing(raw_start) else int(round(float(raw_start)))
        axis = model.level_axis[max(1, min(start, len(model.level_axis))) - 1]

        last: dict[str, Any] = {feat: None for feat in model.feature_cols}
        for t in range(1, k + 1):
            if t > 1:
                axis += _resolve_step(r.get(f"{_DT_PREFIX}{t}"), model)
                if axis > last_level:  # voorbij het laatste waargenomen tijdniveau
                    break
            record = {
                model.seq_key: new_id,
                model.seq_index: _from_axis(axis, model),
                _AXIS_COL: axis,
            }
            for feat in model.feature_cols:
                val = r.get(f"{feat}__t{t}")
                if _is_missing(val):  # ontbrekende waarde → draag laatst bekende (of fallback) door
                    val = last[feat] if not _is_missing(last[feat]) else model.fallback[feat]
                last[feat] = val
                record[feat] = val
            rows.append(record)

    out = _apply_sequence_patterns(pd.DataFrame(rows), model.seq_key, _AXIS_COL, model.patterns)
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
