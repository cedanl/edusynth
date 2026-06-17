"""Core synthesis functions — fit, sample, column hint detection."""

from __future__ import annotations

import random
import re
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


# ── Synthese ───────────────────────────────────────────────────────────────────


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

    if seed is not None:
        set_seed(seed)
    synthesizer = GaussianCopulaSynthesizer(metadata)

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
