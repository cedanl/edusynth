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
    if schema_path is not None:
        schema = _load_schema(schema_path)
        metadata = _build_metadata(schema)
    else:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)

    if seed is not None:
        set_seed(seed)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    return synthesizer


def sample(model: Any, n_rows: int) -> pd.DataFrame:
    """Generate *n_rows* synthetic rows from a fitted *model*."""
    return model.sample(num_rows=n_rows)


# ── Interne helpers ────────────────────────────────────────────────────────────


def _load_schema(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


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
