"""Core synthesis functions — fit, sample, column hint detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

_DATE_PATTERNS = [
    re.compile(r"^\d{8}$"),  # YYYYMMDD
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),  # YYYY-MM-DD
    re.compile(r"^\d{2}-\d{2}-\d{4}$"),  # DD-MM-YYYY
]


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
            if _looks_like_date(sample):
                hints.append(
                    ColumnHint(
                        name=col,
                        detected_sdtype=detected,
                        suggested_sdtype="datetime",
                        reason="Patroon lijkt op een datum (JJJJMMDD, JJJJ-MM-DD, …)",
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


def _looks_like_date(values: list) -> bool:
    sample = [v for v in values[:10] if isinstance(v, str)]
    if len(sample) < 3:
        return False
    matches = sum(1 for v in sample if any(p.match(v.strip()) for p in _DATE_PATTERNS))
    return matches >= len(sample) * 0.7


def safe_batch_size(n_rows: int) -> int:
    """Bereken een veilige CTGAN batch_size die niet groter is dan de dataset."""
    return min(500, max(50, n_rows // 10))


def auto_epochs(n_rows: int) -> int:
    """Kies automatisch het aantal CTGAN-epochs op basis van datasetgrootte."""
    if n_rows < 1_000:
        return 200
    if n_rows < 10_000:
        return 100
    if n_rows < 50_000:
        return 50
    return 30


def estimate_ctgan_width(df: pd.DataFrame, col_types: dict[str, str]) -> int:
    """Schat de breedte van de CTGAN one-hot encoded matrix.

    Categorische kolommen worden one-hot encoded (breedte = unieke waarden),
    numerieke kolommen tellen als 1.
    """
    width = 0
    for col_name, sdtype in col_types.items():
        if col_name not in df.columns:
            continue
        if sdtype == "categorical":
            width += df[col_name].nunique()
        else:
            width += 1
    return width


def ctgan_is_feasible(n_rows: int, encoded_width: int) -> tuple[bool, str]:
    """Check of CTGAN haalbaar is qua geheugen.

    Returns (feasible, reason).
    """
    estimated_bytes = n_rows * encoded_width * 4 * 10
    limit = 2 * 1024**3
    if estimated_bytes > limit:
        gb = estimated_bytes / 1024**3
        return False, f"Geschat geheugengebruik: {gb:.1f} GiB (limiet: 2 GiB)"
    return True, ""


# ── Synthese ───────────────────────────────────────────────────────────────────


def fit(data: pd.DataFrame, schema_path: Path | None = None) -> GaussianCopulaSynthesizer:
    """Train a synthesizer on *data*.

    Parameters
    ----------
    data:
        Real dataset to learn from.
    schema_path:
        Path to a YAML schema file. If omitted, SDV auto-detects column types.

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
        metadata.add_column(col_name, **kwargs)
    return metadata
