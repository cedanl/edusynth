"""Core synthesis functions — fit a model on real data, sample synthetic rows."""

from __future__ import annotations

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


def fit(data: pd.DataFrame, schema_path: Path) -> GaussianCopulaSynthesizer:
    """Train a synthesizer on *data* using the column definitions in *schema_path*.

    Parameters
    ----------
    data:
        Real dataset to learn from.
    schema_path:
        Path to a YAML schema file describing columns, types, and constraints.

    Returns
    -------
    Fitted SDV synthesizer — pass to :func:`sample` to generate rows.
    """
    schema = _load_schema(schema_path)
    metadata = _build_metadata(schema)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    return synthesizer


def sample(model: Any, n_rows: int) -> pd.DataFrame:
    """Generate *n_rows* synthetic rows from a fitted *model*.

    Parameters
    ----------
    model:
        Fitted synthesizer returned by :func:`fit`.
    n_rows:
        Number of rows to generate.
    """
    return model.sample(num_rows=n_rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
