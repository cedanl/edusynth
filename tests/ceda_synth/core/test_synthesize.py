"""Tests voor synthesize.py — schema laden, metadata opbouwen, hints."""

import inspect
from pathlib import Path

import pandas as pd

from ceda_synth.core.synthesize import (
    ColumnHint,
    _build_metadata,
    _load_schema,
    fit,
    infer_column_hints,
    sample,
)

FIXTURE_SCHEMA = Path(__file__).parent.parent.parent / "fixtures" / "mini_schema.yaml"


def test_load_schema_returns_dict():
    schema = _load_schema(FIXTURE_SCHEMA)
    assert isinstance(schema, dict)
    assert "columns" in schema


def test_build_metadata_sets_primary_key():
    schema = _load_schema(FIXTURE_SCHEMA)
    metadata = _build_metadata(schema)
    assert metadata.primary_key is not None


def test_fit_schema_path_is_optional():
    sig = inspect.signature(fit)
    assert sig.parameters["schema_path"].default is None


def test_fit_seed_is_optional():
    sig = inspect.signature(fit)
    assert sig.parameters["seed"].default is None


def test_same_seed_gives_identical_output():
    import numpy as np

    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "leeftijd": rng.integers(18, 70, 200),
            "geslacht": rng.choice(["M", "V"], 200),
        }
    )
    out_a = sample(fit(data, seed=42), 100)
    out_b = sample(fit(data, seed=42), 100)
    pd.testing.assert_frame_equal(out_a, out_b)


# ── Kolomtype-hints ────────────────────────────────────────────────────────────


def test_hints_suggests_categorical_for_code_column():
    # opl_vorm: 12 unieke waarden (1-12), SDV detecteert als numerical,
    # heuristiek herkent als categorische code (n_unique <= 15, max <= 100)
    df = pd.DataFrame({"opl_vorm": list(range(1, 13)) * 50})
    hints = infer_column_hints(df)
    h = next((h for h in hints if h.name == "opl_vorm"), None)
    assert h is not None
    assert h.suggested_sdtype == "categorical"
    assert h.has_suggestion is True


def test_hints_returns_column_hint_objects():
    df = pd.DataFrame({"x": [1, 2, 1, 2] * 5})
    hints = infer_column_hints(df)
    assert all(isinstance(h, ColumnHint) for h in hints)


def test_hints_no_suggestion_for_continuous_numeric():
    import numpy as np

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"inkomen": rng.integers(20_000, 80_000, 200)})
    hints = infer_column_hints(df)
    cat_hints = [h for h in hints if h.name == "inkomen" and h.has_suggestion]
    assert len(cat_hints) == 0


def test_hints_warning_for_high_missingness():
    df = pd.DataFrame({"col": [1, None, None, None, None] * 20})
    hints = infer_column_hints(df)
    warn = [h for h in hints if h.name == "col" and not h.has_suggestion]
    assert len(warn) == 1
    assert "missende" in warn[0].reason


def test_column_hint_has_suggestion_property():
    h_with = ColumnHint("x", "numerical", "categorical", "test", 0.8)
    h_without = ColumnHint("y", "categorical", "categorical", "test", 0.0)
    assert h_with.has_suggestion is True
    assert h_without.has_suggestion is False
