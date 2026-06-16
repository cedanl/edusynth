"""Tests voor synthesize.py — schema laden, metadata opbouwen, hints."""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sdv.cag import FixedCombinations, Inequality

from edu_synth.core.synthesize import (
    ColumnHint,
    _build_constraints,
    _build_metadata,
    _detect_date_format,
    _load_schema,
    build_sequential_metadata,
    detect_datetime_format,
    fit,
    infer_column_hints,
    infer_sequence_columns,
    sample,
)

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"
FIXTURE_SCHEMA = FIXTURES / "mini_schema.yaml"
FIXTURE_CSV = FIXTURES / "mini_inschrijving.csv"


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


# ── datetime_format ──────────────────────────────────────────────────────────────


def test_build_metadata_sets_datetime_format():
    schema = {
        "columns": {
            "inschrijfdatum": {"dtype": "date", "datetime_format": "%Y%m%d"},
        }
    }
    metadata = _build_metadata(schema)
    col = metadata.columns["inschrijfdatum"]
    assert col["sdtype"] == "datetime"
    assert col["datetime_format"] == "%Y%m%d"


def test_build_metadata_datetime_format_defaults():
    schema = {"columns": {"d": {"dtype": "date"}}}
    metadata = _build_metadata(schema)
    assert metadata.columns["d"]["datetime_format"] == "%Y-%m-%d"


def test_detect_date_format_duo_yyyymmdd():
    assert _detect_date_format(["20190101", "20200315", "20211231"]) == "%Y%m%d"


def test_detect_date_format_iso():
    assert _detect_date_format(["2019-01-01", "2020-03-15", "2021-12-31"]) == "%Y-%m-%d"


def test_detect_date_format_none_for_non_dates():
    assert _detect_date_format(["hbo", "wo", "mbo"]) is None


def test_detect_datetime_format_on_series():
    # Publieke helper die de app gebruikt — werkt op een Series, negeert NaN.
    s = pd.Series(["20190101", None, "20200315", "20211231"])
    assert detect_datetime_format(s) == "%Y%m%d"


def test_detect_datetime_format_none_for_categorical_series():
    assert detect_datetime_format(pd.Series(["hbo", "wo", "hbo"])) is None


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


# ── Sequentieel / longitudinaal ──────────────────────────────────────────────────


def _longitudinal_df(n_students: int = 20, years: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "student_id": np.repeat(np.arange(n_students), years),
            "studiejaar": list(range(2018, 2018 + years)) * n_students,
            "ec": rng.integers(0, 60, n_students * years),
        }
    )


def test_infer_sequence_columns_detects_longitudinal():
    looks, seq_key, seq_index = infer_sequence_columns(_longitudinal_df())
    assert looks is True
    assert seq_key == "student_id"
    assert seq_index == "studiejaar"  # tijd-naam-hint wint


def test_infer_sequence_columns_flat_data_not_longitudinal():
    # Alle rijen uniek → geen herhaalde entiteit → niet longitudinaal
    df = pd.DataFrame({"id": range(50), "waarde": range(50)})
    looks, seq_key, _ = infer_sequence_columns(df)
    assert looks is False
    assert seq_key is None


def test_build_sequential_metadata_sets_key_and_index():
    df = _longitudinal_df()
    md = build_sequential_metadata(df, "student_id", "studiejaar")
    table = md.tables["data"]
    assert table.sequence_key == "student_id"
    assert table.sequence_index == "studiejaar"
    assert table.columns["student_id"]["sdtype"] == "id"
    assert table.columns["studiejaar"]["sdtype"] == "numerical"


def test_build_sequential_metadata_feeds_par():
    from sdv.sequential import PARSynthesizer

    df = _longitudinal_df()
    md = build_sequential_metadata(df, "student_id", "studiejaar")
    model = PARSynthesizer(md, epochs=1, verbose=False)
    model.fit(df)
    out = model.sample(num_sequences=3)
    assert set(out.columns) == set(df.columns)


# ── Cross-column constraints ─────────────────────────────────────────────────────


def test_build_constraints_inequality():
    schema = {
        "constraints": [
            {"type": "inequality", "low": "in_jaar", "high": "uit_jaar", "strict": True},
        ]
    }
    constraints = _build_constraints(schema)
    assert len(constraints) == 1
    assert isinstance(constraints[0], Inequality)
    assert constraints[0]._low_column_name == "in_jaar"
    assert constraints[0]._high_column_name == "uit_jaar"
    assert constraints[0].strict_boundaries is True


def test_build_constraints_fixed_combinations():
    schema = {
        "constraints": [
            {"type": "fixed_combinations", "columns": ["instelling", "opleiding"]},
        ]
    }
    constraints = _build_constraints(schema)
    assert len(constraints) == 1
    assert isinstance(constraints[0], FixedCombinations)
    assert constraints[0].column_names == ["instelling", "opleiding"]


def test_build_constraints_default_strict_false():
    schema = {"constraints": [{"type": "inequality", "low": "a", "high": "b"}]}
    assert _build_constraints(schema)[0].strict_boundaries is False


def test_build_constraints_unknown_type_raises():
    schema = {"constraints": [{"type": "zoiets_bestaat_niet"}]}
    with pytest.raises(ValueError, match="onbekend type"):
        _build_constraints(schema)


def test_build_constraints_missing_key_raises():
    schema = {"constraints": [{"type": "inequality", "low": "a"}]}
    with pytest.raises(ValueError, match="verplichte sleutel"):
        _build_constraints(schema)


def test_build_constraints_empty_when_absent():
    assert _build_constraints({"columns": {}}) == []


def test_fit_respects_inequality():
    data = pd.read_csv(FIXTURE_CSV)
    model = fit(data, FIXTURE_SCHEMA, seed=42)
    out = sample(model, 200)
    assert (out["inschrijvingsjaar"] > out["uitschrijvingsjaar"]).sum() == 0


def test_fit_respects_fixed_combinations():
    data = pd.read_csv(FIXTURE_CSV)
    model = fit(data, FIXTURE_SCHEMA, seed=42)
    out = sample(model, 200)
    real_combos = set(map(tuple, data[["instellingscode", "opleidingscode"]].astype(str).values))
    out_combos = set(map(tuple, out[["instellingscode", "opleidingscode"]].astype(str).values))
    assert out_combos <= real_combos


def test_fit_constraint_on_missing_column_raises_clear_error(tmp_path):
    schema = tmp_path / "schema.yaml"
    schema.write_text(
        "columns:\n  a:\n    dtype: integer\n"
        "constraints:\n  - type: inequality\n    low: a\n    high: bestaat_niet\n",
        encoding="utf-8",
    )
    data = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    with pytest.raises(ValueError, match="niet worden toegepast"):
        fit(data, schema)
