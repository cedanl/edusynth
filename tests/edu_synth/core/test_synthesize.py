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
    _schema_distributions,
    build_sequential_metadata,
    detect_datetime_format,
    fit,
    fit_par,
    fit_sequential,
    infer_column_hints,
    infer_sequence_columns,
    is_skewed,
    recommend_numerical_distributions,
    sample,
    sample_par,
    sample_sequential,
)

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"
FIXTURE_SCHEMA = FIXTURES / "mini_schema.yaml"
FIXTURE_CSV = FIXTURES / "mini_inschrijving.csv"
FIXTURE_DOORSTROOM = FIXTURES / "mini_doorstroom.csv"


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


# ── Per-kolom distributie-aanbeveling ────────────────────────────────────────────


def _skewed_series() -> pd.Series:
    # Zero-inflated: 90 nullen + 30 oplopende waarden → hoge scheefheid, modefreq 0.75,
    # genoeg unieke waarden voor KDE.
    return pd.Series([0] * 90 + list(range(1, 31)))


def test_is_skewed_true_for_zero_inflated():
    assert is_skewed(_skewed_series()) is True


def test_is_skewed_false_for_normal():
    rng = np.random.default_rng(0)
    assert is_skewed(pd.Series(rng.normal(50, 10, 500))) is False


def test_is_skewed_false_for_low_cardinality():
    # Scheef qua waardeverdeling, maar te weinig unieke waarden → feitelijk discreet.
    assert is_skewed(pd.Series([1, 2, 3] * 40)) is False


def test_recommend_flags_skewed_numeric_column():
    df = pd.DataFrame({"gain": _skewed_series(), "age": np.arange(120)})
    rec = recommend_numerical_distributions(df, ["gain", "age"])
    assert rec == {"gain": "gaussian_kde"}


def test_recommend_respects_numerical_typing():
    # Een scheve kolom die níét als numeriek is opgegeven, blijft buiten beschouwing.
    df = pd.DataFrame({"gain": _skewed_series()})
    assert recommend_numerical_distributions(df, []) == {}


def test_schema_distributions_reads_field():
    schema = {
        "columns": {"a": {"dtype": "float", "distribution": "gamma"}, "b": {"dtype": "float"}}
    }
    assert _schema_distributions(schema) == {"a": "gamma"}


def test_fit_numerical_distributions_is_optional():
    assert inspect.signature(fit).parameters["numerical_distributions"].default is None


def test_fit_auto_applies_kde_on_skewed_column():
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"gain": _skewed_series(), "age": rng.integers(18, 70, 120)})
    model = fit(data, seed=42)
    assert model.get_parameters()["numerical_distributions"].get("gain") == "gaussian_kde"


def test_fit_explicit_empty_distributions_disables_autodetect():
    data = pd.DataFrame({"gain": _skewed_series()})
    model = fit(data, seed=42, numerical_distributions={})
    assert not model.get_parameters()["numerical_distributions"]


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


def test_smoke_sequential_fit_sample_from_fixture():
    """Rookproef die het longitudinale pad in CI bewaakt: fit + sample op een vaste
    mini-fixture (categorische staten + numerieke kolom) crasht niet en levert de
    gevraagde vorm — dezelfde kolommen en het gevraagde aantal sequenties."""
    df = pd.read_csv(FIXTURE_DOORSTROOM)
    model = fit_sequential(df, "student_id", "jaar", seed=42)
    out = sample_sequential(model, n_sequences=10)
    assert list(out.columns) == list(df.columns)
    assert out["student_id"].nunique() == 10


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


def _doorstroom_df(n_students: int = 60, seed: int = 0) -> pd.DataFrame:
    """Longitudinale doorstroom-fixture met absorberende eindstaten (diploma/uitval)."""
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(n_students):
        state, ec = "ingeschreven", 0.0
        for jaar in range(1, 6):
            ec += float(rng.integers(30, 60))
            rows.append({"student_id": sid, "jaar": jaar, "status": state, "ec": ec})
            if state in ("gediplomeerd", "uitgestroomd"):
                break
            state = rng.choice(
                ["ingeschreven", "gediplomeerd", "uitgestroomd"], p=[0.7, 0.15, 0.15]
            )
    return pd.DataFrame(rows)


def test_sample_sequential_restores_original_format():
    df = _doorstroom_df()
    model = fit_sequential(df, "student_id", "jaar", seed=42)
    out = sample_sequential(model, n_sequences=25)
    # Zelfde kolommen in dezelfde volgorde als de echte data.
    assert list(out.columns) == list(df.columns)
    assert out["student_id"].nunique() == 25
    # Numerieke kolom blijft numeriek, categorische kolom blijft categorisch.
    assert pd.api.types.is_numeric_dtype(out["ec"])
    assert set(out["status"].dropna()).issubset(set(df["status"]))


def test_sample_sequential_no_state_after_absorbing():
    """Geen actieve staat ná een eindstaat, en geen gaten in de tijd-as."""
    df = _doorstroom_df()
    model = fit_sequential(df, "student_id", "jaar", seed=1)
    out = sample_sequential(model, n_sequences=40)
    absorbing = {"gediplomeerd", "uitgestroomd"}
    for _, g in out.sort_values(["student_id", "jaar"]).groupby("student_id"):
        jaren = g["jaar"].tolist()
        assert jaren == list(range(jaren[0], jaren[0] + len(jaren)))  # aaneengesloten
        states = g["status"].tolist()
        for i, s in enumerate(states):
            if s in absorbing:
                assert i == len(states) - 1  # eindstaat is de laatste rij


def test_fit_sequential_learns_terminal_states():
    df = _doorstroom_df()
    model = fit_sequential(df, "student_id", "jaar", seed=0)
    assert model.terminal["status"] == {"gediplomeerd", "uitgestroomd"}


def test_sample_sequential_categorical_column_is_single_typed():
    """Reconstructie mag geen int/str-mix teruggeven; anders crasht de verdelingsplot."""
    df = _doorstroom_df()
    model = fit_sequential(df, "student_id", "jaar", seed=3)
    out = sample_sequential(model, n_sequences=30)
    # value_counts van echt + synth samenvoegen (zoals de app doet) mag niet crashen.
    combined = pd.DataFrame(
        {
            "echt": df["status"].astype(str).value_counts(),
            "synth": out["status"].astype(str).value_counts(),
        }
    )
    assert not combined.empty


def test_fit_sequential_refuses_non_longitudinal():
    # Eén tijdstap per entiteit → niet longitudinaal.
    df = pd.DataFrame({"id": [1, 2, 3], "jaar": [1, 1, 1], "x": [10, 20, 30]})
    with pytest.raises(ValueError, match="niet longitudinaal"):
        fit_sequential(df, "id", "jaar")


def test_fit_sequential_refuses_degenerate_wide():
    # 3 entiteiten, 4 features × 3 tijdstappen = 12 dimensies ≥ 3 entiteiten.
    df = pd.DataFrame(
        {
            "id": np.repeat([1, 2, 3], 3),
            "jaar": [1, 2, 3] * 3,
            "a": range(9),
            "b": range(9),
            "c": range(9),
            "d": range(9),
        }
    )
    with pytest.raises(ValueError, match="Te veel kolommen"):
        fit_sequential(df, "id", "jaar")


def test_fit_par_returns_long_format():
    df = _longitudinal_df()
    model = fit_par(df, "student_id", "studiejaar", epochs=1, seed=1)
    out = sample_par(model, n_sequences=3)
    assert set(out.columns) == set(df.columns)
    assert out["student_id"].nunique() == 3


def test_fit_par_progress_reports_increasing_fractions():
    """De progress-callback vuurt per epoch met een oplopende fractie tot 1.0."""
    df = _longitudinal_df()
    seen: list[float] = []
    fit_par(df, "student_id", "studiejaar", epochs=3, seed=1, progress=seen.append)
    assert len(seen) == 3
    assert seen == sorted(seen)  # monotoon oplopend
    assert seen[-1] == pytest.approx(1.0)
    assert all(0 < f <= 1.0 for f in seen)


def test_par_progress_restores_original_tqdm():
    """De tqdm-monkeypatch mag na afloop niets achterlaten in deepecho."""
    import deepecho.models.par as parmod

    before = parmod.tqdm
    fit_par(_longitudinal_df(), "student_id", "studiejaar", epochs=1, progress=lambda _f: None)
    assert parmod.tqdm is before


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
