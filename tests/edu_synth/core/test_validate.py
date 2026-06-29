"""Tests voor validate.py — Report, afstandsmetrieken en privacyvalidatie."""

import numpy as np
import pandas as pd

from edu_synth.core.validate import (
    RECOMMENDATION_DISCLAIMER,
    PairsReport,
    PrivacyReport,
    Report,
    SDMetricsReport,
    _count_modes,
    _spread,
    _tv_distance,
    build_validation_report,
    correlation_risk,
    evaluate,
    evaluate_pairs,
    evaluate_privacy,
    evaluate_sdmetrics,
    improvement_advice,
    usage_recommendation,
)


def _metadata(df: pd.DataFrame) -> dict:
    from sdv.metadata import SingleTableMetadata

    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)
    return meta.to_dict()


def _make_df():
    return pd.DataFrame(
        {
            "geslacht": ["1", "2", "1", "2", "1"],
            "inschrijvingsjaar": [2019, 2020, 2021, 2019, 2022],
        }
    )


def _make_numeric_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(5, 2, n),
        }
    )


# ── Statistische validatie ─────────────────────────────────────────────────────


def test_evaluate_returns_report():
    df = _make_df()
    report = evaluate(df, df.copy())
    assert isinstance(report, Report)
    assert len(report.rows) == 2


def test_identical_data_has_zero_tv():
    s = pd.Series(["a", "b", "a", "b"])
    assert _tv_distance(s, s) == 0.0


def test_report_to_dataframe():
    df = _make_df()
    report = evaluate(df, df.copy())
    result = report.to_dataframe()
    assert "column" in result.columns
    assert "distance" in result.columns


def test_passed_on_identical_data():
    df = _make_df()
    report = evaluate(df, df.copy())
    assert report.passed()


def test_report_has_modal_flags_field():
    df = _make_df()
    report = evaluate(df, df.copy())
    assert hasattr(report, "modal_flags")
    assert isinstance(report.modal_flags, list)


# ── Wasserstein IQR-normalisatie ────────────────────────────────────────────────


def test_numeric_row_has_score_and_ok():
    df = _make_df()
    report = evaluate(df, df.copy())
    num = next(r for r in report.rows if r["metric"] == "wasserstein")
    assert "score" in num
    assert "ok" in num
    assert num["score"] == 0.0  # identieke data
    assert num["ok"] is True


def test_score_is_scale_invariant():
    # Zelfde relatieve verschuiving op twee verschillende schalen → zelfde score.
    rng = np.random.default_rng(7)
    base = rng.normal(0, 1, 200)
    small = pd.DataFrame({"x": base})
    large = pd.DataFrame({"x": base * 1000})
    # synth = echte data 0.5·sd verschoven, op elke schaal
    small_s = pd.DataFrame({"x": base + 0.5})
    large_s = pd.DataFrame({"x": base * 1000 + 500})
    score_small = evaluate(small, small_s).rows[0]["score"]
    score_large = evaluate(large, large_s).rows[0]["score"]
    assert abs(score_small - score_large) < 0.01


def test_constant_column_does_not_crash():
    df = pd.DataFrame({"const": [5, 5, 5, 5, 5]})
    report = evaluate(df, df.copy())
    assert report.rows[0]["score"] == 0.0
    assert report.rows[0]["ok"] is True


def test_boolean_column_treated_as_categorical():
    # Regressie: bool telt bij pandas als numeriek, maar de IQR-berekening crasht
    # erop. Een ja/nee-kolom hoort daarom via TV-afstand vergeleken te worden.
    df = pd.DataFrame({"geslaagd": [True, False, True, True, False]})
    report = evaluate(df, df.copy())
    assert report.rows[0]["metric"] == "tv"
    assert report.rows[0]["score"] == 0.0


def test_dtype_divergence_does_not_crash():
    # Regressie: SDV kan een int-kolom anonimiseren naar tekst (bv. stadscodes →
    # stadsnamen). De kolom is dan numeriek in echt, tekst in synthetisch — dat
    # mag niet crashen, maar terugvallen op de TV-afstand.
    real = pd.DataFrame({"stad": [1001, 1002, 1003, 1001, 1002]})
    synth = pd.DataFrame({"stad": ["Lisaville", "Port X", "Lisaville", "Port X", "Lisaville"]})
    report = evaluate(real, synth)
    assert report.rows[0]["metric"] == "tv"
    # Ook pairs en privacy mogen niet crashen op de gedivergeerde kolom.
    assert evaluate_pairs(real, synth).available is False  # < 2 numerieke kolommen
    assert isinstance(evaluate_privacy(real, synth), PrivacyReport)


def test_spread_fallback_chain():
    assert _spread(pd.Series([1, 2, 3, 4, 5])) > 0  # IQR
    assert _spread(pd.Series([5, 5, 5, 5, 5])) == 1.0  # constante kolom → 1.0


def test_usage_recommendation_flags_bad_numeric():
    # Numerieke kolom volledig verkeerd gesynthetiseerd → mag niet "rapportages" opleveren.
    rng = np.random.default_rng(8)
    real = pd.DataFrame({"score": rng.normal(100, 10, 200)})
    synth = pd.DataFrame({"score": rng.normal(500, 10, 200)})  # ver weg
    report = evaluate(real, synth)
    assert not report.rows[0]["ok"]
    rec = usage_recommendation(report)
    assert "rapportages, kruistabellen en publicatie" not in rec


# ── Privacyvalidatie ───────────────────────────────────────────────────────────


def test_privacy_report_identical_data():
    df = _make_numeric_df(60)
    priv = evaluate_privacy(df, df.copy())
    assert isinstance(priv, PrivacyReport)
    assert priv.available
    assert priv.dcr_ratio >= 0.0


def test_privacy_report_categorical_only_is_available():
    # Categorische quasi-identifiers tellen nu mee (bug #11).
    df = pd.DataFrame({"cat": ["a", "b", "c"] * 10})
    priv = evaluate_privacy(df, df.copy())
    assert priv.available
    assert priv.n_categorical_cols == 1
    assert priv.n_numeric_cols == 0


def test_privacy_report_mixed_columns_counts_both():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "geslacht": rng.choice(["1", "2"], 60),
            "opleiding": rng.choice(["A", "B", "C"], 60),
            "leeftijd": rng.integers(17, 30, 60),
        }
    )
    priv = evaluate_privacy(df, df.copy())
    assert priv.available
    assert priv.n_numeric_cols == 1
    assert priv.n_categorical_cols == 2
    assert priv.n_cols == 3


def test_privacy_report_excludes_high_cardinality_column():
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        {
            "geslacht": rng.choice(["1", "2"], 60),
            "naam": [f"student_{i}" for i in range(60)],  # uniek → identifier
        }
    )
    priv = evaluate_privacy(df, df.copy())
    assert "naam" in priv.excluded_cols
    assert priv.n_categorical_cols == 1


def test_privacy_report_excludes_primary_key():
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {
            "id": range(60),
            "geslacht": rng.choice(["1", "2"], 60),
            "leeftijd": rng.integers(17, 30, 60),
        }
    )
    priv = evaluate_privacy(df, df.copy(), primary_key="id")
    assert priv.available
    # id mag niet als numerieke kolom meetellen
    assert priv.n_numeric_cols == 1
    assert priv.n_categorical_cols == 1


def test_privacy_report_no_usable_columns():
    df = pd.DataFrame({"naam": [f"persoon_{i}" for i in range(30)]})
    priv = evaluate_privacy(df, df.copy())
    assert not priv.available
    assert "naam" in priv.excluded_cols


def test_privacy_report_too_few_rows():
    df = _make_numeric_df(5)
    priv = evaluate_privacy(df, df.copy())
    assert not priv.available


def test_privacy_report_risk_levels():
    assert PrivacyReport(available=True, risk_level="laag").passed() is True
    assert PrivacyReport(available=True, risk_level="hoog").passed() is False
    assert PrivacyReport(available=False).passed() is False


def test_privacy_report_dcr_ratio_range():
    df = _make_numeric_df(100)
    priv = evaluate_privacy(df, df.copy())
    assert priv.dcr_ratio >= 0.0
    assert priv.nndr_median >= 0.0
    assert priv.n_cols == 2


# ── Gebruiksaanbeveling ────────────────────────────────────────────────────────


def test_usage_recommendation_returns_string():
    df = _make_df()
    report = evaluate(df, df.copy())
    rec = usage_recommendation(report)
    assert isinstance(rec, str)
    assert len(rec) > 0


def test_usage_recommendation_high_privacy_risk():
    df = _make_df()
    report = evaluate(df, df.copy())
    priv = PrivacyReport(available=True, risk_level="hoog")
    rec = usage_recommendation(report, priv)
    assert "privacyrisico" in rec.lower() or "niet aanbevolen" in rec.lower()


def test_usage_recommendation_good_data():
    df = _make_df()
    report = evaluate(df, df.copy())
    rec = usage_recommendation(report)
    assert "kwaliteit" in rec.lower()


def test_usage_recommendation_never_claims_publication():
    # #21: het automatische oordeel mag geen publicatiegeschiktheid suggereren.
    df = _make_df()
    good = usage_recommendation(evaluate(df, df.copy()))
    high_priv = usage_recommendation(
        evaluate(df, df.copy()), PrivacyReport(available=True, risk_level="hoog")
    )
    rng = np.random.default_rng(8)
    bad = usage_recommendation(
        evaluate(
            pd.DataFrame({"score": rng.normal(100, 10, 200)}),
            pd.DataFrame({"score": rng.normal(500, 10, 200)}),
        )
    )
    for rec in (good, high_priv, bad):
        assert "publicatie" not in rec.lower()


def test_build_validation_report_structure():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "geslacht": rng.choice(["1", "2"], 200),
            "opleiding": rng.choice(["A", "B", "C"], 200),
            "leeftijd": rng.integers(17, 30, 200),
        }
    )
    report = evaluate(df, df.copy())
    priv = evaluate_privacy(df, df.copy())
    sdm = evaluate_sdmetrics(df, df.copy(), _metadata(df))
    out = build_validation_report(
        report=report,
        priv=priv,
        sdm=sdm,
        recommendation=usage_recommendation(report, priv),
        synthesizer="gaussian",
        n_training_rows=len(df),
        n_generated_rows=100,
        sdv_version="1.37.0",
        generated_at="2026-06-15",
        random_seed=42,
        intended_use="Intern onderzoek / analyse",
    )
    assert out["sdv_version"] == "1.37.0"
    assert out["synthesizer"] == "gaussian"
    assert out["n_training_rows"] == len(df)
    assert out["random_seed"] == 42
    assert out["intended_use"] == "Intern onderzoek / analyse"
    assert len(out["column_stats"]) == len(report.rows)
    assert out["privacy"]["available"] is True
    assert out["sdmetrics"]["available"] is True
    assert out["disclaimer"] == RECOMMENDATION_DISCLAIMER


def test_build_validation_report_is_json_serializable():
    import json

    df = _make_df()
    report = evaluate(df, df.copy())
    priv = evaluate_privacy(df, df.copy())
    sdm = evaluate_sdmetrics(df, df.copy(), _metadata(df))
    out = build_validation_report(
        report=report,
        priv=priv,
        sdm=sdm,
        recommendation="x",
        synthesizer="gaussian",
        n_training_rows=len(df),
        n_generated_rows=5,
        sdv_version="1.37.0",
        generated_at="2026-06-15",
    )
    # mag niet crashen en seed/intended_use zijn None wanneer niet meegegeven
    text = json.dumps(out)
    assert '"random_seed": null' in text
    assert '"intended_use": null' in text


def test_build_validation_report_unavailable_privacy():
    df = pd.DataFrame({"naam": [f"p_{i}" for i in range(30)]})
    report = evaluate(df, df.copy())
    priv = evaluate_privacy(df, df.copy())  # geen bruikbare kolommen → unavailable
    sdm = evaluate_sdmetrics(df, df.copy(), None)  # geen metadata → unavailable
    out = build_validation_report(
        report=report,
        priv=priv,
        sdm=sdm,
        recommendation="x",
        synthesizer="gaussian",
        n_training_rows=len(df),
        n_generated_rows=5,
        sdv_version="1.37.0",
        generated_at="2026-06-15",
    )
    assert out["privacy"]["available"] is False
    assert "reason" in out["privacy"]
    assert out["sdmetrics"]["available"] is False


def test_recommendation_disclaimer_is_neutral():
    low = RECOMMENDATION_DISCLAIMER.lower()
    assert "vuistregel" in low
    assert "beoordeel zelf" in low
    # Doelgroep is data-analisten, geen onderzoekers: geen publicatie-/onderzoeksframing.
    assert "publicatie" not in low
    assert "onderzoeker" not in low
    assert "peer-reviewed" not in low


# ── evaluate_pairs ─────────────────────────────────────────────────────────────


def test_evaluate_pairs_returns_pairs_report():
    df = _make_numeric_df(50)
    result = evaluate_pairs(df, df.copy())
    assert isinstance(result, PairsReport)
    assert result.available


def test_evaluate_pairs_identical_has_no_flags():
    df = _make_numeric_df(50)
    result = evaluate_pairs(df, df.copy())
    assert result.available
    assert result.flagged == []


def test_evaluate_pairs_detects_divergence():
    rng = np.random.default_rng(1)
    n = 100
    real = pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
        }
    )
    # synth with different correlation
    synth = pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n) * 5,
        }
    )
    result = evaluate_pairs(real, synth)
    assert result.available
    # delta should be non-negative
    for f in result.flagged:
        assert f["delta"] >= 0.0


def test_evaluate_pairs_too_few_numeric_columns():
    df = pd.DataFrame({"cat": ["a", "b", "c"] * 10, "num": [1.0, 2.0, 3.0] * 10})
    result = evaluate_pairs(df, df.copy())
    assert not result.available
    assert "2" in result.reason


def test_evaluate_pairs_flagged_sorted_by_delta():
    rng = np.random.default_rng(2)
    n = 80
    real = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "c": rng.normal(0, 1, n),
        }
    )
    synth = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 5, n),
            "c": rng.normal(0, 10, n),
        }
    )
    result = evaluate_pairs(real, synth)
    if len(result.flagged) > 1:
        deltas = [f["delta"] for f in result.flagged]
        assert deltas == sorted(deltas, reverse=True)


def test_evaluate_pairs_flagged_structure():
    rng = np.random.default_rng(3)
    n = 60
    real = pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
        }
    )
    synth = pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "y": real["x"] * 10 + rng.normal(0, 0.1, n),
        }
    )
    result = evaluate_pairs(real, synth)
    for f in result.flagged:
        assert "col_a" in f
        assert "col_b" in f
        assert "real_corr" in f
        assert "synth_corr" in f
        assert "delta" in f


# ── correlation_risk (#67) ──────────────────────────────────────────────────────
def test_correlation_risk_laag_without_flags():
    assert correlation_risk(PairsReport(available=True, flagged=[])) == "laag"


def test_correlation_risk_laag_when_unavailable():
    # Te weinig numerieke kolommen mag het oordeel niet verlagen.
    assert correlation_risk(PairsReport(available=False, flagged=[])) == "laag"


def test_correlation_risk_hoog_on_sign_flip():
    # Een betekenisvol positief verband wordt negatief → omgeklapt → hoog.
    pairs = PairsReport(
        available=True,
        flagged=[
            {"col_a": "a", "col_b": "b", "real_corr": 0.2, "synth_corr": -0.11, "delta": 0.31}
        ],
    )
    assert correlation_risk(pairs) == "hoog"


def test_correlation_risk_hoog_on_large_delta():
    # Grote delta zonder tekenomslag is ook ernstig.
    pairs = PairsReport(
        available=True,
        flagged=[{"col_a": "a", "col_b": "b", "real_corr": 0.6, "synth_corr": 0.25, "delta": 0.35}],
    )
    assert correlation_risk(pairs) == "hoog"


def test_correlation_risk_matig_on_mild_deviation():
    # Wel geflagd (delta > 0.1) maar mild en geen tekenomslag.
    pairs = PairsReport(
        available=True,
        flagged=[{"col_a": "a", "col_b": "b", "real_corr": 0.4, "synth_corr": 0.25, "delta": 0.15}],
    )
    assert correlation_risk(pairs) == "matig"


def test_correlation_risk_ignores_trivial_sign_flip():
    # Tekenomslag op een verwaarloosbaar verband (|real_corr| < 0.15) is ruis,
    # geen omgeklapt verband → niet 'hoog'.
    pairs = PairsReport(
        available=True,
        flagged=[
            {"col_a": "a", "col_b": "b", "real_corr": 0.05, "synth_corr": -0.08, "delta": 0.13}
        ],
    )
    assert correlation_risk(pairs) == "matig"


def test_usage_recommendation_warns_on_flipped_correlation():
    df = _make_df()
    report = evaluate(df, df.copy())
    pairs = PairsReport(
        available=True,
        flagged=[
            {"col_a": "a", "col_b": "b", "real_corr": 0.2, "synth_corr": -0.11, "delta": 0.31}
        ],
    )
    rec = usage_recommendation(report, None, pairs)
    assert "omgeklapt" in rec.lower() or "verband" in rec.lower()


# ── sdmetrics QualityReport ──────────────────────────────────────────────────────


def test_sdmetrics_no_metadata_unavailable():
    df = _make_df()
    sdm = evaluate_sdmetrics(df, df.copy(), None)
    assert isinstance(sdm, SDMetricsReport)
    assert not sdm.available
    assert sdm.overall_score is None


def test_sdmetrics_identical_data_scores_high():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "geslacht": rng.choice(["1", "2"], 300),
            "opleiding": rng.choice(["A", "B", "C"], 300),
            "leeftijd": rng.integers(17, 30, 300),
        }
    )
    sdm = evaluate_sdmetrics(df, df.copy(), _metadata(df))
    assert sdm.available
    assert sdm.overall_score is not None
    assert sdm.overall_score > 0.9
    # Column Shapes bevat één rij per (niet-id) kolom.
    assert len(sdm.column_shapes) == 3


def test_sdmetrics_strong_association_has_pair_score():
    rng = np.random.default_rng(1)
    x = rng.choice(["1", "2", "3"], 400)
    df = pd.DataFrame({"x": x, "y": x})  # y volledig bepaald door x
    sdm = evaluate_sdmetrics(df, df.copy(), _metadata(df))
    assert sdm.available
    assert len(sdm.column_pair_trends) == 1
    assert sdm.column_pair_trends[0]["Score"] == 1.0


# ── _count_modes ───────────────────────────────────────────────────────────────


def test_count_modes_unimodal():
    rng = np.random.default_rng(42)
    series = pd.Series(rng.normal(0, 1, 200))
    modes = _count_modes(series)
    assert modes >= 1


def test_count_modes_bimodal():
    rng = np.random.default_rng(42)
    series = pd.Series(np.concatenate([rng.normal(-5, 0.5, 100), rng.normal(5, 0.5, 100)]))
    modes = _count_modes(series)
    assert modes >= 2


def test_count_modes_returns_at_least_one():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _count_modes(series) >= 1


def test_count_modes_with_nan():
    series = pd.Series([1.0, np.nan, 2.0, np.nan, 1.0, 2.0] * 10)
    modes = _count_modes(series)
    assert modes >= 1


# ── Verbeteradvies (#31) ─────────────────────────────────────────────────────────


def _failing_row(col: str, dtype: str = "categorical") -> dict:
    return {
        "column": col,
        "dtype": dtype,
        "distance": 0.5,
        "score": 0.5,
        "metric": "tv",
        "ok": False,
    }


def test_advice_empty_when_all_ok():
    report = Report(rows=[{"column": "a", "dtype": "categorical", "score": 0.05, "ok": True}])
    df = pd.DataFrame({"a": ["x", "y"] * 300})
    assert improvement_advice(report, df) == []


def test_advice_flags_mistyped_column():
    # Integer-code (1..12) die SDV als numeriek ziet maar categorisch hoort te zijn.
    df = pd.DataFrame({"opl_vorm": list(range(1, 13)) * 50})
    report = Report(rows=[_failing_row("opl_vorm", dtype="numeric")])
    advice = improvement_advice(report, df)
    assert any("opl_vorm" in a and "categorisch" in a for a in advice)


def test_advice_flags_multimodal_column():
    df = pd.DataFrame({"score": list(range(600))})
    report = Report(
        rows=[_failing_row("score", dtype="numeric")],
        modal_flags=[{"column": "score", "real_modes": 2, "synth_modes": 1}],
    )
    advice = improvement_advice(report, df)
    assert any("score" in a and "gaussian_kde" in a for a in advice)


def test_advice_flags_high_cardinality():
    df = pd.DataFrame({"stad": [f"plaats_{i}" for i in range(600)]})
    report = Report(rows=[_failing_row("stad", dtype="categorical")])
    advice = improvement_advice(report, df)
    assert any("stad" in a and "unieke waarden" in a for a in advice)


def test_advice_warns_small_dataset():
    df = pd.DataFrame({"a": ["x", "y"] * 50})  # 100 rijen < 500
    report = Report(rows=[{"column": "a", "dtype": "categorical", "score": 0.05, "ok": True}])
    advice = improvement_advice(report, df)
    assert any("klein" in a for a in advice)


def test_advice_privacy_risk_comes_first():
    df = pd.DataFrame({"a": ["x", "y"] * 300})
    report = Report(rows=[_failing_row("a")])
    priv = PrivacyReport(available=True, risk_level="hoog")
    advice = improvement_advice(report, df, priv)
    assert "Privacy" in advice[0]


def test_advice_capped_at_max():
    df = pd.DataFrame({f"c{i}": [f"v_{j}" for j in range(600)] for i in range(8)})
    report = Report(rows=[_failing_row(f"c{i}", dtype="categorical") for i in range(8)])
    assert len(improvement_advice(report, df)) <= 4
