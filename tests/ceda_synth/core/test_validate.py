"""Tests voor validate.py — Report, afstandsmetrieken en privacyvalidatie."""

import numpy as np
import pandas as pd

from ceda_synth.core.validate import (
    PairsReport,
    PrivacyReport,
    Report,
    _count_modes,
    _tv_distance,
    evaluate,
    evaluate_pairs,
    evaluate_privacy,
    usage_recommendation,
)


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


# ── Privacyvalidatie ───────────────────────────────────────────────────────────


def test_privacy_report_identical_data():
    df = _make_numeric_df(60)
    priv = evaluate_privacy(df, df.copy())
    assert isinstance(priv, PrivacyReport)
    assert priv.available
    assert priv.dcr_ratio >= 0.0


def test_privacy_report_no_numeric_columns():
    df = pd.DataFrame({"cat": ["a", "b", "c"] * 10})
    priv = evaluate_privacy(df, df.copy())
    assert not priv.available
    assert "numerieke" in priv.reason


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
    assert "rapportage" in rec.lower() or "analyse" in rec.lower()


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
