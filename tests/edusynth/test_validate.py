"""Tests voor validate.py — Report en afstandsmetrieken."""

import pandas as pd

from edusynth.validate import Report, _tv_distance, evaluate


def _make_df():
    return pd.DataFrame({
        "geslacht": ["1", "2", "1", "2", "1"],
        "inschrijvingsjaar": [2019, 2020, 2021, 2019, 2022],
    })


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
