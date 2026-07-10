"""Tests voor de regressie-vergelijking in scripts/benchmark.py.

Het script zit buiten het package (zoals de andere scripts/), dus we laden het
los via importlib. Alleen de pure vergelijkingslogica wordt getest — geen synthese,
geen netwerk.
"""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).parents[2] / "scripts" / "benchmark.py"
_spec = importlib.util.spec_from_file_location("benchmark", _SCRIPT)
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


def _baseline(**overrides) -> dict:
    row = {
        "dataset": "demo",
        "mean_score": 0.166,
        "worst_score": 0.8,
        "cols_failed": 3,
        "sdmetrics_overall": 0.79,
    }
    row.update(overrides)
    return {"datasets": [row]}


def test_is_regression_lower_metric():
    # Lager is beter: een stijging binnen de marge is geen regressie, ver erboven wel.
    assert benchmark._is_regression(0.166, 0.166, "lower") is False
    assert benchmark._is_regression(0.166, 0.17, "lower") is False  # < 10% + vloer
    assert benchmark._is_regression(0.166, 0.30, "lower") is True


def test_is_regression_higher_metric():
    # Hoger is beter: een daling binnen de marge is oké, een echte daling niet.
    assert benchmark._is_regression(0.79, 0.79, "higher") is False
    assert benchmark._is_regression(0.79, 0.78, "higher") is False
    assert benchmark._is_regression(0.79, 0.60, "higher") is True


def test_check_no_regression_on_identical_run():
    base = _baseline()
    current = list(base["datasets"])
    assert benchmark.check_against_baseline(base, current) == []


def test_check_detects_worsened_metric():
    base = _baseline()
    current = [{**base["datasets"][0], "mean_score": 0.5}]
    regressions = benchmark.check_against_baseline(base, current)
    assert len(regressions) == 1
    assert regressions[0]["metric"] == "mean_score"


def test_check_flags_failed_run():
    # Dataset stond in de baseline maar levert nu geen scores op → regressie.
    base = _baseline()
    current = [{"dataset": "demo", "rows": 0, "cols": 0, "worst_col": "ERR: boom"}]
    regressions = benchmark.check_against_baseline(base, current)
    assert len(regressions) == 1
    assert regressions[0]["metric"] == "(run mislukt)"


def test_check_ignores_dataset_without_baseline():
    base = _baseline()
    current = [{**base["datasets"][0], "dataset": "nieuw"}]
    assert benchmark.check_against_baseline(base, current) == []


# ── Longitudinaal spoor ───────────────────────────────────────────────────────────


def _seq_baseline(**overrides) -> dict:
    row = {
        "dataset": "doorstroom",
        "length_distance": 0.1567,
        "temporal_worst": 0.1567,
        "temporal_failed": 0,
    }
    row.update(overrides)
    return {"sequential": [row]}


def test_seq_check_no_regression_on_identical_run():
    base = _seq_baseline()
    current = list(base["sequential"])
    assert benchmark.check_sequential_against_baseline(base, current) == []


def test_seq_check_detects_worsened_temporal_metric():
    base = _seq_baseline()
    current = [{**base["sequential"][0], "temporal_worst": 0.5}]
    regressions = benchmark.check_sequential_against_baseline(base, current)
    assert len(regressions) == 1
    assert regressions[0]["metric"] == "temporal_worst"


def test_seq_check_ignores_fit_seconds():
    # Wall-clock is niet gegate: een veel tragere fit is geen regressie.
    base = _seq_baseline()
    current = [{**base["sequential"][0], "fit_seconds": 99.0}]
    assert benchmark.check_sequential_against_baseline(base, current) == []


def test_longitudinal_ground_truth_is_deterministic_and_longitudinal():
    a = benchmark._longitudinal_ground_truth(50, seed=42)
    b = benchmark._longitudinal_ground_truth(50, seed=42)
    assert a.equals(b)  # vaste seed → identieke data
    assert list(a.columns) == ["student_id", "jaar", "status", "ec"]
    assert a["student_id"].nunique() == 50
    # Meerdere tijdstappen per entiteit én absorberende eindstaten aanwezig.
    assert a.groupby("student_id").size().max() >= 2
    assert {"gediplomeerd", "uitgestroomd"} & set(a["status"])
