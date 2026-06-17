"""Tests voor results.py — reproductiecode, metadata-samenvatting, kolomprioritering."""

from edu_synth.core.validate import Report
from edu_synth.ui.results import (
    _build_code,
    _rank_columns_by_deviation,
    _summarize_metadata,
)


def test_build_code_includes_seed_when_set():
    code = _build_code(
        col_types={"leeftijd": "numerical"},
        primary_key=None,
        modality="single_table",
        demo_name=None,
        n_generated=100,
        sdv_version="1.17.0",
        random_seed=42,
    )
    assert "import numpy as np" in code
    assert "np.random.seed(42)" in code


def test_build_code_omits_seed_when_none():
    code = _build_code(
        col_types={"leeftijd": "numerical"},
        primary_key=None,
        modality="single_table",
        demo_name=None,
        n_generated=100,
        sdv_version="1.17.0",
        random_seed=None,
    )
    assert "np.random.seed" not in code


# ── _summarize_metadata (#22) ─────────────────────────────────────────────────
def test_summarize_metadata_counts_types_and_pii():
    meta = {
        "columns": {
            "a": {"sdtype": "categorical"},
            "b": {"sdtype": "categorical"},
            "c": {"sdtype": "numerical"},
            "d": {"sdtype": "datetime"},
            "e": {"sdtype": "id", "pii": True},
        }
    }
    summary = _summarize_metadata(meta)
    assert "2 categorisch" in summary
    assert "1 numeriek" in summary
    assert "1 datum" in summary
    assert "1 kolom(men) gemarkeerd als privacygevoelig" in summary


def test_summarize_metadata_returns_none_without_columns():
    assert _summarize_metadata(None) is None
    assert _summarize_metadata({}) is None
    assert _summarize_metadata({"columns": {}}) is None


# ── _rank_columns_by_deviation (#24) ──────────────────────────────────────────
def test_rank_orders_by_descending_score():
    report = Report(
        rows=[
            {"column": "laag", "score": 0.05},
            {"column": "hoog", "score": 0.42},
            {"column": "mid", "score": 0.20},
        ]
    )
    assert _rank_columns_by_deviation(["laag", "hoog", "mid"], report) == [
        "hoog",
        "mid",
        "laag",
    ]


def test_rank_puts_columns_without_score_last():
    report = Report(rows=[{"column": "a", "score": 0.3}])
    assert _rank_columns_by_deviation(["a", "b"], report) == ["a", "b"]
