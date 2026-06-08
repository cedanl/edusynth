"""Tests voor CTGAN helper-functies — batch_size, epochs, geheugencheck."""

import pandas as pd

from ceda_synth.core.synthesize import (
    auto_epochs,
    ctgan_is_feasible,
    estimate_ctgan_width,
    safe_batch_size,
)

# ── safe_batch_size ───────────────────────────────────────────────────────────


def test_safe_batch_size_small_dataset():
    assert safe_batch_size(100) == 50


def test_safe_batch_size_medium_dataset():
    assert safe_batch_size(2000) == 200


def test_safe_batch_size_large_dataset():
    assert safe_batch_size(10_000) == 500


def test_safe_batch_size_minimum_clamp():
    assert safe_batch_size(10) == 50


def test_safe_batch_size_exactly_500_threshold():
    assert safe_batch_size(5000) == 500


# ── auto_epochs ───────────────────────────────────────────────────────────────


def test_auto_epochs_small():
    assert auto_epochs(500) == 200


def test_auto_epochs_medium():
    assert auto_epochs(5_000) == 100


def test_auto_epochs_large():
    assert auto_epochs(30_000) == 50


def test_auto_epochs_very_large():
    assert auto_epochs(100_000) == 30


# ── estimate_ctgan_width ──────────────────────────────────────────────────────


def test_estimate_width_categorical_columns():
    df = pd.DataFrame({"a": ["x", "y", "z"], "b": [1, 2, 3]})
    col_types = {"a": "categorical", "b": "numerical"}
    assert estimate_ctgan_width(df, col_types) == 3 + 1


def test_estimate_width_high_cardinality():
    df = pd.DataFrame({"code": [f"c{i}" for i in range(500)]})
    col_types = {"code": "categorical"}
    assert estimate_ctgan_width(df, col_types) == 500


def test_estimate_width_all_numerical():
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    col_types = {"x": "numerical", "y": "numerical"}
    assert estimate_ctgan_width(df, col_types) == 2


# ── ctgan_is_feasible ─────────────────────────────────────────────────────────


def test_feasible_small_dataset():
    feasible, _ = ctgan_is_feasible(1000, 50)
    assert feasible is True


def test_infeasible_huge_encoded_width():
    feasible, reason = ctgan_is_feasible(100_000, 5000)
    assert feasible is False
    assert "GiB" in reason


def test_feasible_boundary():
    feasible, _ = ctgan_is_feasible(1000, 100)
    assert feasible is True
