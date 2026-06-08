"""Tests voor safe_batch_size — CTGAN batch_size berekening."""

from ceda_synth.core.synthesize import safe_batch_size


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
