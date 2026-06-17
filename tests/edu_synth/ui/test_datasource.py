"""Tests voor datasource.py — confidence-splitsing van type-suggesties."""

from edu_synth.core.synthesize import ColumnHint
from edu_synth.ui.datasource import partition_by_confidence


def _hint(name: str, confidence: float) -> ColumnHint:
    return ColumnHint(
        name=name,
        detected_sdtype="numerical",
        suggested_sdtype="categorical",
        reason="test",
        confidence=confidence,
    )


def test_partition_splits_on_default_threshold():
    high, low = partition_by_confidence([_hint("a", 0.9), _hint("b", 0.65)])
    assert [h.name for h in high] == ["a"]
    assert [h.name for h in low] == ["b"]


def test_partition_threshold_is_inclusive():
    high, low = partition_by_confidence([_hint("a", 0.9)])
    assert [h.name for h in high] == ["a"]
    assert low == []


def test_partition_all_uncertain_when_below_threshold():
    high, low = partition_by_confidence([_hint("a", 0.7), _hint("b", 0.8)])
    assert high == []
    assert [h.name for h in low] == ["a", "b"]
