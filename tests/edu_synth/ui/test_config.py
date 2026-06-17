"""Tests voor config.py — vertaling van operator-keuze naar inequality-rules."""

from edu_synth.core.synthesize import build_constraints
from edu_synth.ui.config import inequality_rule


def test_inequality_le_keeps_order_not_strict():
    assert inequality_rule("a", "≤", "b") == {
        "type": "inequality",
        "low": "a",
        "high": "b",
        "strict": False,
    }


def test_inequality_lt_keeps_order_strict():
    rule = inequality_rule("a", "<", "b")
    assert rule["low"] == "a" and rule["high"] == "b" and rule["strict"] is True


def test_inequality_ge_flips_order_not_strict():
    rule = inequality_rule("a", "≥", "b")
    assert rule["low"] == "b" and rule["high"] == "a" and rule["strict"] is False


def test_inequality_gt_flips_order_strict():
    rule = inequality_rule("a", ">", "b")
    assert rule["low"] == "b" and rule["high"] == "a" and rule["strict"] is True


def test_rule_feeds_build_constraints():
    # De rule-vorm uit de UI moet door de engine-vertaler komen (één vertaalpad).
    cag = build_constraints([inequality_rule("start", "≤", "eind")])
    assert len(cag) == 1
    assert type(cag[0]).__name__ == "Inequality"
