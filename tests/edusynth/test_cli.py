"""Tests voor CLI argument parsing en exitcodes."""

import pytest

from edusynth.cli import _parse_args


def test_synthesize_required_args():
    args = _parse_args(["synthesize", "data.csv", "schema.yaml", "out.csv"])
    assert args.command == "synthesize"
    assert args.rows == 1000


def test_synthesize_custom_rows():
    args = _parse_args(["synthesize", "data.csv", "schema.yaml", "out.csv", "--rows", "500"])
    assert args.rows == 500


def test_validate_required_args():
    args = _parse_args(["validate", "real.csv", "synth.csv"])
    assert args.command == "validate"


def test_no_command_exits():
    with pytest.raises(SystemExit):
        _parse_args([])
