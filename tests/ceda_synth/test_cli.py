"""Tests voor CLI argument parsing en exitcodes."""

import pytest

from ceda_synth.cli import _parse_args


def test_app_command():
    args = _parse_args(["app"])
    assert args.command == "app"


def test_synthesize_required_args():
    args = _parse_args(["synthesize", "data.csv", "out.csv"])
    assert args.command == "synthesize"
    assert args.rows is None
    assert args.schema is None


def test_synthesize_with_options():
    args = _parse_args(["synthesize", "data.csv", "out.csv", "--schema", "s.yaml", "--rows", "500"])
    assert args.rows == 500
    assert str(args.schema) == "s.yaml"


def test_validate_required_args():
    args = _parse_args(["validate", "real.csv", "synth.csv"])
    assert args.command == "validate"


def test_no_command_exits():
    with pytest.raises(SystemExit):
        _parse_args([])
