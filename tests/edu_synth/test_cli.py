"""Tests voor CLI argument parsing en exitcodes."""

import os

import pytest

from edu_synth import cli
from edu_synth.cli import _parse_args


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


def _stub_streamlit(monkeypatch):
    """Voorkom dat _cmd_app de echte Streamlit-app start."""
    import streamlit.web.cli as stcli

    monkeypatch.setattr(stcli, "main", lambda: 0)


def test_app_sets_theme_env_without_forcing_headless(monkeypatch):
    _stub_streamlit(monkeypatch)
    monkeypatch.delenv("STREAMLIT_THEME_PRIMARY_COLOR", raising=False)
    monkeypatch.delenv("STREAMLIT_SERVER_HEADLESS", raising=False)

    with pytest.raises(SystemExit):
        cli._cmd_app()

    # huisstijl reist mee, maar headless laten we aan de omgeving over
    assert os.environ["STREAMLIT_THEME_PRIMARY_COLOR"] == "#3D68EC"
    assert "STREAMLIT_SERVER_HEADLESS" not in os.environ


def test_app_respects_existing_env(monkeypatch):
    _stub_streamlit(monkeypatch)
    monkeypatch.setenv("STREAMLIT_THEME_PRIMARY_COLOR", "#123456")

    with pytest.raises(SystemExit):
        cli._cmd_app()

    assert os.environ["STREAMLIT_THEME_PRIMARY_COLOR"] == "#123456"
