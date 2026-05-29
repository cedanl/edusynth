"""Tests voor synthesize.py — schema laden en metadata opbouwen."""

from pathlib import Path

import pytest
from edusynth.synthesize import _load_schema, _build_metadata


FIXTURE_SCHEMA = Path(__file__).parent.parent / "fixtures" / "mini_schema.yaml"


def test_load_schema_returns_dict():
    schema = _load_schema(FIXTURE_SCHEMA)
    assert isinstance(schema, dict)
    assert "columns" in schema


def test_build_metadata_sets_primary_key():
    schema = _load_schema(FIXTURE_SCHEMA)
    metadata = _build_metadata(schema)
    assert metadata.primary_key is not None
