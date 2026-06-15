"""Tests voor results.py — gegenereerde reproductiecode."""

from ceda_synth.ui.results import _build_code


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
