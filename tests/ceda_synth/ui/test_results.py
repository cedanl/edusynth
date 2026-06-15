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


def test_build_code_upload_sequential_uses_par_on_own_csv():
    code = _build_code(
        col_types=None,
        primary_key=None,
        modality="sequential",
        demo_name=None,
        n_generated=50,
        sdv_version="1.37.0",
        seq_info={"key": "student_id", "index": "studiejaar", "index_sdtype": "numerical"},
    )
    assert "PARSynthesizer" in code
    assert "download_demo" not in code  # géén demo-code voor een upload
    assert 'set_sequence_key("student_id"' in code
    assert 'set_sequence_index("studiejaar"' in code
    assert "epochs=128" in code


def test_build_code_demo_sequential_uses_download_demo():
    code = _build_code(
        col_types=None,
        primary_key=None,
        modality="sequential",
        demo_name="nasdaq100_2019",
        n_generated=10,
        sdv_version="1.37.0",
        seq_info=None,
    )
    assert "download_demo" in code
    assert "epochs=128" in code
