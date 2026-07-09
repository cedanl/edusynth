"""Tests voor report_pdf.py — PDF-export van het validatierapport."""

from edu_synth.core.report_pdf import build_report_pdf

_REPORT = {
    "generated_at": "2026-07-09",
    "synthesizer": "par",
    "sdv_version": "1.37.0",
    "n_training_rows": 100,
    "n_generated_rows": 100,
    "random_seed": 42,
    "usage_recommendation": "Bruikbaar met voorbehoud.",
    "disclaimer": "Meet statistische gelijkenis, geen privacygarantie.",
    "column_stats": [
        {
            "column": "leeftijd",
            "dtype": "numerical",
            "score": 0.05,
            "metric": "wasserstein",
            "ok": True,
        },
        {"column": "status", "dtype": "categorical", "score": 0.30, "metric": "tv", "ok": False},
    ],
    "privacy": {"available": True, "dcr_ratio": 1.1, "nndr_median": 0.9, "risk_level": "laag"},
    "temporal": {
        "available": True,
        "length_distance": 0.1,
        "length_ok": True,
        "columns": [{"column": "status", "kind": "transition", "score": 0.3, "ok": False}],
    },
}

_VERDICT = {
    "brk_label": "Bruikbaar met voorbehoud",
    "brk_risk": "matig",
    "verd_label": "Goed",
    "verd_risk": "laag",
    "temp_label": "Let op",
    "temp_risk": "matig",
    "priv_label": "Laag risico",
    "priv_risk": "laag",
}


def test_build_report_pdf_returns_pdf_bytes():
    pdf = build_report_pdf(_REPORT, _VERDICT)
    assert isinstance(pdf, bytes)
    assert pdf.startswith(b"%PDF-")  # geldig PDF-magic getal
    assert len(pdf) > 1000


def test_build_report_pdf_without_verdict():
    # Zonder UI-context (geen verdict) moet de PDF nog steeds genereren.
    pdf = build_report_pdf(_REPORT)
    assert pdf.startswith(b"%PDF-")


def test_build_report_pdf_minimal_report():
    # Alleen de verplichte velden — geen privacy/temporal/stats.
    pdf = build_report_pdf({"generated_at": "2026-07-09", "synthesizer": "gaussian"})
    assert pdf.startswith(b"%PDF-")
