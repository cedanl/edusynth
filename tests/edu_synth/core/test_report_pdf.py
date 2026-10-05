"""Tests voor report_pdf.py — PDF-export van het validatierapport."""

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph

from edu_synth.core.report_pdf import _correlations_section, build_report_pdf

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
        "consistency": [
            {
                "aspect": "duplicate",
                "label": "Dubbele rijen per entiteit en tijdstip",
                "column": None,
                "real": 0.0,
                "synth": 0.25,
                "score": 0.25,
                "ok": False,
            }
        ],
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


def _section_text(report: dict) -> str:
    """Alle alineatekst van de sectie 'Samenhang tussen kolommen'."""
    styles = {k: getSampleStyleSheet()["BodyText"] for k in ("h2", "body", "cell", "small")}
    story = _correlations_section(report, styles)
    return " ".join(p.getPlainText() for p in story if isinstance(p, Paragraph))


_PAIR = {"col_a": "ec", "col_b": "leeftijd", "real_corr": 0.6, "synth_corr": 0.1, "delta": 0.5}
_PAIR_TREND = {"Column 1": "status", "Column 2": "vorm", "Metric": "Contingency", "Score": 0.4}


def test_correlations_section_lists_flagged_pairs():
    report = {"correlations": {"available": True, "threshold": 0.1, "flagged": [_PAIR]}}
    text = _section_text(report)
    assert "Samenhang tussen kolommen" in text
    assert "1 correlatie(s)" in text
    assert build_report_pdf(report).startswith(b"%PDF-")


def test_correlations_section_longitudinal_note():
    note = "Deze correlaties gaan over de losse rijen."
    report = {"correlations": {"available": True, "threshold": 0.1, "flagged": [], "note": note}}
    text = _section_text(report)
    assert note in text
    assert "zijn bewaard" in text


def test_correlations_section_without_numeric_columns_shows_pair_trends():
    # Geen numerieke kolommen: de reden staat erin en de sdmetrics-paren vullen de sectie.
    report = {
        "correlations": {"available": False, "reason": "Minder dan 2 numerieke kolommen"},
        "sdmetrics": {"available": True, "column_pair_trends": [_PAIR_TREND]},
    }
    text = _section_text(report)
    assert "Minder dan 2 numerieke kolommen" in text
    assert "zwakste van 1" in text
    assert build_report_pdf(report).startswith(b"%PDF-")


def test_correlations_section_absent_without_data():
    assert _correlations_section({"generated_at": "2026-07-09"}, {}) == []
