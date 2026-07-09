"""PDF-export van het validatierapport.

Zet het machine-leesbare rapport uit :func:`build_validation_report` om in een leesbaar
PDF, zodat de gebruiker de uitkomsten kan delen of archiveren. Bewust met **reportlab**
(pure-pip, cross-platform, headless) — geen systeem-deps zoals weasyprint of wkhtmltopdf,
zodat het draait bij een lokale ``pip install`` op Windows én macOS.

De databron is dezelfde dict als de JSON-export, zodat PDF en JSON nooit uit elkaar lopen.
"""

from __future__ import annotations

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# Risiconiveau → kleur, gelijk aan de banner in de app (groen/oranje/rood).
_RISK_COLOR = {
    "laag": colors.HexColor("#2e7d32"),
    "matig": colors.HexColor("#ed6c02"),
    "hoog": colors.HexColor("#d32f2f"),
    "onbekend": colors.HexColor("#616161"),
}
_HEADER_BG = colors.HexColor("#f0f2f6")


def _table(data: list[list], styles: dict, col_widths=None) -> Table:
    """Bouw een tabel met een lichte header-rij en dunne rasterlijnen."""
    rows = [[Paragraph(str(c), styles["cell"]) for c in row] for row in data]
    table = Table(rows, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _HEADER_BG),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _scorecards(verdict: dict, has_temporal: bool) -> list[list[str]]:
    """De deeloordelen als tabelrijen — Tijdsgedrag vervangt Samenhang bij longitudinaal."""
    cards = [("Verdeling", verdict.get("verd_label"), verdict.get("verd_risk"))]
    if has_temporal:
        cards.append(("Tijdsgedrag", verdict.get("temp_label"), verdict.get("temp_risk")))
    else:
        cards.append(("Samenhang", verdict.get("corr_label"), verdict.get("corr_risk")))
    cards.append(("Privacy", verdict.get("priv_label"), verdict.get("priv_risk")))
    cards.append(("Bruikbaarheid", verdict.get("brk_label"), verdict.get("brk_risk")))
    return [["Dimensie", "Oordeel"]] + [[naam, label or "—"] for naam, label, _ in cards]


def build_report_pdf(report: dict, verdict: dict | None = None) -> bytes:
    """Genereer het validatierapport als PDF en geef de bytes terug.

    *report* is de dict uit :func:`build_validation_report`; *verdict* levert de
    gewone-taal-labels voor de oordeel-banner en de scorekaarten (optioneel, zodat de
    functie ook zonder UI-context een PDF kan opleveren).
    """
    base = getSampleStyleSheet()
    styles = {
        "title": base["Title"],
        "h2": base["Heading2"],
        "body": base["BodyText"],
        "cell": base["BodyText"],
        "small": base["Italic"],
    }

    story: list = [Paragraph("edu-synth — Validatierapport", styles["title"])]
    meta = (
        f"Gegenereerd op {report.get('generated_at', '—')} · "
        f"synthesizer: {report.get('synthesizer', '—')} · "
        f"SDV {report.get('sdv_version', '—')}"
    )
    story += [Paragraph(meta, styles["small"]), Spacer(1, 6 * mm)]

    if verdict:
        risk = verdict.get("brk_risk", "onbekend")
        story.append(
            Paragraph(
                f'<font color="{_RISK_COLOR.get(risk, _RISK_COLOR["onbekend"]).hexval()}">'
                f"<b>Oordeel: {verdict.get('brk_label', '—')}</b></font>",
                styles["h2"],
            )
        )
        story += [_table(_scorecards(verdict, "temporal" in report), styles), Spacer(1, 4 * mm)]

    if report.get("usage_recommendation"):
        story.append(Paragraph(f"<b>Gebruik:</b> {report['usage_recommendation']}", styles["body"]))
    if report.get("disclaimer"):
        story.append(Paragraph(report["disclaimer"], styles["small"]))
    story.append(Spacer(1, 5 * mm))

    story += _column_stats_section(report, styles)
    story += _temporal_section(report, styles)
    story += _privacy_section(report, styles)
    story += _params_section(report, styles)

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, title="edu-synth validatierapport")
    doc.build(story)
    return buffer.getvalue()


def _column_stats_section(report: dict, styles: dict) -> list:
    stats = report.get("column_stats") or []
    if not stats:
        return []
    rows = [["Kolom", "Type", "Score", "Metric", "OK"]]
    rows += [
        [
            r.get("column", "—"),
            r.get("dtype", "—"),
            _fmt(r.get("score")),
            r.get("metric", "—"),
            _ok_symbol(r.get("ok")),
        ]
        for r in stats
    ]
    return _section("Verdeling per kolom", rows, styles)


def _temporal_section(report: dict, styles: dict) -> list:
    temporal = report.get("temporal")
    if not temporal or not temporal.get("available"):
        return []
    story = [
        Paragraph("Tijdsgedrag (longitudinaal)", styles["h2"]),
        Paragraph(
            f"Sequentielengte-afstand: {_fmt(temporal.get('length_distance'))} "
            f"({'binnen grens' if temporal.get('length_ok') else 'boven grens'}).",
            styles["body"],
        ),
    ]
    cols = temporal.get("columns") or []
    if cols:
        rows = [["Kolom", "Metriek", "Score", "OK"]]
        rows += [
            [
                r.get("column", "—"),
                r.get("kind", "—"),
                _fmt(r.get("score")),
                _ok_symbol(r.get("ok")),
            ]
            for r in cols
        ]
        story.append(_table(rows, styles))
    story.append(Spacer(1, 5 * mm))
    return story


def _privacy_section(report: dict, styles: dict) -> list:
    priv = report.get("privacy") or {}
    if not priv.get("available"):
        return []
    rows = [
        ["Maat", "Waarde"],
        ["DCR-ratio", _fmt(priv.get("dcr_ratio"))],
        ["NNDR (mediaan)", _fmt(priv.get("nndr_median"))],
        ["Risiconiveau", priv.get("risk_level", "—")],
    ]
    return _section("Privacy (DCR / NNDR)", rows, styles)


def _params_section(report: dict, styles: dict) -> list:
    labels = [
        ("Synthesizer", "synthesizer"),
        ("Trainingsrijen", "n_training_rows"),
        ("Gegenereerde rijen", "n_generated_rows"),
        ("Random seed", "random_seed"),
        ("Beoogd gebruik", "intended_use"),
        ("SDV-versie", "sdv_version"),
    ]
    rows = [["Parameter", "Waarde"]]
    rows += [
        [label, str(report.get(key, "—")) if report.get(key) is not None else "—"]
        for label, key in labels
    ]
    return _section("Reproductie & parameters", rows, styles, spacer=False)


def _fmt(value: object) -> str:
    """Nummers op 3 decimalen; niet-nummers ongewijzigd (of '—' bij None)."""
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return str(value)


def _ok_symbol(ok: bool | None) -> str:
    """✓ bij geslaagd, ✗ bij gezakt, — als de kolom geen ok-oordeel heeft."""
    if ok is None:
        return "—"
    return "✓" if ok else "✗"


def _section(title: str, rows: list[list], styles: dict, spacer: bool = True) -> list:
    """Eén rapport-sectie: kop, tabel en (optioneel) witruimte eronder."""
    block = [Paragraph(title, styles["h2"]), _table(rows, styles)]
    if spacer:
        block.append(Spacer(1, 5 * mm))
    return block
