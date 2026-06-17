"""Stepper — horizontale voortgangsindicator voor de 5-stappen flow."""

from __future__ import annotations

import streamlit as st

from edu_synth.ui.theme import NPULS

STEP_LABELS = ["Data laden", "Genereren", "Resultaten"]

_GREY_BG = "#E5E7EB"
_GREY_FG = "#9CA3AF"


def step_status(index: int, current: int) -> str:
    """Status van stap `index` (1-based) ten opzichte van de huidige stap."""
    if index < current:
        return "done"
    if index == current:
        return "active"
    return "future"


def _circle(index: int, status: str) -> str:
    if status == "done":
        bg, fg, mark = NPULS["groen"], "#fff", "✓"
    elif status == "active":
        bg, fg, mark = NPULS["blauw"], "#fff", str(index)
    else:
        bg, fg, mark = _GREY_BG, _GREY_FG, str(index)

    label = STEP_LABELS[index - 1]
    label_color = _GREY_FG if status == "future" else NPULS["zwart"]
    label_weight = "600" if status == "active" else "400"
    return (
        "<div style='display:flex;flex-direction:column;align-items:center;flex:1;min-width:0'>"
        f"<div style='width:32px;height:32px;border-radius:50%;background:{bg};color:{fg};"
        "display:flex;align-items:center;justify-content:center;font-weight:600;"
        f"font-size:.9rem'>{mark}</div>"
        f"<div style='margin-top:6px;font-size:.78rem;color:{label_color};"
        f"font-weight:{label_weight};text-align:center;line-height:1.15'>{label}</div>"
        "</div>"
    )


def render(current: int) -> None:
    """Toon de horizontale stepper; `current` is 1-based (1..5)."""
    parts: list[str] = []
    for i in range(1, len(STEP_LABELS) + 1):
        parts.append(_circle(i, step_status(i, current)))
        if i < len(STEP_LABELS):
            line = NPULS["groen"] if i < current else _GREY_BG
            parts.append(
                f"<div style='flex:0 0 24px;height:2px;background:{line};margin-top:16px'></div>"
            )

    html = (
        "<div style='display:flex;align-items:flex-start;justify-content:space-between;"
        "gap:4px;margin:2px 0 10px'>" + "".join(parts) + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)
