"""Npuls huisstijl — kleuren, CSS-injectie en Plotly helper."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

NPULS: dict[str, str] = {
    "blauw": "#3D68EC",
    "oranje": "#DD784B",
    "groen": "#00AF81",
    "geel": "#F4D74B",
    "roze": "#F4D9DC",
    "zwart": "#000000",
}

PALETTE: list[str] = [
    NPULS["blauw"],
    NPULS["oranje"],
    NPULS["groen"],
    NPULS["geel"],
    NPULS["roze"],
    NPULS["zwart"],
]

_CSS_PATH = Path(__file__).parent.parent / "assets" / "streamlit-custom.css"


def inject_css() -> None:
    if _CSS_PATH.exists():
        st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


def apply_plotly_style(fig: object, height: int = 300) -> object:
    fig.update_layout(
        height=height,
        margin=dict(t=40, b=20, l=10, r=10),
        legend_title_text="",
        font_family="Inter, 'General Sans', sans-serif",
        font_color=NPULS["zwart"],
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#F3F4F6")
    return fig
