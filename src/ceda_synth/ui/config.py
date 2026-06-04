"""Configuratie UI — kolomtypes, synthesizer-keuze en sequentie-instellingen."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import streamlit as st
from sdv.metadata import SingleTableMetadata

_SDTYPES = ["categorical", "numerical", "datetime", "id"]


@dataclass
class TabularConfig:
    col_types: dict[str, str]
    primary_key: str | None
    n_rows: int
    synthesizer: str = "gaussian"


@dataclass
class SequentialConfig:
    n_sequences: int
    seq_key: str | None = field(default=None)
    seq_idx: str | None = field(default=None)


def render_tabular(
    df: pd.DataFrame,
    type_overrides: dict[str, str] | None = None,
) -> TabularConfig:
    detected = SingleTableMetadata()
    detected.detect_from_dataframe(df)
    overrides = type_overrides or {}

    with st.expander("Kolomtypes aanpassen (optioneel)", expanded=False):
        grid = st.columns(3)
        col_types: dict[str, str] = {}
        for i, col_name in enumerate(df.columns):
            sdtype = overrides.get(
                col_name,
                detected.columns.get(col_name, {}).get("sdtype", "categorical"),
            )
            idx = _SDTYPES.index(sdtype) if sdtype in _SDTYPES else 0
            col_types[col_name] = grid[i % 3].selectbox(
                col_name, _SDTYPES, index=idx, key=f"t_{col_name}"
            )
        pk_opts = ["(geen)"] + list(df.columns)
        raw_pk = st.selectbox("Primaire sleutel", pk_opts)
        primary_key: str | None = None if raw_pk == "(geen)" else raw_pk

    # ── Synthesizer-keuze (guided) ──────────────────────────────────────────
    st.markdown("**Synthesizer**")
    c1, c2, c3 = st.columns(3)
    large = (
        c1.radio(
            "Dataset groter dan 50.000 rijen?",
            ["Nee", "Ja"],
            horizontal=True,
            key="q_large",
        )
        == "Ja"
    )
    complex_r = (
        c2.radio(
            "Complexe, niet-lineaire verbanden?",
            ["Nee — meeste onderwijsdata", "Ja — specifieke domeinkeuze"],
            horizontal=True,
            key="q_complex",
        )
        != "Nee — meeste onderwijsdata"
    )
    longitudinal = (
        c3.radio(
            "Heeft elke entiteit (student/instelling) meerdere rijen over de tijd?",
            ["Nee", "Ja — longitudinale data"],
            horizontal=True,
            key="q_longitudinal",
        )
        == "Ja — longitudinale data"
    )

    if longitudinal:
        st.warning(
            "⚠️ Longitudinale data: de tabular synthesizer behandelt elke rij als "
            "onafhankelijk en behoudt temporele samenhang niet. Voor betere resultaten: "
            "gebruik 'SDV demo-data → Sequentieel' of PAR Synthesizer in SDV direct."
        )

    recommended = "ctgan" if (large or complex_r) else "gaussian"
    rec_name = "CTGAN" if recommended == "ctgan" else "Gaussian Copula"
    rec_reason = (
        "beter voor grote datasets en complexe verbanden — traint langer (±2 min)"
        if recommended == "ctgan"
        else "snel en stabiel voor de meeste tabellaire onderwijsdata"
    )
    st.info(f"Aanbevolen: **{rec_name}** — {rec_reason}")

    synthesizer = recommended
    with st.expander("Geavanceerd: overschrijf synthesizer-keuze", expanded=False):
        override = st.radio(
            "Kies synthesizer:",
            ["Gaussian Copula", "CTGAN"],
            index=0 if recommended == "gaussian" else 1,
            key="synth_override",
            horizontal=True,
        )
        synthesizer = "gaussian" if override == "Gaussian Copula" else "ctgan"

    n_rows = int(
        st.number_input("Aantal rijen", min_value=10, max_value=500_000, value=len(df), step=100)
    )
    return TabularConfig(
        col_types=col_types,
        primary_key=primary_key,
        n_rows=n_rows,
        synthesizer=synthesizer,
    )


def render_sequential(df: pd.DataFrame, demo_meta: object) -> SequentialConfig:
    table_meta = list(demo_meta.tables.values())[0]
    seq_key = table_meta.sequence_key
    seq_idx = table_meta.sequence_index

    with st.expander("Sequentie-configuratie", expanded=True):
        c1, c2 = st.columns(2)
        c1.text_input("Sequence key", value=seq_key or "(niet ingesteld)", disabled=True)
        c2.text_input("Sequence index", value=seq_idx or "(niet ingesteld)", disabled=True)

    n_sequences = int(
        st.number_input(
            "Aantal sequenties",
            min_value=1,
            max_value=10_000,
            value=min(10, df[seq_key].nunique()) if seq_key else 10,
        )
    )
    return SequentialConfig(n_sequences=n_sequences, seq_key=seq_key, seq_idx=seq_idx)
