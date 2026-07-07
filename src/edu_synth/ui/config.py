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
    seed: int = 42
    # Per-kolom marginale verdeling (alleen niet-default entries); leeg = alles 'norm'.
    distributions: dict[str, str] = field(default_factory=dict)


@dataclass
class SequentialConfig:
    n_sequences: int
    seq_key: str | None = field(default=None)
    seq_idx: str | None = field(default=None)
    seed: int = 42


def _render_seed() -> int:
    """Geavanceerde optie (niveau 2): vaste seed voor reproduceerbare output."""
    with st.expander("Geavanceerd — reproduceerbaarheid", expanded=False):
        seed = int(
            st.number_input(
                "Random seed",
                min_value=0,
                max_value=2**32 - 1,
                value=42,
                step=1,
                help="Dezelfde seed + dezelfde data geeft identieke synthetische output. "
                "Wordt opgeslagen in de parameters en de gegenereerde Python-code.",
            )
        )
    return seed


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

    distributions = _render_distributions(df, col_types)

    n_rows = int(
        st.number_input("Aantal rijen", min_value=10, max_value=500_000, value=len(df), step=100)
    )
    seed = _render_seed()
    return TabularConfig(
        col_types=col_types,
        primary_key=primary_key,
        n_rows=n_rows,
        seed=seed,
        distributions=distributions,
    )


def _render_distributions(df: pd.DataFrame, col_types: dict[str, str]) -> dict[str, str]:
    """Niveau 2: marginale verdeling per numerieke kolom.

    Scheve/zero-inflated kolommen krijgen automatisch ``gaussian_kde`` als default —
    de normaalverdeling (SDV-default) faalt daar. De expander toont die keuze en laat
    'm overschrijven. Alleen niet-default keuzes ('norm') komen in het resultaat, zodat
    de synthesizer-aanroep minimaal blijft. De widgets renderen ook ingeklapt, dus de
    aanbeveling werkt door zonder dat de gebruiker de expander opent.
    """
    from edu_synth.core.synthesize import (
        DISTRIBUTION_CHOICES,
        recommend_numerical_distributions,
    )

    num_cols = [c for c, t in col_types.items() if t == "numerical"]
    if not num_cols:
        return {}

    # De scheefheid-scan is O(n) per kolom en hangt alleen van de data af. Bij een
    # grote upload zou 'm bij elke Streamlit-rerun herberekenen merkbaar laggen, dus
    # cachen we het resultaat per bestand en filteren we daarna op de numerieke kolommen.
    cache = st.session_state.setdefault("_dist_cache", {})
    file_key = st.session_state.get("_file")
    if cache.get("file") != file_key:
        cache["file"] = file_key
        cache["skewed"] = recommend_numerical_distributions(df, list(df.columns))
    recommended = {c: cache["skewed"][c] for c in num_cols if c in cache["skewed"]}

    chosen: dict[str, str] = {}
    n_rec = len(recommended)
    with st.expander(
        f"Verdelingen — scheve kolommen ({n_rec} aanbevolen)" if n_rec else "Verdelingen",
        expanded=False,
    ):
        st.caption(
            "Voor scheve of zero-inflated kolommen kiest de app automatisch "
            "`gaussian_kde`, die de echte vorm beter volgt dan een normaalverdeling. "
            "⭐ = aanbevolen. Pas hier per kolom aan."
        )
        grid = st.columns(3)
        for i, col in enumerate(num_cols):
            default = recommended.get(col, "norm")
            label = f"{col} ⭐" if col in recommended else col
            choice = grid[i % 3].selectbox(
                label,
                DISTRIBUTION_CHOICES,
                index=DISTRIBUTION_CHOICES.index(default),
                key=f"dist_{col}",
            )
            if choice != "norm":
                chosen[col] = choice
    return chosen


def _render_seq_column_selectors(
    df: pd.DataFrame, default_key: str | None, default_index: str | None
) -> tuple[str, str]:
    """Twee selectboxes voor sequence key + index, met gedetecteerde defaults.

    Gedeeld door de upload- en demo-flow zodat de gebruiker in beide gevallen de
    kolommen zelf kiest en corrigeert.
    """
    cols = list(df.columns)
    c1, c2 = st.columns(2)
    seq_key = c1.selectbox(
        "Sequence key — ID per entiteit",
        cols,
        index=cols.index(default_key) if default_key in cols else 0,
    )
    idx_default = default_index if default_index in cols else cols[0]
    seq_idx = c2.selectbox(
        "Sequence index — tijdkolom",
        cols,
        index=cols.index(idx_default),
    )
    if seq_key == seq_idx:
        st.warning("Kies verschillende kolommen voor sequence key en index.")
    return seq_key, seq_idx


def render_upload_sequential(df: pd.DataFrame) -> SequentialConfig | None:
    """Longitudinale upload-flow: detecteer + bevestig sequence key/index.

    Retourneert een SequentialConfig als de gebruiker (of de auto-detectie)
    longitudinaal kiest, anders None — dan volgt de gewone tabulaire flow.
    """
    from edu_synth.core.synthesize import infer_sequence_columns

    looks, default_key, default_index = infer_sequence_columns(df)
    is_longitudinal = (
        st.radio(
            "Heeft elke entiteit (student/instelling) meerdere rijen over de tijd?",
            ["Nee", "Ja — longitudinale data"],
            index=1 if looks else 0,
            horizontal=True,
            key="q_longitudinal",
        )
        == "Ja — longitudinale data"
    )
    if not is_longitudinal:
        return None

    with st.expander("Longitudinale configuratie", expanded=True):
        st.caption(
            "Voor longitudinale data gebruikt de app de PAR-synthesizer, die de "
            "volgorde per entiteit behoudt. Training kan enkele minuten duren."
        )
        seq_key, seq_idx = _render_seq_column_selectors(df, default_key, default_index)

    n_sequences = int(
        st.number_input(
            "Aantal sequenties (entiteiten)",
            min_value=1,
            max_value=100_000,
            value=min(100, df[seq_key].nunique()),
        )
    )
    seed = _render_seed()
    return SequentialConfig(n_sequences=n_sequences, seq_key=seq_key, seq_idx=seq_idx, seed=seed)


_INEQ_OPERATORS = ["≤", "<", "≥", ">"]


def inequality_rule(low_col: str, operator: str, high_col: str) -> dict:
    """Vertaal een 'A operator B'-keuze naar een inequality-rule-dict.

    ``≤``/``<``: low=A, high=B. ``≥``/``>``: kolommen omgedraaid. ``strict`` bij
    ``<`` en ``>``. Vorm gelijk aan het schema-``constraints``-blok, zodat
    ``core.synthesize.build_constraints`` het kan vertalen naar SDV-cag.
    """
    if operator in ("≥", ">"):
        low_col, high_col = high_col, low_col
    return {
        "type": "inequality",
        "low": low_col,
        "high": high_col,
        "strict": operator in ("<", ">"),
    }


def render_constraints(df: pd.DataFrame) -> list[dict]:
    """Niveau 2: point-and-click logische regels tussen kolommen.

    Retourneert rule-dicts (zelfde vorm als het schema-``constraints``-blok). Geen
    voor-validatie: ongeldige of botsende regels worden afgevangen bij het genereren.
    """
    cols = list(df.columns)
    rules: list[dict] = []

    with st.expander("Logische regels (optioneel)", expanded=False):
        st.caption(
            "Regels tussen kolommen die de tool niet uit de data kan afleiden. "
            "Laat leeg als je niets wilt afdwingen."
        )

        st.markdown("**Volgorde tussen kolommen**")
        st.caption("Bijvoorbeeld: een einddatum hoort ≥ de startdatum te zijn.")
        saved: list[dict] = st.session_state.setdefault("_ineq_rules", [])
        for idx, rule in enumerate(list(saved)):
            symbol = "<" if rule["strict"] else "≤"
            c_txt, c_del = st.columns([5, 1])
            c_txt.markdown(f"`{rule['low']}` {symbol} `{rule['high']}`")
            if c_del.button("❌", key=f"del_ineq_{idx}", help="Regel verwijderen"):
                saved.pop(idx)
                st.rerun()

        c1, c2, c3, c4 = st.columns([3, 2, 3, 2])
        col_a = c1.selectbox("Kolom A", cols, key="ineq_a", label_visibility="collapsed")
        op = c2.selectbox("Operator", _INEQ_OPERATORS, key="ineq_op", label_visibility="collapsed")
        col_b = c3.selectbox("Kolom B", cols, key="ineq_b", label_visibility="collapsed")
        if c4.button("➕ Regel", use_container_width=True):
            if col_a == col_b:
                st.warning("Kies twee verschillende kolommen.")
            else:
                saved.append(inequality_rule(col_a, op, col_b))
                st.rerun()
        rules.extend(saved)

        st.markdown("**Geldige combinaties**")
        st.caption("Houd alleen kolomcombinaties die in de echte data voorkomen.")
        fixed = st.multiselect(
            "Houd deze kolommen logisch bij elkaar",
            cols,
            key="fixed_combo",
            label_visibility="collapsed",
        )
        if len(fixed) >= 2:
            rules.append({"type": "fixed_combinations", "columns": list(fixed)})

    return rules


def render_sequential(df: pd.DataFrame, demo_meta: object) -> SequentialConfig:
    """Longitudinale demo-flow: detecteer + bevestig sequence key/index.

    SDV's demo-metadata heeft niet altijd een sequence key gezet (bv.
    SelfRegulationSCP1 → PAR faalt met "geen sequence key"). We nemen de key/index
    uit de demo-metadata over als die er zijn, en detecteren anders zelf — net als
    bij een upload — zodat de gebruiker altijd geldige kolommen kiest.
    """
    from edu_synth.core.synthesize import infer_sequence_columns

    table_meta = list(demo_meta.tables.values())[0]
    _, det_key, det_index = infer_sequence_columns(df)
    default_key = table_meta.sequence_key or det_key
    default_index = table_meta.sequence_index or det_index

    with st.expander("Sequentie-configuratie", expanded=True):
        seq_key, seq_idx = _render_seq_column_selectors(df, default_key, default_index)

    n_sequences = int(
        st.number_input(
            "Aantal sequenties",
            min_value=1,
            max_value=10_000,
            value=min(10, df[seq_key].nunique()),
        )
    )
    seed = _render_seed()
    return SequentialConfig(n_sequences=n_sequences, seq_key=seq_key, seq_idx=seq_idx, seed=seed)
