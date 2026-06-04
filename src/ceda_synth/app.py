"""Streamlit-app — ceda-synth. Dunne orchestrator."""

from __future__ import annotations

import streamlit as st
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

from ceda_synth.ui import config as cfg_ui
from ceda_synth.ui import datasource, results
from ceda_synth.ui.theme import inject_css


# ── Generate helpers ───────────────────────────────────────────────────────────
def _run_sequential(src: datasource.DataSource, cfg: cfg_ui.SequentialConfig) -> None:
    from sdv.sequential import PARSynthesizer

    with st.spinner("PAR-model trainen…"):
        try:
            model = PARSynthesizer(src.demo_meta, epochs=20, verbose=False)
            model.fit(src.df)
            st.session_state["synth"] = model.sample(num_sequences=cfg.n_sequences)
            st.session_state["n_label"] = f"{cfg.n_sequences} sequenties"
            st.session_state["n_generated"] = cfg.n_sequences
            st.session_state["col_types"] = None
            st.session_state["primary_key"] = None
            st.session_state["metadata_dict"] = src.demo_meta.to_dict()
        except Exception as exc:
            st.error(f"Fout bij genereren: {exc}")
            st.stop()


def _run_tabular(src: datasource.DataSource, cfg: cfg_ui.TabularConfig) -> None:
    from sdv.single_table import CTGANSynthesizer

    meta = SingleTableMetadata()
    for col_name, sdtype in cfg.col_types.items():
        meta.add_column(col_name, sdtype=sdtype)
    if cfg.primary_key:
        meta.set_primary_key(cfg.primary_key)

    label = (
        "CTGAN trainen (duurt langer)…"
        if cfg.synthesizer == "ctgan"
        else "Model trainen en data genereren…"
    )
    with st.spinner(label):
        try:
            model = (
                CTGANSynthesizer(meta, epochs=100)
                if cfg.synthesizer == "ctgan"
                else GaussianCopulaSynthesizer(meta)
            )
            model.fit(src.df)
            st.session_state["synth"] = model.sample(num_rows=cfg.n_rows)
            st.session_state["n_label"] = f"{cfg.n_rows:,} rijen"
            st.session_state["n_generated"] = cfg.n_rows
            st.session_state["col_types"] = cfg.col_types
            st.session_state["primary_key"] = cfg.primary_key
            st.session_state["synthesizer"] = cfg.synthesizer
            st.session_state["metadata_dict"] = meta.to_dict()
        except Exception as exc:
            st.error(f"Fout bij genereren: {exc}")
            st.stop()


# ── Pagina-setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="ceda-synth", page_icon="🔬", layout="wide")
inject_css()

col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown(
        "<div style='font-size:2.5rem;line-height:1;padding-top:6px'>🔬</div>",
        unsafe_allow_html=True,
    )
with col_title:
    st.markdown(
        "<h1 style='margin:0'>ceda-synth</h1>"
        "<p style='margin:0;color:#6B7280;font-size:.9rem'>"
        "Synthetische data genereren · CEDA/Npuls · "
        "powered by <a href='https://github.com/sdv-dev/SDV' target='_blank'>SDV</a></p>",
        unsafe_allow_html=True,
    )
st.divider()

# ── Databron ───────────────────────────────────────────────────────────────────
source = datasource.render()

if source.file_key != st.session_state.get("_file"):
    st.session_state.pop("synth", None)
    st.session_state["_file"] = source.file_key
    if source.modality == "sequential" and source.demo_meta:
        st.session_state["real_metadata_dict"] = source.demo_meta.to_dict()
    else:
        real_meta = SingleTableMetadata()
        real_meta.detect_from_dataframe(source.df)
        st.session_state["real_metadata_dict"] = real_meta.to_dict()

# ── Metrics + preview ──────────────────────────────────────────────────────────
m1, m2, m3 = st.columns(3)
m1.metric("Rijen", f"{len(source.df):,}")
m2.metric("Kolommen", len(source.df.columns))
m3.metric("Type", source.type_label)

with st.expander("Dataset preview", expanded=False):
    st.dataframe(source.df.head(10), use_container_width=True)

# ── Kolomtype-hints ────────────────────────────────────────────────────────────
if source.modality != "sequential":
    type_overrides = datasource.render_column_hints(source.df, source.file_key)
else:
    type_overrides = {}

# ── Configuratie ───────────────────────────────────────────────────────────────
if source.modality == "sequential":
    seq_cfg = cfg_ui.render_sequential(source.df, source.demo_meta)
else:
    tab_cfg = cfg_ui.render_tabular(source.df, type_overrides=type_overrides)

# ── Genereren ──────────────────────────────────────────────────────────────────
st.divider()
if st.button("Genereer synthetische data", type="primary", use_container_width=True):
    if source.modality == "sequential":
        _run_sequential(source, seq_cfg)
    else:
        _run_tabular(source, tab_cfg)

if "synth" not in st.session_state:
    st.stop()

st.success(f"✓ {st.session_state['n_label']} synthetische data gegenereerd")

# ── Resultaten ─────────────────────────────────────────────────────────────────
results.render(
    source.df,
    st.session_state["synth"],
    col_types=st.session_state.get("col_types"),
    primary_key=st.session_state.get("primary_key"),
    modality=source.modality,
    demo_name=source.demo_name,
    n_generated=st.session_state["n_generated"],
    synthesizer=st.session_state.get("synthesizer", "gaussian"),
    metadata_dict=st.session_state.get("metadata_dict"),
    real_metadata_dict=st.session_state.get("real_metadata_dict"),
)
