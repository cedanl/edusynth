"""Streamlit-app — edu-synth. Dunne orchestrator."""

from __future__ import annotations

import streamlit as st
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

from edu_synth.core.synthesize import build_constraints, detect_datetime_format, set_seed
from edu_synth.ui import config as cfg_ui
from edu_synth.ui import datasource, results, stepper
from edu_synth.ui.theme import inject_css

# ── Generate helpers ───────────────────────────────────────────────────────────


def _run_sequential(src: datasource.DataSource, cfg: cfg_ui.SequentialConfig) -> None:
    from sdv.sequential import PARSynthesizer

    from edu_synth.core.synthesize import build_sequential_metadata

    with st.spinner("PAR-model trainen (kan enkele minuten duren)…"):
        try:
            set_seed(cfg.seed)
            # Demo-data brengt eigen metadata mee; een upload bouwen we zelf op.
            if src.demo_meta is not None:
                metadata = src.demo_meta
                seq_info = None
            else:
                metadata = build_sequential_metadata(src.df, cfg.seq_key, cfg.seq_idx)
                seq_info = {
                    "key": cfg.seq_key,
                    "index": cfg.seq_idx,
                    "index_sdtype": metadata.tables["data"].columns[cfg.seq_idx]["sdtype"],
                }
            model = PARSynthesizer(metadata, epochs=128, verbose=False)
            model.fit(src.df)
            st.session_state["synth"] = model.sample(num_sequences=cfg.n_sequences)
            st.session_state["n_label"] = f"{cfg.n_sequences} sequenties"
            st.session_state["n_generated"] = cfg.n_sequences
            st.session_state["col_types"] = None
            st.session_state["primary_key"] = None
            st.session_state["random_seed"] = cfg.seed
            st.session_state["modality"] = "sequential"
            st.session_state["seq_info"] = seq_info
            st.session_state["metadata_dict"] = metadata.to_dict()
        except Exception as exc:
            st.error(f"Fout bij genereren: {exc}")
            st.stop()


def _run_tabular(
    src: datasource.DataSource,
    cfg: cfg_ui.TabularConfig,
    constraints: list[dict] | None = None,
) -> None:
    meta = SingleTableMetadata()
    for col_name, sdtype in cfg.col_types.items():
        # Datumkolommen: detecteer het formaat uit de data en geef het mee, anders
        # valt SDV terug op ISO 8601 en faalt het op DUO-datums (YYYYMMDD).
        if sdtype == "datetime":
            fmt = detect_datetime_format(src.df[col_name])
            if fmt is not None:
                meta.add_column(col_name, sdtype="datetime", datetime_format=fmt)
                continue
        meta.add_column(col_name, sdtype=sdtype)
    if cfg.primary_key:
        meta.set_primary_key(cfg.primary_key)

    with st.spinner("Model trainen en data genereren…"):
        try:
            set_seed(cfg.seed)
            model = GaussianCopulaSynthesizer(meta)
            cag = build_constraints(constraints or [])
            if cag:
                model.add_constraints(constraints=cag)
            model.fit(src.df)
            st.session_state["synth"] = model.sample(num_rows=cfg.n_rows)
            st.session_state["n_label"] = f"{cfg.n_rows:,} rijen"
            st.session_state["n_generated"] = cfg.n_rows
            st.session_state["col_types"] = cfg.col_types
            st.session_state["primary_key"] = cfg.primary_key
            st.session_state["random_seed"] = cfg.seed
            st.session_state["modality"] = "single_table"
            st.session_state["seq_info"] = None
            st.session_state["metadata_dict"] = meta.to_dict()
        except Exception as exc:
            st.error(f"Fout bij genereren: {exc}")
            st.stop()


# ── Pagina-setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="edu-synth", page_icon="🔬", layout="wide")
inject_css()

col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown(
        "<div style='font-size:2.5rem;line-height:1;padding-top:6px'>🔬</div>",
        unsafe_allow_html=True,
    )
with col_title:
    st.markdown(
        "<h1 style='margin:0'>edu-synth</h1>"
        "<p style='margin:0;color:#6B7280;font-size:.9rem'>"
        "Synthetische data genereren · CEDA/Npuls · "
        "powered by <a href='https://github.com/sdv-dev/SDV' target='_blank'>SDV</a></p>",
        unsafe_allow_html=True,
    )
st.divider()

# ── Wizard-state ─────────────────────────────────────────────────────────────────
# Echte wizard: precies één stap zichtbaar. De gekozen dataset bewaren we in
# session_state["src"], zodat stap 2 en 3 'm niet opnieuw hoeven te tekenen.
step = int(st.session_state.setdefault("step", 1))
synth_ready = "synth" in st.session_state
if step == 3 and not synth_ready:
    step = st.session_state["step"] = 2
stepper.render(step)


def _goto(target: int) -> None:
    st.session_state["step"] = target
    st.rerun()


# ── Stap 1 — Data laden ──────────────────────────────────────────────────────────
if step == 1:
    st.markdown("#### ① Data laden")
    picked = datasource.render()

    if picked is not None:
        if picked.file_key != st.session_state.get("_file"):
            # Nieuwe dataset: vorige synthese én logische regels vervallen
            # (die verwijzen naar kolommen van het oude bestand).
            st.session_state.pop("synth", None)
            st.session_state.pop("_ineq_rules", None)
            st.session_state.pop("fixed_combo", None)
            st.session_state["_file"] = picked.file_key
            if picked.modality == "sequential" and picked.demo_meta:
                st.session_state["real_metadata_dict"] = picked.demo_meta.to_dict()
            else:
                real_meta = SingleTableMetadata()
                real_meta.detect_from_dataframe(picked.df)
                st.session_state["real_metadata_dict"] = real_meta.to_dict()
        st.session_state["src"] = picked

    src = st.session_state.get("src")
    if src is None:
        st.stop()

    if picked is None:
        st.caption(
            f"Huidige selectie: **{src.file_key}** — kies hierboven een ander "
            "bestand om te vervangen, of ga verder."
        )
    m1, m2, m3 = st.columns(3)
    m1.metric("Rijen", f"{len(src.df):,}")
    m2.metric("Kolommen", len(src.df.columns))
    m3.metric("Type", src.type_label)
    with st.expander("Dataset preview", expanded=False):
        st.dataframe(src.df.head(10), use_container_width=True)

    if st.button("Volgende → Genereren", type="primary", use_container_width=True):
        _goto(2)
    st.stop()

# Vanaf hier is er altijd een dataset.
src = st.session_state["src"]

# ── Stap 2 — Genereren ───────────────────────────────────────────────────────────
# Kolomtypes en synthesizer-instellingen zijn optionele verfijning: bij demo- of
# schone data klopt de auto-detectie en klik je meteen op Genereer. Twijfelachtige
# type-suggesties (<90%) komen vanzelf bovenaan te staan (render_column_hints).
if step == 2:
    st.markdown("#### ② Genereren")
    # Een eigen upload kan longitudinaal zijn → PAR. Demo-data heeft een vaste
    # modaliteit. De longitudinale keuze bepaalt de flow, dus die staat vooraan.
    upload_seq_cfg = cfg_ui.render_upload_sequential(src.df) if src.modality is None else None
    is_sequential = src.modality == "sequential" or upload_seq_cfg is not None

    if is_sequential:
        type_overrides = {}
        if src.modality == "sequential":
            st.caption("Longitudinale demo-data — kolomtypes komen uit de meegeleverde metadata.")
            seq_cfg = cfg_ui.render_sequential(src.df, src.demo_meta)
        else:
            seq_cfg = upload_seq_cfg
    else:
        type_overrides = datasource.render_column_hints(src.df, src.file_key)
        tab_cfg = cfg_ui.render_tabular(src.df, type_overrides=type_overrides)
        constraint_rules = cfg_ui.render_constraints(src.df)

    if st.button("Genereer synthetische data", type="primary", use_container_width=True):
        if is_sequential:
            _run_sequential(src, seq_cfg)
        else:
            _run_tabular(src, tab_cfg, constraint_rules)
        _goto(3)

    c_back, c_fwd = st.columns(2)
    if c_back.button("← Vorige", use_container_width=True):
        _goto(1)
    if synth_ready and c_fwd.button("Naar resultaten →", use_container_width=True):
        _goto(3)
    st.stop()

# ── Stap 3 — Resultaten ──────────────────────────────────────────────────────────
st.markdown("#### ③ Resultaten")
st.success(f"✓ {st.session_state['n_label']} synthetische data gegenereerd")

results.render(
    src.df,
    st.session_state["synth"],
    col_types=st.session_state.get("col_types"),
    primary_key=st.session_state.get("primary_key"),
    modality=st.session_state.get("modality", src.modality),
    demo_name=src.demo_name,
    n_generated=st.session_state["n_generated"],
    random_seed=st.session_state.get("random_seed"),
    seq_info=st.session_state.get("seq_info"),
    metadata_dict=st.session_state.get("metadata_dict"),
    real_metadata_dict=st.session_state.get("real_metadata_dict"),
)

st.divider()
if st.button("← Terug naar instellingen", use_container_width=True):
    _goto(2)
