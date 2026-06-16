"""Databron UI — upload of SDV demo-data, plus kolomtype-hints."""

from __future__ import annotations

import io
from dataclasses import dataclass

import pandas as pd
import streamlit as st

_TYPE_OPTIONS: dict[str, str] = {
    "Tabular": "single_table",
    "Sequentieel": "sequential",
}

_MODALITY_LABEL: dict[str | None, str] = {
    "single_table": "Tabular",
    "sequential": "Sequentieel",
    None: "Tabular",
}

_SDTYPE_NL: dict[str, str] = {
    "categorical": "Categorisch",
    "numerical": "Numeriek",
    "datetime": "Datum/tijd",
    "id": "ID / tekst",
}


@dataclass
class DataSource:
    df: pd.DataFrame
    demo_meta: object | None
    file_key: str
    modality: str | None
    demo_name: str | None

    @property
    def type_label(self) -> str:
        return _MODALITY_LABEL.get(self.modality, "Tabular")


@st.cache_data
def _load_file(name: str, data: bytes) -> pd.DataFrame:
    buf = io.BytesIO(data)
    return pd.read_parquet(buf) if name.endswith(".parquet") else pd.read_csv(buf)


@st.cache_data
def _list_demos(modality: str) -> list[str]:
    from sdv.datasets.demo import get_available_demos

    return get_available_demos(modality=modality)["dataset_name"].tolist()


@st.cache_data
def _load_demo(name: str, modality: str) -> tuple[pd.DataFrame, object]:
    from sdv.datasets.demo import download_demo

    return download_demo(modality=modality, dataset_name=name)


def render() -> DataSource:
    source = st.radio("**Databron**", ["Upload bestand", "SDV demo-data"], horizontal=True)

    if source == "Upload bestand":
        uploaded = st.file_uploader(
            "Sleep een bestand hierheen of klik om te bladeren",
            type=["csv", "parquet"],
            label_visibility="collapsed",
        )
        st.caption(
            "🔒 Jouw data verlaat dit apparaat niet — "
            "verwerking vindt lokaal in je browser-sessie plaats."
        )
        if not uploaded:
            st.info("Upload een CSV- of Parquet-bestand om te beginnen.")
            st.stop()

        # Wis "all accepted" state als een nieuw bestand geladen wordt
        prev_key = st.session_state.get("_datasource_file_key")
        new_key = uploaded.name
        if prev_key != new_key:
            for k in list(st.session_state.keys()):
                if k.startswith("all_accepted_"):
                    del st.session_state[k]
            st.session_state["_datasource_file_key"] = new_key

        return DataSource(
            df=_load_file(uploaded.name, uploaded.getvalue()),
            demo_meta=None,
            file_key=uploaded.name,
            modality=None,
            demo_name=None,
        )

    c1, c2 = st.columns([1, 3])
    with c1:
        selected_type = st.radio("**Type**", list(_TYPE_OPTIONS.keys()))
    modality = _TYPE_OPTIONS[selected_type]

    with c2:
        demo_name = st.selectbox("**Dataset**", _list_demos(modality))

    df, demo_meta = _load_demo(demo_name, modality)
    return DataSource(
        df=df,
        demo_meta=demo_meta,
        file_key=f"{modality}/{demo_name}",
        modality=modality,
        demo_name=demo_name,
    )


def render_column_hints(df: pd.DataFrame, cache_key: str = "") -> dict[str, str]:
    """Toon kolomtype-hints en geef gebruikersgekozen overrides terug.

    Retourneert een dict {kolomnaam: sdtype} voor kolommen waarbij de gebruiker
    de suggestie heeft geaccepteerd of handmatig een type heeft gekozen.
    """
    from edu_synth.core.synthesize import infer_column_hints

    hints = infer_column_hints(df)
    suggestions = [h for h in hints if h.has_suggestion]
    warnings = [h for h in hints if not h.has_suggestion]

    if not hints:
        return {}

    overrides: dict[str, str] = {}

    if suggestions:
        all_key = f"all_accepted_{cache_key}"
        with st.expander(
            f"⚠️ {len(suggestions)} kolomtype(s) om te controleren",
            expanded=True,
        ):
            st.caption(
                "SDV heeft de onderstaande kolommen anders gedetecteerd dan verwacht. "
                "Controleer en pas aan waar nodig — dit heeft direct invloed op de kwaliteit."
            )

            col_btn1, col_btn2 = st.columns([2, 1])
            if col_btn1.button(
                "✓ Pas alle aanbevelingen toe",
                key=f"accept_all_{cache_key}",
                use_container_width=True,
            ):
                st.session_state[all_key] = True

            if st.session_state.get(all_key):
                if col_btn2.button(
                    "↩ Handmatig aanpassen",
                    key=f"manual_adjust_{cache_key}",
                    use_container_width=True,
                ):
                    st.session_state[all_key] = False
                    st.rerun()

            if st.session_state.get(all_key):
                st.success("Alle aanbevelingen zijn toegepast.")
                for hint in suggestions:
                    overrides[hint.name] = hint.suggested_sdtype
            else:
                for hint in suggestions:
                    c1, c2, c3 = st.columns([3, 3, 1])
                    c1.markdown(f"**{hint.name}**")
                    c1.caption(hint.reason)
                    choices = [hint.suggested_sdtype, hint.detected_sdtype]
                    choice = c2.radio(
                        f"type_{hint.name}",
                        choices,
                        format_func=lambda x: (
                            f"✅ {_SDTYPE_NL.get(x, x)} (aanbevolen)"
                            if x == hint.suggested_sdtype
                            else f"↩ {_SDTYPE_NL.get(x, x)} (origineel)"
                        ),
                        key=f"hint_{cache_key}_{hint.name}",
                        label_visibility="collapsed",
                    )
                    c3.metric("Zekerheid", f"{hint.confidence:.0%}")
                    overrides[hint.name] = choice

    if warnings:
        with st.expander(f"ℹ️ {len(warnings)} waarschuwing(en)", expanded=False):
            for hint in warnings:
                st.warning(f"**{hint.name}**: {hint.reason}")

    return overrides
