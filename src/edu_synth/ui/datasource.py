"""Databron UI — upload of SDV demo-data, plus kolomtype-hints."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from edu_synth.core.synthesize import ColumnHint

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

# Drempel waarboven een type-suggestie zeker genoeg is om in bulk toe te passen.
# Suggesties eronder (bijv. 0.65/0.7) worden nooit blind overgenomen.
_HIGH_CONFIDENCE = 0.9


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
        st.info(
            "🔒 Verwerking vindt volledig lokaal in je browser-sessie plaats. "
            "Je data wordt nergens opgeslagen of verzonden."
        )
        consent = st.checkbox(
            "Ik bevestig dat ik toestemming heb om deze data te verwerken "
            "voor synthese-doeleinden.",
            key="_privacy_consent",
        )
        with st.popover("Wat telt als persoonsgegevens?"):
            st.markdown(
                "Persoonsgegevens zijn alle gegevens over een identificeerbaar "
                "persoon: naam, studentnummer, e-mailadres, geboortedatum, maar "
                "ook combinaties die samen tot één persoon herleidbaar zijn "
                "(bijvoorbeeld opleiding + cohort + postcode). Verwerk alleen "
                "data waarvoor je een grondslag hebt."
            )

        uploaded = st.file_uploader(
            "Sleep een bestand hierheen of klik om te bladeren",
            type=["csv", "parquet"],
            label_visibility="collapsed",
            disabled=not consent,
        )
        if not consent:
            st.caption("Bevestig eerst de toestemming hierboven om te uploaden.")
            st.stop()
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


def partition_by_confidence(
    suggestions: list[ColumnHint], threshold: float = _HIGH_CONFIDENCE
) -> tuple[list[ColumnHint], list[ColumnHint]]:
    """Splits type-suggesties in (zeker ≥ drempel, onzeker < drempel).

    Onzekere suggesties mogen nooit blind in bulk worden toegepast; de gebruiker
    bevestigt die handmatig.
    """
    high = [h for h in suggestions if h.confidence >= threshold]
    low = [h for h in suggestions if h.confidence < threshold]
    return high, low


def _render_hint_row(hint: ColumnHint, cache_key: str, *, uncertain: bool = False) -> str:
    """Render één kolomtype-keuze (radio) en geef de gekozen sdtype terug."""
    c1, c2, c3 = st.columns([3, 3, 1])
    marker = "🔸 " if uncertain else ""
    c1.markdown(f"**{marker}{hint.name}**")
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
    return choice


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
        high_conf, low_conf = partition_by_confidence(suggestions)
        all_key = f"all_accepted_{cache_key}"
        with st.expander(
            f"⚠️ {len(suggestions)} kolomtype(s) om te controleren",
            expanded=True,
        ):
            st.caption(
                "SDV heeft de onderstaande kolommen anders gedetecteerd dan verwacht. "
                "Controleer en pas aan waar nodig — dit heeft direct invloed op de kwaliteit."
            )

            # Alleen de zekere suggesties (≥90%) kunnen in bulk worden toegepast.
            applied_all = bool(high_conf) and st.session_state.get(all_key, False)
            if high_conf:
                col_btn1, col_btn2 = st.columns([2, 1])
                if col_btn1.button(
                    f"✓ Pas {len(high_conf)} zekere aanbeveling(en) toe",
                    key=f"accept_all_{cache_key}",
                    use_container_width=True,
                ):
                    st.session_state[all_key] = True
                    applied_all = True

                if applied_all and col_btn2.button(
                    "↩ Handmatig aanpassen",
                    key=f"manual_adjust_{cache_key}",
                    use_container_width=True,
                ):
                    st.session_state[all_key] = False
                    st.rerun()

            if applied_all:
                st.success(f"{len(high_conf)} zekere aanbeveling(en) toegepast.")
                for hint in high_conf:
                    overrides[hint.name] = hint.suggested_sdtype
            else:
                for hint in high_conf:
                    overrides[hint.name] = _render_hint_row(hint, cache_key)

            # Onzekere suggesties (<90%) altijd handmatig — nooit blind toepassen.
            if low_conf:
                st.warning(
                    f"🔸 {len(low_conf)} onzekere suggestie(s) — deze worden niet "
                    "automatisch toegepast. Controleer ze hieronder."
                )
                for hint in low_conf:
                    overrides[hint.name] = _render_hint_row(hint, cache_key, uncertain=True)

    if warnings:
        with st.expander(f"ℹ️ {len(warnings)} waarschuwing(en)", expanded=False):
            for hint in warnings:
                st.warning(f"**{hint.name}**: {hint.reason}")

    return overrides
