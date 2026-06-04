"""Resultaten UI — tabs voor validatie, distributies en download+reproductie."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

from ceda_synth.core.validate import (
    PairsReport,
    Report,
    evaluate,
    evaluate_pairs,
    evaluate_privacy,
    usage_recommendation,
)
from ceda_synth.ui.theme import NPULS, apply_plotly_style

# ── Verdict-helpers ────────────────────────────────────────────────────────────
_RISK_ICON = {"laag": "✅", "matig": "⚠️", "hoog": "❌", "onbekend": "○"}
_RISK_MSG = {"laag": st.success, "matig": st.warning, "hoog": st.error}


def _verdeling_verdict(report: Report) -> tuple[str, str]:
    tv_rows = [r for r in report.rows if r.get("metric") == "tv"]
    if not tv_rows:
        return "Alleen numeriek", "laag"
    max_tv = max(r["distance"] for r in tv_rows)
    n_failed = sum(1 for r in tv_rows if not r.get("ok", True))
    if max_tv < 0.1:
        return "Uitstekend", "laag"
    if n_failed == 0:
        return "Goed", "laag"
    if n_failed <= max(1, len(tv_rows) // 3):
        return "Matig", "matig"
    return "Let op", "hoog"


def _privacy_verdict(priv) -> tuple[str, str]:
    if not priv.available:
        return "Niet berekend", "onbekend"
    labels = {"laag": "Laag risico", "matig": "Matig risico", "hoog": "Hoog risico"}
    return labels.get(priv.risk_level, "Onbekend"), priv.risk_level


def _bruikbaarheid_verdict(verd_risk: str, priv_risk: str) -> tuple[str, str]:
    risks = {verd_risk, priv_risk}
    if "hoog" in risks:
        return "Niet aanbevolen", "hoog"
    if "matig" in risks or "onbekend" in risks:
        return "Bruikbaar met voorbehoud", "matig"
    return "Aanbevolen voor rapportages", "laag"


def _scorecard(col, label: str, verdict: str, risk: str) -> None:
    col.metric(label, f"{_RISK_ICON.get(risk, '○')} {verdict}")


# ── Download-dialog ────────────────────────────────────────────────────────────
_GEBRUIK_OPTIES = [
    "Selecteer beoogd gebruik…",
    "Intern onderzoek / analyse",
    "Rapportage aan BRON / DUO",
    "Delen met externe onderzoekspartner",
    "Technisch testen / softwareontwikkeling",
]


@st.dialog("Klaar om te downloaden", width="large")
def _download_dialog(csv_bytes: bytes, verdict: dict, recommendation: str) -> None:
    c1, c2, c3 = st.columns(3)
    _scorecard(c1, "Verdeling", verdict["verd_label"], verdict["verd_risk"])
    _scorecard(c2, "Privacy", verdict["priv_label"], verdict["priv_risk"])
    _scorecard(c3, "Bruikbaarheid", verdict["brk_label"], verdict["brk_risk"])
    st.caption(recommendation)
    st.divider()

    gebruik = st.selectbox("Beoogd gebruik van deze dataset", _GEBRUIK_OPTIES)
    confirmed = gebruik != _GEBRUIK_OPTIES[0]

    if confirmed:
        st.session_state["intended_use"] = gebruik

    st.download_button(
        "Bevestig en download (CSV)",
        csv_bytes,
        file_name="synthetisch.csv",
        mime="text/csv",
        disabled=not confirmed,
        use_container_width=True,
        type="primary",
    )
    st.caption(
        "Synthetische data vervangt geen anonimisering. "
        "Raadpleeg uw FG bij gebruik buiten de instelling."
    )


# ── Publieke interface ─────────────────────────────────────────────────────────
def render(
    df: pd.DataFrame,
    synth: pd.DataFrame,
    *,
    col_types: dict[str, str] | None,
    primary_key: str | None,
    modality: str | None,
    demo_name: str | None,
    n_generated: int,
    synthesizer: str = "gaussian",
    metadata_dict: dict | None = None,
    real_metadata_dict: dict | None = None,
) -> None:
    with st.spinner("Validatie en privacyanalyse berekenen…"):
        report = evaluate(df, synth)
        priv = evaluate_privacy(df, synth)
        pairs = evaluate_pairs(df, synth)

    verd_label, verd_risk = _verdeling_verdict(report)
    priv_label, priv_risk = _privacy_verdict(priv)
    brk_label, brk_risk = _bruikbaarheid_verdict(verd_risk, priv_risk)
    recommendation = usage_recommendation(report, priv)

    verdict = {
        "verd_label": verd_label,
        "verd_risk": verd_risk,
        "priv_label": priv_label,
        "priv_risk": priv_risk,
        "brk_label": brk_label,
        "brk_risk": brk_risk,
    }

    tab_val, tab_dist, tab_dl = st.tabs(
        ["Validatierapport", "Distributies", "Download & Reproductie"]
    )
    with tab_val:
        _render_validation(report, priv, verdict, recommendation, pairs, primary_key)
    with tab_dist:
        _render_distributions(df, synth)
    with tab_dl:
        _render_download(
            synth,
            col_types=col_types,
            primary_key=primary_key,
            modality=modality,
            demo_name=demo_name,
            n_generated=n_generated,
            synthesizer=synthesizer,
            verdict=verdict,
            recommendation=recommendation,
            metadata_dict=metadata_dict,
            real_metadata_dict=real_metadata_dict,
        )


# ── Validatie-tab ──────────────────────────────────────────────────────────────
def _render_validation(
    report: Report,
    priv,
    verdict: dict,
    recommendation: str,
    pairs: PairsReport,
    primary_key: str | None,
) -> None:
    c1, c2, c3 = st.columns(3)
    _scorecard(c1, "Verdeling", verdict["verd_label"], verdict["verd_risk"])
    _scorecard(c2, "Privacy", verdict["priv_label"], verdict["priv_risk"])
    _scorecard(c3, "Bruikbaarheid", verdict["brk_label"], verdict["brk_risk"])

    brk_risk = verdict["brk_risk"]
    _VERDICT_TEXT = {
        "laag": (
            "Deze synthetische dataset is **geschikt voor gebruik in rapportages en analyses**."
        ),
        "matig": (
            "Deze synthetische dataset is **bruikbaar met voorbehoud**. "
            "Controleer de details hieronder voor gebruik."
        ),
        "hoog": (
            "Deze synthetische dataset is **niet aanbevolen voor publicatie**. "
            "Bekijk de details en pas de syntheseinstellingen aan."
        ),
    }
    msg_fn = _RISK_MSG.get(brk_risk, st.info)
    msg_fn(_VERDICT_TEXT.get(brk_risk, ""))
    st.caption(f"**Gebruik:** {recommendation}")

    st.divider()

    with st.expander("Statistisch detail — verdeling per kolom", expanded=False):
        rdf = report.to_dataframe()
        if "ok" in rdf.columns:
            rdf = rdf.copy()
            rdf["ok"] = rdf["ok"].map({True: "✓", False: "✗"})
        # Toon modal warnings per rij
        if "modal_warning" in rdf.columns:
            for _, row in rdf[rdf["modal_warning"].notna()].iterrows():
                st.warning(f"⚠️ **{row['column']}**: bimodaliteit — {row['modal_warning']}")
            rdf = rdf.drop(columns=["modal_warning"])
        st.dataframe(rdf, use_container_width=True)
        st.caption(
            "TV-afstand: < 0.1 = uitstekend · < 0.2 = goed · > 0.2 = let op. "
            "Wasserstein-schaal hangt af van de kolomwaarden. "
            "**Deze metrieken meten statistische gelijkenis, niet privacy.**"
        )

    with st.expander("Privacydetail — DCR / NNDR", expanded=False):
        if not priv.available:
            st.info(f"Privacyanalyse niet beschikbaar: {priv.reason}")
        else:
            p1, p2, p3 = st.columns(3)
            p1.metric("DCR synth (mediaan)", f"{priv.dcr_synth_median:.4f}")
            p2.metric("DCR holdout (mediaan)", f"{priv.dcr_holdout_median:.4f}")
            p3.metric("DCR-ratio", f"{priv.dcr_ratio:.2f}")
            st.metric("NNDR mediaan", f"{priv.nndr_median:.2f}")
            _PRIV_TEXT = {
                "laag": (
                    "Synthetische rijen gedragen zich als onbekende data — "
                    "ze staan even ver van de trainingsdata als een echte holdout-set."
                ),
                "matig": (
                    "Synthetische rijen liggen iets dichter bij de trainingsdata "
                    "dan een holdout-set. "
                    "Beoordeel de dataset op quasi-identifiers vóór publicatie."
                ),
                "hoog": (
                    "Synthetische rijen zijn beduidend dichter bij de trainingsdata. "
                    "Mogelijke memorisatie in het model. "
                    "Publiceer niet zonder nader onderzoek."
                ),
            }
            msg_fn = _RISK_MSG.get(priv.risk_level, st.info)
            msg_fn(_PRIV_TEXT.get(priv.risk_level, ""))
            st.caption(
                f"Gebaseerd op {priv.n_cols} numerieke kolom(men). "
                "DCR-ratio > 0.9 = laag · 0.7–0.9 = matig · < 0.7 = hoog. "
                "**Geen formele privacygarantie — "
                "een DPIA blijft vereist bij publicatie.**"
            )

    if primary_key is not None:
        st.info(
            f"ℹ️ Primary key '{primary_key}' is vervangen door nieuwe anonieme ID's "
            "en maakt geen deel uit van de privacyberekening."
        )

    with st.expander("Bivariate correlaties — verband tussen kolompar", expanded=False):
        if not pairs.available:
            st.info(f"Niet beschikbaar: {pairs.reason}")
        elif len(pairs.flagged) == 0:
            st.success("Alle kolomverbanden zijn goed bewaard (delta ≤ 0.1).")
        else:
            pairs_df = pd.DataFrame(pairs.flagged)
            st.dataframe(pairs_df, use_container_width=True)


# ── Distributies-tab ───────────────────────────────────────────────────────────
def _render_distributions(df: pd.DataFrame, synth: pd.DataFrame) -> None:
    cols = st.columns(2)
    for i, col_name in enumerate(df.columns):
        if col_name not in synth.columns:
            continue
        with cols[i % 2]:
            if pd.api.types.is_numeric_dtype(df[col_name]):
                plot_df = pd.concat(
                    [
                        df[[col_name]].assign(bron="echt"),
                        synth[[col_name]].assign(bron="synthetisch"),
                    ]
                )
                fig = px.histogram(
                    plot_df,
                    x=col_name,
                    color="bron",
                    barmode="overlay",
                    opacity=0.75,
                    title=col_name,
                    color_discrete_map={"echt": NPULS["blauw"], "synthetisch": NPULS["oranje"]},
                )
            else:
                counts = (
                    pd.DataFrame(
                        {
                            "echt": df[col_name].value_counts(normalize=True),
                            "synthetisch": synth[col_name].value_counts(normalize=True),
                        }
                    )
                    .fillna(0)
                    .reset_index()
                )
                counts.columns = [col_name, "echt", "synthetisch"]
                fig = px.bar(
                    counts,
                    x=col_name,
                    y=["echt", "synthetisch"],
                    barmode="group",
                    opacity=0.85,
                    title=col_name,
                    color_discrete_map={"echt": NPULS["blauw"], "synthetisch": NPULS["oranje"]},
                )
            st.plotly_chart(apply_plotly_style(fig), use_container_width=True)
            html = fig.to_html(include_plotlyjs="cdn", full_html=False)
            st.download_button(
                f"⬇ {col_name}.html",
                html.encode("utf-8"),
                file_name=f"{col_name}_distributie.html",
                mime="text/html",
                key=f"dl_chart_{col_name}",
            )


# ── Download-tab ───────────────────────────────────────────────────────────────
def _render_download(
    synth: pd.DataFrame,
    *,
    col_types: dict[str, str] | None,
    primary_key: str | None,
    modality: str | None,
    demo_name: str | None,
    n_generated: int,
    synthesizer: str,
    verdict: dict,
    recommendation: str,
    metadata_dict: dict | None,
    real_metadata_dict: dict | None,
) -> None:
    import sdv as _sdv

    sdv_version = _sdv.__version__

    csv_bytes = synth.to_csv(index=False).encode("utf-8")
    if st.button("Download synthetische data", use_container_width=True, type="primary"):
        _download_dialog(csv_bytes, verdict, recommendation)

    st.divider()

    with st.expander("Reproductie & Parameters", expanded=False):
        params: dict = {
            "synthesizer": synthesizer,
            "n_rows_or_sequences": n_generated,
            "modality": modality or "single_table",
            "sdv_version": sdv_version,
        }
        if col_types:
            params["columns"] = col_types
        if primary_key:
            params["primary_key"] = primary_key
        if demo_name:
            params["demo_dataset"] = demo_name
        intended = st.session_state.get("intended_use")
        if intended:
            params["intended_use"] = intended

        st.caption("Parameters van deze synthese — bewaar voor reproduceerbaarheid.")
        st.code(
            yaml.dump(params, allow_unicode=True, default_flow_style=False),
            language="yaml",
        )
        if real_metadata_dict or metadata_dict:
            import json

            with st.expander("SDV Metadata (JSON)", expanded=False):
                st.caption(
                    "Download de metadata als JSON om de synthese buiten ceda-synth te "
                    "reproduceren via `SingleTableMetadata.load_from_json()`."
                )
                mc1, mc2 = st.columns(2)

                with mc1:
                    st.markdown("**Originele data** — auto-gedetecteerd door SDV")
                    if real_metadata_dict:
                        real_json = json.dumps(real_metadata_dict, indent=2, ensure_ascii=False)
                        st.code(real_json, language="json")
                        st.download_button(
                            "Download metadata_origineel.json",
                            real_json.encode("utf-8"),
                            file_name="metadata_origineel.json",
                            mime="application/json",
                            key="dl_real_meta",
                        )
                    else:
                        st.info("Niet beschikbaar.")

                with mc2:
                    st.markdown("**Gebruikte synthese** — geconfigureerd in de app")
                    if metadata_dict:
                        synth_json = json.dumps(metadata_dict, indent=2, ensure_ascii=False)
                        st.code(synth_json, language="json")
                        st.download_button(
                            "Download metadata_synthese.json",
                            synth_json.encode("utf-8"),
                            file_name="metadata_synthese.json",
                            mime="application/json",
                            key="dl_synth_meta",
                        )
                    else:
                        st.info("Genereer eerst synthetische data.")

        with st.expander("Python-code voor automatisering", expanded=False):
            py_code = _build_code(
                col_types, primary_key, modality, demo_name, n_generated, sdv_version
            )
            st.code(py_code, language="python")
            requirements_txt = f"sdv=={sdv_version}\npandas>=2.0\npyyaml>=6.0\n"
            st.download_button(
                "Download requirements.txt",
                requirements_txt,
                "requirements.txt",
                "text/plain",
            )


def _build_code(
    col_types: dict[str, str] | None,
    primary_key: str | None,
    modality: str | None,
    demo_name: str | None,
    n_generated: int,
    sdv_version: str = "",
) -> str:
    version_comment = f"# sdv=={sdv_version}\n" if sdv_version else ""
    if modality == "sequential":
        return f"""\
{version_comment}import pandas as pd
from sdv.datasets.demo import download_demo
from sdv.sequential import PARSynthesizer

real, metadata = download_demo(modality="sequential", dataset_name="{demo_name}")

synthesizer = PARSynthesizer(metadata, epochs=20)
synthesizer.fit(real)
synth = synthesizer.sample(num_sequences={n_generated})
synth.to_csv("synthetisch.csv", index=False)
"""
    col_defs = "\n".join(
        f'metadata.add_column("{c}", sdtype="{t}")' for c, t in (col_types or {}).items()
    )
    pk_line = f'\nmetadata.set_primary_key("{primary_key}")' if primary_key else ""
    return f"""\
{version_comment}import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

real = pd.read_csv("jouw_data.csv")

metadata = SingleTableMetadata()
{col_defs}{pk_line}

synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(real)
synth = synthesizer.sample(num_rows={n_generated})
synth.to_csv("synthetisch.csv", index=False)
"""
