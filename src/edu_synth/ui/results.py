"""Resultaten UI — tabs voor validatie, distributies en download+reproductie."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

from edu_synth.core.validate import (
    RECOMMENDATION_DISCLAIMER,
    PairsReport,
    PrivacyReport,
    Report,
    SDMetricsReport,
    SequentialReport,
    build_validation_report,
    correlation_risk,
    evaluate,
    evaluate_pairs,
    evaluate_privacy,
    evaluate_sdmetrics,
    evaluate_sequential,
    improvement_advice,
    score_verdict,
    sequential_recommendation,
    sequential_verdict,
    usage_recommendation,
    worst_sequential_component,
)
from edu_synth.ui.theme import NPULS, apply_plotly_style

# ── Verdict-helpers ────────────────────────────────────────────────────────────
_RISK_ICON = {"laag": "✅", "matig": "⚠️", "hoog": "❌", "onbekend": "○"}
_RISK_MSG = {"laag": st.success, "matig": st.warning, "hoog": st.error}


def _verdeling_verdict(report: Report) -> tuple[str, str]:
    # Numeriek (genorm. Wasserstein) en categorisch (TV) staan na normalisatie op
    # dezelfde schaal en tellen even zwaar mee via het `score`-veld.
    scored = [r for r in report.rows if "score" in r]
    if not scored:
        return "Geen kolommen", "onbekend"
    max_score = max(r["score"] for r in scored)
    n_failed = sum(1 for r in scored if not r.get("ok", True))
    return score_verdict(max_score, n_failed, len(scored))


def _privacy_verdict(priv) -> tuple[str, str]:
    if not priv.available:
        return "Niet berekend", "onbekend"
    labels = {"laag": "Laag risico", "matig": "Matig risico", "hoog": "Hoog risico"}
    return labels.get(priv.risk_level, "Onbekend"), priv.risk_level


def _samenhang_verdict(pairs: PairsReport, risk: str) -> tuple[str, str]:
    """Scorecard-label voor correlatiebehoud.

    *risk* is de uitkomst van :func:`correlation_risk`. Die geeft 'laag' wanneer
    correlatie niet berekenbaar is (< 2 numerieke kolommen), zodat het
    bruikbaarheidsoordeel niet verlaagt — maar op de scorecard tonen we dat
    geval expliciet als 'Niet berekend' / 'onbekend'.
    """
    if not pairs.available:
        return "Niet berekend", "onbekend"
    # Kort gehouden zodat de scorecard ook op smalle schermen niet afkapt; de
    # volledige uitleg staat in de sectie "Samenhang tussen kolommen".
    labels = {"laag": "Goed", "matig": "Afwijkingen", "hoog": "Omgeklapt"}
    return labels[risk], risk


def _bruikbaarheid_verdict(
    verd_risk: str, priv_risk: str, corr_risk: str, temp_risk: str | None = None
) -> tuple[str, str]:
    """Combineer de deeloordelen tot één bruikbaarheidsoordeel.

    Geen solo-veto: één enkele `hoog`-dimensie verlaagt naar "Bruikbaar met
    voorbehoud", niet meteen naar "Niet aanbevolen" — anders zou bv. één
    omgeklapt verband een verder uitstekende dataset afkeuren. "Niet aanbevolen"
    alleen bij privacy-risico (altijd zwaarwegend) of bij ≥2 dimensies hoog.

    *temp_risk* (tijdsgedrag) telt alleen mee bij longitudinale data; is ``None``
    voor tabulaire synthese en verandert het tabular-oordeel dan niet.
    """
    risks = [verd_risk, priv_risk, corr_risk]
    non_priv = [verd_risk, corr_risk]
    if temp_risk is not None:
        risks.append(temp_risk)
        non_priv.append(temp_risk)
    # Privacy telt niet mee in de ≥2-telling: een hoog privacyrisico is hierboven
    # al los diskwalificerend. non_priv weegt alleen de overige dimensies.
    if priv_risk == "hoog" or non_priv.count("hoog") >= 2:
        return "Niet aanbevolen", "hoog"
    if "hoog" in risks or "matig" in risks or "onbekend" in risks:
        return "Bruikbaar met voorbehoud", "matig"
    return "Hoge bruikbaarheid", "laag"


def _sequential_report(
    df: pd.DataFrame,
    synth: pd.DataFrame,
    modality: str | None,
    metadata_dict: dict | None = None,
) -> SequentialReport | None:
    """Bereken de temporele validatie voor longitudinale data, anders None.

    De sequence-key en -index komen uit de SDV-metadata (``tables.data``), zodat
    dit werkt voor zowel een eigen upload als een demo-dataset. Bij tabulaire
    synthese of ontbrekende sequence-velden gebeurt er niets.
    """
    if modality != "sequential":
        return None
    table_meta = (metadata_dict or {}).get("tables", {}).get("data", {})
    seq_key = table_meta.get("sequence_key")
    seq_index = table_meta.get("sequence_index")
    if not seq_key or not seq_index:
        return None
    return evaluate_sequential(df, synth, seq_key, seq_index, table_meta)


def _scorecard(col, label: str, verdict: str, risk: str) -> None:
    col.metric(label, f"{_RISK_ICON.get(risk, '○')} {verdict}")


_VERDICT_TEXT = {
    "laag": (
        "Deze synthetische dataset vertoont een **hoge statistische gelijkenis** met de echte data."
    ),
    "matig": (
        "Deze synthetische dataset is **bruikbaar met voorbehoud**. "
        "Controleer de details in het validatierapport voor gebruik."
    ),
    "hoog": (
        "Deze synthetische dataset wijkt **sterk af van de echte data**. "
        "Bekijk de details en pas de syntheseinstellingen aan."
    ),
}


def _render_verdict_banner(verdict: dict, recommendation: str) -> None:
    """Toon het overall bruikbaarheidsoordeel als gekleurde banner vóór de tabs.

    Staat boven de tabs zodat ook wie direct naar Download gaat het oordeel ziet.
    """
    brk_risk = verdict["brk_risk"]
    icon = _RISK_ICON.get(brk_risk, "○")
    msg_fn = _RISK_MSG.get(brk_risk, st.info)
    msg_fn(f"**{icon} Oordeel: {verdict['brk_label']}** — {_VERDICT_TEXT.get(brk_risk, '')}")
    st.caption(f"**Gebruik:** {recommendation}")
    st.caption(f"ℹ️ {RECOMMENDATION_DISCLAIMER}")


def _render_improvement_advice(advice: list[str]) -> None:
    """Toon concrete verbeterpunten onder het oordeel (alleen bij matig/onvoldoende)."""
    if not advice:
        return
    with st.container(border=True):
        st.markdown("**Wat kun je verbeteren?**")
        for tip in advice:
            st.markdown(f"- {tip}")


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
    st.caption(f"ℹ️ {RECOMMENDATION_DISCLAIMER}")
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
    numerical_distributions: dict[str, str] | None = None,
    primary_key: str | None,
    modality: str | None,
    demo_name: str | None,
    n_generated: int,
    random_seed: int | None = None,
    seq_info: dict | None = None,
    metadata_dict: dict | None = None,
    real_metadata_dict: dict | None = None,
    synthesizer: str | None = None,
) -> None:
    # Naam van de daadwerkelijk gebruikte synthesizer, voor het rapport en de code.
    # Valt terug op de modaliteit-default als de app 'm (nog) niet doorgeeft.
    synth_name = synthesizer or ("sequential_copula" if modality == "sequential" else "gaussian")
    with st.spinner("Validatie en privacyanalyse berekenen…"):
        report = evaluate(df, synth, metadata_dict or real_metadata_dict)
        priv = evaluate_privacy(df, synth, primary_key=primary_key)
        pairs = evaluate_pairs(df, synth)
        sdm = evaluate_sdmetrics(df, synth, metadata_dict)
        seq = _sequential_report(df, synth, modality, metadata_dict)

    verd_label, verd_risk = _verdeling_verdict(report)
    priv_label, priv_risk = _privacy_verdict(priv)
    corr_risk = correlation_risk(pairs)
    corr_label, corr_disp_risk = _samenhang_verdict(pairs, corr_risk)
    temp_label, temp_risk = sequential_verdict(seq) if seq is not None else (None, None)
    brk_label, brk_risk = _bruikbaarheid_verdict(verd_risk, priv_risk, corr_risk, temp_risk)
    recommendation = usage_recommendation(report, priv, pairs)

    verdict = {
        "verd_label": verd_label,
        "verd_risk": verd_risk,
        "priv_label": priv_label,
        "priv_risk": priv_risk,
        "corr_label": corr_label,
        "corr_risk": corr_disp_risk,
        "temp_label": temp_label,
        "temp_risk": temp_risk,
        "brk_label": brk_label,
        "brk_risk": brk_risk,
    }

    _render_verdict_banner(verdict, recommendation)
    if seq is not None:
        st.caption(f"🕒 **Tijdsgedrag:** {sequential_recommendation(seq)}")

    if verdict["brk_risk"] in ("matig", "hoog"):
        _render_improvement_advice(
            improvement_advice(report, df, priv, numerical_distributions, pairs)
        )

    tab_val, tab_dist, tab_dl = st.tabs(
        ["Validatierapport", "Distributies", "Download & Reproductie"]
    )
    with tab_val:
        _render_validation(report, priv, verdict, pairs, primary_key, sdm, seq)
    with tab_dist:
        _render_distributions(df, synth, report)
    with tab_dl:
        _render_download(
            synth,
            col_types=col_types,
            numerical_distributions=numerical_distributions,
            primary_key=primary_key,
            modality=modality,
            demo_name=demo_name,
            n_generated=n_generated,
            random_seed=random_seed,
            seq_info=seq_info,
            verdict=verdict,
            recommendation=recommendation,
            metadata_dict=metadata_dict,
            real_metadata_dict=real_metadata_dict,
            report=report,
            priv=priv,
            sdm=sdm,
            seq=seq,
            n_training_rows=len(df),
            synthesizer=synth_name,
        )


# ── Validatie-tab ──────────────────────────────────────────────────────────────
def _render_validation(
    report: Report,
    priv,
    verdict: dict,
    pairs: PairsReport,
    primary_key: str | None,
    sdm: SDMetricsReport,
    seq: SequentialReport | None = None,
) -> None:
    # Het overall bruikbaarheidsoordeel staat al prominent in de banner vóór de
    # tabs; hier tonen we de drie deeloordelen die daarin samenkomen. Drie kaarten
    # i.p.v. vier houdt de labels ook op smalle schermen (14") leesbaar. Bij
    # longitudinale data vervangt Tijdsgedrag de Samenhang-kaart: het temporele
    # oordeel is daar de kerndimensie, correlatie blijft in het detail zichtbaar.
    c1, c2, c3 = st.columns(3)
    _scorecard(c1, "Verdeling", verdict["verd_label"], verdict["verd_risk"])
    if seq is not None:
        _scorecard(c2, "Tijdsgedrag", verdict["temp_label"], verdict["temp_risk"])
    else:
        _scorecard(c2, "Samenhang", verdict["corr_label"], verdict["corr_risk"])
    _scorecard(c3, "Privacy", verdict["priv_label"], verdict["priv_risk"])

    st.divider()

    if seq is not None:
        _render_sequential_detail(seq)
        st.divider()

    _render_correlations(pairs)
    st.divider()

    with st.expander("Statistisch detail — verdeling per kolom", expanded=False):
        rdf = report.to_dataframe().copy()
        # Toon modal warnings per rij, dan de kolom verwijderen
        if "modal_warning" in rdf.columns:
            for _, row in rdf[rdf["modal_warning"].notna()].iterrows():
                st.warning(f"⚠️ **{row['column']}**: bimodaliteit — {row['modal_warning']}")
            rdf = rdf.drop(columns=["modal_warning"])
        if "ok" in rdf.columns:
            rdf["ok"] = rdf["ok"].map({True: "✓", False: "✗"})
        # Score (vergelijkbaar) als primaire kolom, ruwe afstand ernaast
        rdf = rdf.rename(
            columns={
                "column": "Kolom",
                "dtype": "Type",
                "score": "Score (genorm.)",
                "distance": "Ruwe afstand",
                "metric": "Metric",
                "ok": "OK",
            }
        )
        order = [
            c
            for c in ["Kolom", "Type", "Score (genorm.)", "Ruwe afstand", "Metric", "OK"]
            if c in rdf.columns
        ]
        st.dataframe(rdf[order], use_container_width=True)
        st.caption(
            "**TV-afstand** (categorisch) = hoeveel de categoriefrequenties verschillen; "
            "**Wasserstein** (numeriek) = hoeveel de numerieke verdelingen verschillen, "
            "genormaliseerd op de IQR van de echte kolom zodat alle kolommen vergelijkbaar zijn. "
            "Beide: lager = beter. Score < 0.1 = uitstekend · < 0.2 = goed · > 0.2 = let op. "
            "**Deze metrieken meten statistische gelijkenis, niet privacy.**"
        )

    with st.expander("Privacydetail — DCR / NNDR", expanded=False):
        if not priv.available:
            st.info(f"Privacyanalyse niet beschikbaar: {priv.reason}")
        else:
            _DCR_HELP = (
                "Distance to Closest Record: hoe ver een synthetische rij van de "
                "dichtstbijzijnde échte rij ligt. Hoger = verder weg = veiliger."
            )
            p1, p2, p3 = st.columns(3)
            p1.metric(
                "DCR synth (mediaan)",
                f"{priv.dcr_synth_median:.4f}",
                help=f"{_DCR_HELP} Dit is de mediaan over de synthetische rijen.",
            )
            p2.metric(
                "DCR holdout (mediaan)",
                f"{priv.dcr_holdout_median:.4f}",
                help=f"{_DCR_HELP} Referentie: een echte holdout-set die het model nooit zag.",
            )
            p3.metric(
                "DCR-ratio",
                f"{priv.dcr_ratio:.2f}",
                help="DCR synth ÷ DCR holdout. ~1 = synthetische data ligt even ver als "
                "een echte holdout (veilig); hoger = beter. Grens: > 0.9 laag · 0.7–0.9 "
                "matig · < 0.7 hoog risico.",
            )
            st.metric(
                "NNDR mediaan",
                f"{priv.nndr_median:.2f}",
                help="Nearest Neighbor Distance Ratio: afstand tot de dichtstbijzijnde "
                "échte rij ÷ tot de op-één-na dichtstbijzijnde. Hoger = beter; lage waarden "
                "betekenen dat een synthetische rij vlak op één echte rij zit.",
            )
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
            if priv.excluded_cols:
                cols = ", ".join(f"**{c}**" for c in priv.excluded_cols)
                st.warning(
                    f"⚠️ Deze kolommen zijn buiten de privacyberekening gehouden "
                    f"(te veel unieke waarden — vrije tekst of identifier): {cols}. "
                    "Beoordeel ze handmatig op re-identificatierisico."
                )
            st.caption(
                f"Gebaseerd op {priv.n_cols} kolom(men) "
                f"({priv.n_numeric_cols} numeriek, {priv.n_categorical_cols} categorisch). "
                "DCR-ratio > 0.9 = laag · 0.7–0.9 = matig · < 0.7 = hoog. "
                "**Geen formele privacygarantie — "
                "een DPIA blijft vereist bij publicatie.**"
            )

    if primary_key is not None:
        st.info(
            f"ℹ️ Primary key '{primary_key}' is vervangen door nieuwe anonieme ID's "
            "en maakt geen deel uit van de privacyberekening."
        )

    _render_sdmetrics(sdm)


def _render_correlations(pairs: PairsReport) -> None:
    """Toon het correlatiebehoud prominent — niet weggeklikt in een expander.

    Een omgeklapt verband is voor wie op samenhang analyseert het gevaarlijkste
    signaal, dus het staat direct onder de scorecards.
    """
    st.markdown("**Samenhang tussen kolommen** (bivariate correlaties)")
    if not pairs.available:
        st.info(f"Niet beschikbaar: {pairs.reason}")
    elif len(pairs.flagged) == 0:
        st.success("Alle kolomverbanden zijn goed bewaard (delta ≤ 0.1).")
    else:
        st.warning(
            f"{len(pairs.flagged)} verband(en) wijken af. Een tekenomslag betekent dat een "
            "positief verband negatief is geworden (of omgekeerd) — controleer dit vóór "
            "analyses die op samenhang tussen kolommen leunen."
        )
        st.dataframe(pd.DataFrame(pairs.flagged), use_container_width=True)
    st.caption(
        "Delta = het verschil in correlatie tussen twee kolommen, echt vs. synthetisch. "
        "Lager = beter; ≤ 0.1 = goed bewaard. Een tekenomslag (+ wordt −) weegt het zwaarst."
    )


_SEQ_KIND_NL = {"transition": "overgangsmatrix", "autocorrelation": "autocorrelatie"}


def _render_sequential_detail(seq: SequentialReport) -> None:
    """Toon het temporele detail: sequentielengte + per-kolom overgangs-/autocorrelatiescore.

    Staat direct onder de scorecards omdat het tijdsgedrag voor longitudinale data
    de kerndimensie is — net zoals correlaties dat zijn voor tabulaire analyse.
    """
    st.markdown("**Tijdsgedrag** (longitudinaal — overgangen, trends, trajectlengte)")
    if not seq.available:
        st.info(f"Niet beschikbaar: {seq.reason}")
        return

    icon = "✅" if seq.length_ok else "⚠️"
    st.metric(
        "Sequentielengte-afstand",
        f"{icon} {seq.length_distance:.3f}",
        help="Hoeveel de lengtes van de trajecten (aantal tijdstappen per entiteit) "
        "afwijken van de echte data. Lager = beter; grens 0.2.",
    )

    # Bij een afwijking: benoem expliciet wélke kolom + metriek het oordeel drijft.
    # De bovenste scorekaart aggregeert alle componenten; zonder deze regel lijkt een
    # groene lengte-score tegenstrijdig met een rood totaaloordeel.
    driver = worst_sequential_component(seq)
    if driver is not None:
        kind_nl = {
            "transition": "overgangsmatrix",
            "autocorrelation": "autocorrelatie",
            "length": "sequentielengte",
        }.get(driver["kind"], driver["kind"])
        where = f"kolom `{driver['column']}`" if driver["column"] else "de trajectlengtes"
        st.warning(
            f"⚠️ Grootste afwijking: {where} ({kind_nl}) — score {driver['score']:.2f}, "
            f"boven de grens {driver['threshold']:.1f}. Dit drijft het tijdsgedrag-oordeel."
        )

    if seq.rows:
        rdf = pd.DataFrame(seq.rows)[["column", "kind", "score", "ok"]].copy()
        rdf["kind"] = rdf["kind"].map(_SEQ_KIND_NL).fillna(rdf["kind"])
        rdf["ok"] = rdf["ok"].map({True: "✓", False: "✗"})
        rdf = rdf.rename(
            columns={
                "column": "Kolom",
                "kind": "Metriek",
                "score": "Score (genorm.)",
                "ok": "OK",
            }
        )
        st.dataframe(rdf, use_container_width=True)

    st.caption(
        "**Overgangsmatrix** (categorisch) = hoeveel de doorstroomkansen tussen statussen "
        "afwijken (0 = identiek, 1 = compleet anders). **Autocorrelatie** (numeriek) = hoeveel "
        "de samenhang tussen opeenvolgende tijdstappen (trend) afwijkt. **Sequentielengte** = "
        "hoeveel de trajectlengtes afwijken. Alle drie: lager = beter, grens 0.2."
    )


def _render_sdmetrics(sdm: SDMetricsReport) -> None:
    """Geavanceerde sdmetrics QualityReport (niveau 3) — uitgebreide kwaliteitsmetrieken."""
    with st.expander("Geavanceerde kwaliteitsscore (sdmetrics)", expanded=False):
        if not sdm.available:
            st.info(f"Niet beschikbaar: {sdm.reason}")
            return

        if sdm.overall_score is not None:
            st.metric(
                "Overall quality score",
                f"{sdm.overall_score:.1%}",
                help="Similarity-score: hoger = beter, 100% = perfect. Let op: tegengesteld "
                "aan de afstands-metrieken elders (TV, Wasserstein, DCR), waar lager beter is.",
            )
        st.caption(
            "Aanvullende kwaliteitsmetrieken uit sdmetrics — dit zijn *similarity*-scores "
            "(hoger = beter, 100% = perfect), tegengesteld aan de afstanden hierboven. "
            "**Column Shapes** meet de verdeling per kolom (TVComplement / KSComplement); "
            "**Column Pair Trends** meet samenhang tussen kolomparen, inclusief "
            "categorisch × categorisch (ContingencySimilarity)."
        )

        if sdm.column_shapes:
            st.markdown("**Column Shapes**")
            st.dataframe(pd.DataFrame(sdm.column_shapes), use_container_width=True)
        if sdm.column_pair_trends:
            st.markdown("**Column Pair Trends**")
            st.dataframe(pd.DataFrame(sdm.column_pair_trends), use_container_width=True)
            st.caption(
                "Kolomparen met een zwakke samenhang in de echte data krijgen score "
                "*NaN* en blijven buiten het oordeel — dat is geen fout."
            )


# ── Distributies-tab ───────────────────────────────────────────────────────────
def _rank_columns_by_deviation(columns: list[str], report: Report) -> list[str]:
    """Sorteer kolommen op aflopende afwijking — hoogste score (meest afwijkend) eerst.

    Kolommen zonder score belanden achteraan; gelijke scores behouden hun volgorde.
    """
    score_by_col = {r["column"]: r.get("score", 0.0) for r in report.rows if "column" in r}
    return sorted(columns, key=lambda c: score_by_col.get(c, 0.0), reverse=True)


def _render_distributions(df: pd.DataFrame, synth: pd.DataFrame, report: Report) -> None:
    shared = [c for c in df.columns if c in synth.columns]
    if not shared:
        st.info("Geen gedeelde kolommen om te visualiseren.")
        return

    # Standaard de 8 meest afwijkende kolommen tonen — bij brede datasets (50+
    # kolommen) voorkomt dit een eindeloze scroll en tientallen zware Plotly-figuren.
    default_cols = _rank_columns_by_deviation(shared, report)[:8]
    selected = st.multiselect(
        "Kolommen om te visualiseren (standaard: 8 meest afwijkende)",
        options=shared,
        default=default_cols,
    )
    if not selected:
        st.info("Selecteer minimaal één kolom om de verdeling te tonen.")
        return

    cols = st.columns(2)
    for i, col_name in enumerate(selected):
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
                # Categorieën als string tellen: echt en synthetisch kunnen na
                # synthese verschillende waarde-types hebben (bv. int vs str), en
                # dan crasht het samenvoegen van de twee value_counts-indexen op het
                # sorteren ('<' not supported between int and str). String-index is
                # altijd sorteerbaar en voor een verdelingsstaafje prima.
                counts = (
                    pd.DataFrame(
                        {
                            "echt": df[col_name].astype(str).value_counts(normalize=True),
                            "synthetisch": synth[col_name].astype(str).value_counts(normalize=True),
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
_SDTYPE_SUMMARY_NL = {
    "categorical": "categorisch",
    "numerical": "numeriek",
    "datetime": "datum",
    "id": "ID",
    "boolean": "boolean",
}


def _summarize_metadata(metadata_dict: dict | None) -> str | None:
    """Vat een SDV-metadata-dict samen in gewone taal (kolomtypes + privacyvlaggen).

    Retourneert None als er geen kolommen zijn, zodat de aanroeper niets toont.
    """
    columns = (metadata_dict or {}).get("columns") or {}
    if not columns:
        return None

    counts: dict[str, int] = {}
    pii = 0
    for spec in columns.values():
        sdtype = spec.get("sdtype", "onbekend")
        counts[sdtype] = counts.get(sdtype, 0) + 1
        if spec.get("pii"):
            pii += 1

    parts = [f"{n} {_SDTYPE_SUMMARY_NL.get(t, t)}" for t, n in counts.items()]
    return (
        f"Kolomtypes herkend: {', '.join(parts)}. {pii} kolom(men) gemarkeerd als privacygevoelig."
    )


def _render_download(
    synth: pd.DataFrame,
    *,
    col_types: dict[str, str] | None,
    numerical_distributions: dict[str, str] | None = None,
    primary_key: str | None,
    modality: str | None,
    demo_name: str | None,
    n_generated: int,
    random_seed: int | None,
    seq_info: dict | None,
    verdict: dict,
    recommendation: str,
    metadata_dict: dict | None,
    real_metadata_dict: dict | None,
    report: Report,
    priv: PrivacyReport,
    sdm: SDMetricsReport,
    seq: SequentialReport | None,
    n_training_rows: int,
    synthesizer: str = "gaussian",
) -> None:
    import json
    from datetime import date

    import sdv as _sdv

    sdv_version = _sdv.__version__

    if verdict["brk_risk"] == "hoog":
        st.error(
            "❌ **Niet aanbevolen om te downloaden.** Deze dataset scoort hoog risico "
            "op verdeling of privacy — controleer het validatierapport vóór gebruik."
        )

    csv_bytes = synth.to_csv(index=False).encode("utf-8")
    if st.button("Download synthetische data", use_container_width=True, type="primary"):
        _download_dialog(csv_bytes, verdict, recommendation)

    validation_report = build_validation_report(
        report=report,
        priv=priv,
        sdm=sdm,
        recommendation=recommendation,
        synthesizer=synthesizer,
        n_training_rows=n_training_rows,
        n_generated_rows=n_generated,
        sdv_version=sdv_version,
        generated_at=date.today().isoformat(),
        random_seed=random_seed,
        intended_use=st.session_state.get("intended_use"),
        seq=seq,
    )
    st.download_button(
        "Download validation_report.json",
        json.dumps(validation_report, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="validation_report.json",
        mime="application/json",
        use_container_width=True,
    )
    st.caption("Bevat alle scores en synthese-parameters. Bewaar het naast de CSV.")

    meta_summary = _summarize_metadata(metadata_dict or real_metadata_dict)
    if meta_summary:
        st.caption(meta_summary)

    st.divider()

    with st.expander("Reproductie & Parameters", expanded=False):
        params: dict = {
            "synthesizer": synthesizer,
            "n_rows_or_sequences": n_generated,
            "modality": modality or "single_table",
            "sdv_version": sdv_version,
        }
        if random_seed is not None:
            params["random_seed"] = random_seed
        if col_types:
            params["columns"] = col_types
        if numerical_distributions:
            params["numerical_distributions"] = numerical_distributions
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
            with st.expander("Technische details — SDV Metadata (JSON)", expanded=False):
                st.caption(
                    "Voor ontwikkelaars. Download de metadata als JSON om de synthese "
                    "buiten edu-synth te reproduceren via "
                    "`SingleTableMetadata.load_from_json()`."
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
                col_types,
                primary_key,
                modality,
                demo_name,
                n_generated,
                sdv_version,
                random_seed,
                seq_info,
                numerical_distributions,
                synthesizer,
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
    random_seed: int | None = None,
    seq_info: dict | None = None,
    numerical_distributions: dict[str, str] | None = None,
    synthesizer: str = "gaussian",
) -> str:
    version_comment = f"# sdv=={sdv_version}\n" if sdv_version else ""
    # np.random.seed() vóór fit() maakt de output reproduceerbaar (zie set_seed).
    seed_import = "import numpy as np\n" if random_seed is not None else ""
    seed_line = f"\nnp.random.seed({random_seed})" if random_seed is not None else ""
    if modality == "sequential":
        # Voor een eigen upload kies je zelf sequence key/index; bij demo-data komen
        # die uit de gedetecteerde kolommen.
        key = seq_info["key"] if seq_info else "student_id"
        idx = seq_info["index"] if seq_info else "jaar"
        load = (
            'real = pd.read_csv("jouw_data.csv")'
            if seq_info
            else "from sdv.datasets.demo import download_demo\n\n"
            f'real, _ = download_demo(modality="sequential", dataset_name="{demo_name}")'
        )
        seed_arg = f", seed={random_seed}" if random_seed is not None else ""
        if synthesizer == "par":
            # PAR (deep learning) van SDV — de zwaardere, optionele keuze.
            seed_setup = (
                f"import numpy as np\n\nnp.random.seed({random_seed})\n"
                if random_seed is not None
                else ""
            )
            return f"""\
{version_comment}import pandas as pd
from edu_synth.core.synthesize import build_sequential_metadata
from sdv.sequential import PARSynthesizer

{load}

{seed_setup}metadata = build_sequential_metadata(real, seq_key="{key}", seq_index="{idx}")
synthesizer = PARSynthesizer(metadata, epochs=128, verbose=True)
synthesizer.fit(real)
synth = synthesizer.sample(num_sequences={n_generated})
synth.to_csv("synthetisch.csv", index=False)
"""
        # Lichte, niet-neurale sequentiële synthesizer van edu-synth (wide-reshape +
        # GaussianCopula).
        return f"""\
{version_comment}import pandas as pd
from edu_synth.core.synthesize import fit_sequential, sample_sequential

{load}

model = fit_sequential(real, seq_key="{key}", seq_index="{idx}"{seed_arg})
synth = sample_sequential(model, n_sequences={n_generated})
synth.to_csv("synthetisch.csv", index=False)
"""
    col_defs = "\n".join(
        f'metadata.add_column("{c}", sdtype="{t}")' for c, t in (col_types or {}).items()
    )
    pk_line = f'\nmetadata.set_primary_key("{primary_key}")' if primary_key else ""
    # gaussian_kde e.d. op scheve kolommen — anders valt SDV terug op 'norm'.
    dist_arg = (
        f", numerical_distributions={numerical_distributions!r}" if numerical_distributions else ""
    )
    return f"""\
{version_comment}{seed_import}import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

real = pd.read_csv("jouw_data.csv")

metadata = SingleTableMetadata()
{col_defs}{pk_line}
{seed_line}
synthesizer = GaussianCopulaSynthesizer(metadata{dist_arg})
synthesizer.fit(real)
synth = synthesizer.sample(num_rows={n_generated})
synth.to_csv("synthetisch.csv", index=False)
"""
