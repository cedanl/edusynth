# Claude Configuration — edu-synth

## Project Overview

edu-synth is een open-source CEDA/Npuls Streamlit-applicatie die [SDV (Synthetic Data Vault)](https://github.com/sdv-dev/SDV) toegankelijk maakt voor data-analisten bij Nederlandse hogeronderwijsinstellingen. Gebruikers uploaden een dataset, genereren een synthetische versie en bekijken statistische validatierapporten — zonder SDV zelf te hoeven configureren.

**Kernfunctionaliteit:**
- Databestand uploaden (CSV/Parquet) via de Streamlit-app
- Kolomtypes automatisch detecteren, handmatig bijstellen
- Synthetische data genereren via SDV GaussianCopulaSynthesizer
- Statistische validatie tonen (TV-afstand, Wasserstein) met distributieplots
- SDV-codeblok genereren zodat men het zelf kan reproduceren
- CLI voor batchverwerking (`synthesize`, `validate`)

**Doelgroep:** data-analisten en onderzoekers bij Nederlandse hogeronderwijsinstellingen.

## Tech Stack

- **Python 3.12.7** met **uv** voor dependency management
- **Streamlit ≥1.35** als app-framework (primaire interface)
- **SDV** als synthesizer engine (GaussianCopulaSynthesizer)
- **sdmetrics** voor validatiemetrieken
- **pandas / scipy** voor dataverwerking en statistiek
- **Plotly** voor distributieplots in de app
- **Rich** voor CLI-output
- **MkDocs + Material** voor documentatie

## Architectuurprincipes

### App is de primaire interface, CLI is een dun laagje
De app (`app.py`) is het hoofdproduct. `cli.py` biedt `edu-synth app` (lanceert Streamlit) en batchcommando's (`synthesize`, `validate`). Logica leeft in `synthesize.py` en `validate.py` — nooit in `cli.py` of `app.py`.

### Schema is optioneel
SDV detecteert kolomtypes automatisch via `detect_from_dataframe`. Een YAML-schema is alleen nodig voor batchverwerking via de CLI wanneer auto-detectie niet volstaat.

### Strikte dependency-richting
```
synthesize.py ← app.py
validate.py   ← app.py
synthesize.py ← cli.py
validate.py   ← cli.py
```
Nooit omhoog importeren. `synthesize.py` en `validate.py` weten niets van `app.py` of `cli.py`.

### Progressive disclosure — drie niveaus
Elke nieuwe UI-feature krijgt een niveau toegewezen. Niveau bepaalt de standaard zichtbaarheid:

| Niveau | Standaard | Voorbeelden |
|---|---|---|
| **1 — Altijd zichtbaar** | Direct op het scherm | Upload, aanbevolen config, scorecard-oordeel in gewone taal |
| **2 — Één klik** | Ingeklapte expander | Kolomtype-overrides, synthesizer-keuze, scoredetails |
| **3 — Twee kliks** | Geneste expander / aparte tab | YAML-parameters, Python-code, synthesizer-log |

**Regel:** voeg geen nieuwe feature toe aan niveau 1 tenzij de casual gebruiker hem bij elk gebruik nodig heeft. Twijfel je? Zet het op niveau 2. Power users vinden het; beginners worden niet overweldigd.

### Geen abstractie zonder tweede use case
Geen registries, factories of ABCs totdat er een tweede synthesizer-type is.

## Project Structure

```
src/edu_synth/
├── __init__.py       # publieke API: fit, sample, evaluate
├── app.py            # Streamlit-app — primaire interface
├── cli.py            # dun — app-launcher + batchcommando's
├── synthesize.py     # fit() en sample()
└── validate.py       # evaluate() → Report

tests/
├── fixtures/         # kleine hand-crafted CSV/YAML, WEL in git
└── edu_synth/       # spiegelt src/edu_synth/ exact
    ├── test_cli.py
    ├── test_synthesize.py
    └── test_validate.py

docs/                 # MkDocs bronbestanden
data/                 # gitignored — lokale ontwikkeldata
scripts/              # hulpscripts (download_datasets.py)
```

## Development Commands

```bash
# Installeren — app-runtime (dit is wat een gebruiker/tester nodig heeft)
uv sync

# Installeren — dev-omgeving (test/lint + docs + demo)
uv sync --all-groups

# App starten
uv run edu-synth app

# CLI (batch)
uv run edu-synth synthesize data.csv output.csv --rows 1000
uv run edu-synth validate data.csv output.csv

# Tests
uv run pytest

# Linting
uv run ruff check src tests
uv run ruff format src tests

# Datasets downloaden voor ontwikkeling
uv run scripts/download_datasets.py

# Benchmark synthesekwaliteit over vaste SDV-demo-datasets (reproduceerbare meetlat)
uv run scripts/benchmark.py

# Regressiecheck tegen het vastgelegde ijkpunt (exit 1 bij verslechtering)
uv run scripts/benchmark.py --check

# Ijkpunt her-ijken na een bewuste kwaliteitsverbetering
uv run scripts/benchmark.py --update-baseline

# Documentatie lokaal
uv run mkdocs serve
```

## Teststrategie

| Niveau | Bestand | Wat het bewaakt |
|---|---|---|
| Unit | `test_cli.py` | Argument-parsing, exitcodes, app-command |
| Unit | `test_synthesize.py` | Schema laden, metadata opbouwen, optionele schema_path |
| Unit | `test_validate.py` | Report, afstandsmetrieken |

Tests draaien zonder echte data en zonder Streamlit te starten — alleen `tests/fixtures/` wordt gebruikt. CI draait `pytest` + `ruff` bij elke push en PR naar `main`.

### Wanneer schrijf je een test?
- Nieuwe CLI-vlag → test in `test_cli.py`
- Nieuwe publieke functie in API → test in bijbehorend testbestand
- Nieuwe validatiemetriek → test in `test_validate.py`
- Streamlit-UI-logica → test de onderliggende functie, niet Streamlit zelf

## Documentatieregel (verplicht bij elke PR)

| Als je dit wijzigt… | …update dan |
|---|---|
| App-flow / `app.py` | `docs/aan-de-slag.md` |
| CLI-vlaggen / `cli.py` | `docs/aan-de-slag.md` |
| Schema-velden / `synthesize.py` | `docs/configuratie.md` |
| Validatiemetrieken / `validate.py` | `docs/validatie.md` |
| Synthesizer-aanpak | `docs/methodologie/index.md` |

## Validatiestandaard

Huidige implementatie: TV-afstand (categorisch) + Wasserstein (numeriek).
Nog te implementeren: DCR/NNDR privacyvalidatie via `sdmetrics`.

Voeg geen privacyclaims toe aan documentatie zolang DCR/NNDR ontbreekt.
