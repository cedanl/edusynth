# Claude Configuration — edusynth

## Project Overview

edusynth is een open-source CEDA/Npuls tool voor het genereren van privacyveilige synthetische versies van Nederlandse onderwijsdatasets. Het gebruikt SDV (Synthetic Data Vault) als statistische engine en biedt zowel een Python API als een CLI.

**Doelgroep:** data-analisten en onderzoekers bij Nederlandse hogeronderwijsinstellingen.

## Tech Stack

- **Python 3.12.7** met **uv** voor dependency management
- **SDV** als synthesizer engine (GaussianCopulaSynthesizer, later PAR/HMA)
- **sdmetrics** voor validatiemetrieken
- **pandas / scipy** voor dataverwerking en statistiek
- **Rich** voor CLI-output
- **MkDocs + Material** voor documentatie

## Architectuurprincipes

### API-first, CLI is een dun laagje
Alle logica leeft in `synthesize.py` en `validate.py`. `cli.py` parseert argumenten en delegeert — nooit zelf logica toevoegen aan `cli.py`.

### Schema is de enige bron van waarheid
Geen hardcoded kolomnamen, kansen, of drempelwaarden in Python-code. Alles wat datasetspecifiek is, staat in een YAML-schema onder `schemas/` (productie) of `tests/fixtures/` (tests).

### Strikte dependency-richting
```
schema (YAML) → synthesize.py → cli.py
schema (YAML) → validate.py  → cli.py
```
Nooit omhoog importeren. `synthesize.py` weet niets van `cli.py`.

### Geen abstractie zonder tweede use case
Geen registries, factories of ABCs totdat er daadwerkelijk een tweede synthesizer-type is. Drie vergelijkbare implementaties vóór een abstractie.

## Project Structure

```
src/edusynth/
├── __init__.py       # publieke API: fit, sample, evaluate
├── cli.py            # dun — alleen argparse + delegeren
├── synthesize.py     # fit() en sample()
└── validate.py       # evaluate() → Report

tests/
├── fixtures/         # kleine hand-crafted CSV/YAML, WEL in git
└── edusynth/         # spiegelt src/edusynth/ exact
    ├── test_cli.py
    ├── test_synthesize.py
    └── test_validate.py

docs/                 # MkDocs bronbestanden
data/                 # gitignored — lokale ontwikkeldata
scripts/              # hulpscripts (download_datasets.py)
```

## Development Commands

```bash
# Installeren
uv sync --all-extras

# Draaien
uv run edusynth --help

# Tests
uv run pytest

# Linting
uv run ruff check src tests
uv run ruff format src tests

# Datasets downloaden voor ontwikkeling
uv run scripts/download_datasets.py

# Documentatie lokaal
uv run mkdocs serve
```

## Teststrategie

| Niveau | Bestand | Wat het bewaakt |
|---|---|---|
| Unit | `test_cli.py` | Argument-parsing, exitcodes |
| Unit | `test_synthesize.py` | Schema laden, metadata opbouwen |
| Unit | `test_validate.py` | Report, afstandsmetrieken |

Tests draaien zonder echte data — alleen `tests/fixtures/` wordt gebruikt. CI draait `pytest` + `ruff` bij elke push en PR naar `main`.

### Wanneer schrijf je een test?
- Nieuwe CLI-vlag → test in `test_cli.py`
- Nieuwe publieke functie in API → test in bijbehorend testbestand
- Nieuwe validatiemetriek → test in `test_validate.py`

## Documentatieregel (verplicht bij elke PR)

| Als je dit wijzigt… | …update dan |
|---|---|
| CLI-vlaggen / `cli.py` | `docs/aan-de-slag.md` |
| Schema-velden / `synthesize.py` | `docs/configuratie.md` |
| Validatiemetrieken / `validate.py` | `docs/validatie.md` |
| Synthesizer-aanpak | `docs/methodologie/index.md` |

## Validatiestandaard

Huidige implementatie: TV-afstand (categorisch) + Wasserstein (numeriek).
Nog te implementeren: DCR/NNDR privacyvalidatie via `sdmetrics`.

Voeg geen privacyclaims toe aan documentatie zolang DCR/NNDR ontbreekt.
