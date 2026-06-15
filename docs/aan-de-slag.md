# Aan de slag

## Installatie

```bash
pip install ceda-synth
```

Of lokaal vanuit de broncode:

```bash
git clone https://github.com/cedanl/ceda-synth
cd ceda-synth
uv sync
uv run ceda-synth app
```

De app opent automatisch in je browser. Gebeurt dat niet, open dan de URL die in de terminal verschijnt.

## De app gebruiken

### Stap 1 — Data uploaden

Start de app en sleep een CSV- of Parquet-bestand op het uploadveld. ceda-synth toont een preview van de eerste rijen en detecteert automatisch kolomtypes.

!!! tip "Geschikte data"
    De app werkt het best met tabellen van minimaal 500 rijen. Kleinere datasets leveren minder stabiele correlatieramingen op.

### Stap 2 — Kolommen controleren

Na het uploaden zie je per kolom het gedetecteerde SDV-type (`categorical`, `numerical`, `datetime`, `id`). Pas dit aan als de detectie afwijkt.

### Stap 3 — Synthetische data genereren

Kies het aantal gewenste rijen en klik **Genereer**. Het model traint op de achtergrond — bij grotere datasets kan dit enkele seconden duren.

Onder **Geavanceerd — reproduceerbaarheid** stel je een **random seed** in (standaard `42`). Dezelfde seed met dezelfde data levert identieke synthetische output, zodat een collega of reviewer jouw resultaat exact kan reproduceren. De seed komt terug in de geëxporteerde parameters en in het gegenereerde Python-codeblok.

### Stap 4 — Validatierapport bekijken

De app toont per kolom:

- **TV-afstand** (categorisch) — hoe goed kloppen de verhoudingen? Onder `0.2` is groen.
- **Wasserstein-afstand** (numeriek) — hoe dicht liggen de verdelingen bij elkaar?
- Distributieplots echte vs. synthetische data

### Stap 5 — Exporteren

Download de synthetische data als CSV. Daarnaast kun je het **validatierapport** als `validation_report.json` downloaden: dit bundelt alle scores (per-kolom afstanden, sdmetrics, DCR/NNDR) en de synthese-parameters (synthesizer, aantal trainingsrijen, seed, SDV-versie) zodat het oordeel later reproduceerbaar vastligt. De app toont ook het bijbehorende SDV-codeblok om hetzelfde resultaat zelf te reproduceren.

## CLI (geavanceerd)

Voor batchverwerking is er ook een CLI:

```bash
ceda-synth synthesize data.csv output.csv --rows 1000 --seed 42
ceda-synth validate data.csv output.csv
```
