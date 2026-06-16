# Aan de slag

## Installatie

```bash
pip install edu-synth
```

Of lokaal vanuit de broncode:

```bash
git clone https://github.com/cedanl/edusynth
cd edu-synth
uv sync
uv run edu-synth app
```

De app opent automatisch in je browser. Gebeurt dat niet, open dan de URL die in de terminal verschijnt.

## De app gebruiken

### Stap 1 — Data uploaden

Start de app en sleep een CSV- of Parquet-bestand op het uploadveld. edu-synth toont een preview van de eerste rijen en detecteert automatisch kolomtypes.

!!! tip "Geschikte data"
    De app werkt het best met tabellen van minimaal 500 rijen. Kleinere datasets leveren minder stabiele correlatieramingen op.

### Stap 2 — Kolommen controleren

Na het uploaden zie je per kolom het gedetecteerde SDV-type (`categorical`, `numerical`, `datetime`, `id`). Pas dit aan als de detectie afwijkt.

#### Longitudinale data

Heeft elke entiteit (student, instelling) meerdere rijen over de tijd — bijvoorbeeld één rij per student per studiejaar — dan herkent de app dat meestal zelf en zet de vraag *"Heeft elke entiteit meerdere rijen over de tijd?"* op **Ja**. De app kiest dan de **PAR-synthesizer**, die de volgorde per entiteit behoudt, en vult de **sequence key** (ID per entiteit) en **sequence index** (tijdkolom) vast in. Controleer die twee en pas ze aan waar nodig. PAR-training is zwaarder dan de standaardsynthesizer en kan enkele minuten duren.

### Stap 3 — Synthetische data genereren

Kies het aantal gewenste rijen en klik **Genereer**. Het model traint op de achtergrond — bij grotere datasets kan dit enkele seconden duren.

Onder **Geavanceerd — reproduceerbaarheid** stel je een **random seed** in (standaard `42`). Dezelfde seed met dezelfde data levert identieke synthetische output. De seed komt terug in de geëxporteerde parameters en in het gegenereerde Python-codeblok.

### Stap 4 — Validatierapport bekijken

De app toont per kolom:

- **TV-afstand** (categorisch) — hoe goed kloppen de verhoudingen? Onder `0.2` is groen.
- **Wasserstein-afstand** (numeriek) — hoe dicht liggen de verdelingen bij elkaar?
- Distributieplots echte vs. synthetische data

### Stap 5 — Exporteren

Download de synthetische data als CSV. Daarnaast kun je het **validatierapport** als `validation_report.json` downloaden: dit bevat alle scores (per-kolom afstanden, sdmetrics, DCR/NNDR) en de synthese-parameters (synthesizer, aantal trainingsrijen, seed, SDV-versie). De app toont ook het bijbehorende SDV-codeblok om hetzelfde resultaat zelf te reproduceren.

## CLI (geavanceerd)

Voor batchverwerking is er ook een CLI:

```bash
edu-synth synthesize data.csv output.csv --rows 1000 --seed 42
edu-synth validate data.csv output.csv
```
