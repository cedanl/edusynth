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

## De app gebruiken

### Stap 1 — Data uploaden

Start de app en sleep een CSV- of Parquet-bestand op het uploadveld. ceda-synth toont een preview van de eerste rijen en detecteert automatisch kolomtypes.

!!! tip "Geschikte data"
    De app werkt het best met tabellen van minimaal 500 rijen. Kleinere datasets leveren minder stabiele correlatieramingen op.

### Stap 2 — Kolommen controleren

Na het uploaden zie je per kolom het gedetecteerde SDV-type (`categorical`, `numerical`, `datetime`, `id`). Pas dit aan als de detectie afwijkt.

### Stap 3 — Synthetische data genereren

Kies het aantal gewenste rijen en klik **Genereer**. Het model traint op de achtergrond — bij grotere datasets kan dit enkele seconden duren.

### Stap 4 — Validatierapport bekijken

De app toont per kolom:

- **TV-afstand** (categorisch) — hoe goed kloppen de verhoudingen? Onder `0.2` is groen.
- **Wasserstein-afstand** (numeriek) — hoe dicht liggen de verdelingen bij elkaar?
- Distributieplots echte vs. synthetische data

### Stap 5 — Exporteren

Download de synthetische data als CSV. De app toont ook het bijbehorende SDV-codeblok om hetzelfde resultaat zelf te reproduceren.

## CLI (geavanceerd)

Voor batchverwerking is er ook een CLI:

```bash
ceda-synth synthesize data.csv output.csv --rows 1000
ceda-synth validate data.csv output.csv
```
