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

De app leidt je in drie stappen door de flow: **Data laden → Genereren → Resultaten**. Je ziet één stap tegelijk; de balk bovenaan toont waar je bent. Bij stap 2 (Genereren) zijn kolomtypes en synthesizer-instellingen optionele verfijning — bij demo- of schone data klik je meteen op Genereer. Met **Vorige**/**Volgende** loop je heen en weer. Wil je na het bekijken van de resultaten iets aanpassen, klik dan op **Terug naar instellingen** en genereer opnieuw.

### Stap 1 — Data uploaden

Start de app. Vink eerst de toestemming aan dat je deze data mag verwerken — het uploadveld wordt pas daarna actief. Verwerking vindt volledig lokaal in je browser-sessie plaats; data wordt nergens opgeslagen of verzonden. Sleep daarna een CSV- of Parquet-bestand op het uploadveld. edu-synth toont een preview van de eerste rijen en detecteert automatisch kolomtypes.

!!! tip "Geschikte data"
    De app werkt het best met tabellen van minimaal 500 rijen. Kleinere datasets leveren minder stabiele correlatieramingen op.

### Stap 2 — Kolommen controleren

Na het uploaden zie je per kolom het gedetecteerde SDV-type (`categorical`, `numerical`, `datetime`, `id`). Pas dit aan als de detectie afwijkt. Met **"Pas zekere aanbevelingen toe"** neem je in één klik alleen de suggesties met hoge zekerheid (≥90%) over. Onzekere suggesties worden gemarkeerd en bevestig je zelf — die worden nooit automatisch toegepast.

#### Longitudinale data

Heeft elke entiteit (student, instelling) meerdere rijen over de tijd — bijvoorbeeld één rij per student per studiejaar — dan herkent de app dat meestal zelf en zet de vraag *"Heeft elke entiteit meerdere rijen over de tijd?"* op **Ja**. De app kiest dan de **PAR-synthesizer**, die de volgorde per entiteit behoudt, en vult de **sequence key** (ID per entiteit) en **sequence index** (tijdkolom) vast in. Controleer die twee en pas ze aan waar nodig. PAR-training is zwaarder dan de standaardsynthesizer en kan enkele minuten duren.

### Stap 3 — Synthetische data genereren

Kies het aantal gewenste rijen en klik **Genereer**. Het model traint op de achtergrond — bij grotere datasets kan dit enkele seconden duren.

Onder **Geavanceerd — reproduceerbaarheid** stel je een **random seed** in (standaard `42`). Dezelfde seed met dezelfde data levert identieke synthetische output. De seed komt terug in de geëxporteerde parameters en in het gegenereerde Python-codeblok.

### Stap 4 — Validatierapport bekijken

Boven de tabbladen verschijnt een gekleurde **oordeel-banner** (groen/oranje/rood) met het overall oordeel — Hoge bruikbaarheid, Bruikbaar met voorbehoud of Niet aanbevolen — en een advieszin. Zo zie je het oordeel ook als je direct naar het Download-tabblad gaat. Bij hoog risico staat er een extra waarschuwing bij de downloadknop.

De app toont per kolom:

- **TV-afstand** (categorisch) — hoe goed kloppen de verhoudingen? Onder `0.2` is groen.
- **Wasserstein-afstand** (numeriek) — hoe dicht liggen de verdelingen bij elkaar?
- Distributieplots echte vs. synthetische data. Standaard toont de app de 8 meest afwijkende kolommen; via de selectie bovenaan de tab kies je zelf welke kolommen je ziet. Dat houdt brede datasets (tientallen kolommen) overzichtelijk.

### Stap 5 — Exporteren

Download de synthetische data als CSV. Daarnaast kun je het **validatierapport** als `validation_report.json` downloaden: dit bevat alle scores (per-kolom afstanden, sdmetrics, DCR/NNDR) en de synthese-parameters (synthesizer, aantal trainingsrijen, seed, SDV-versie). Een korte samenvatting in gewone taal toont welke kolomtypes zijn herkend en hoeveel kolommen als privacygevoelig zijn gemarkeerd. De ruwe SDV-metadata (JSON) en het SDV-codeblok om het resultaat zelf te reproduceren staan onder **Technische details**.

## CLI (geavanceerd)

Voor batchverwerking is er ook een CLI:

```bash
edu-synth synthesize data.csv output.csv --rows 1000 --seed 42
edu-synth validate data.csv output.csv
```
