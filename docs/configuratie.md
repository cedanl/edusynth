# Configuratie

## Kolomtypes in de app

ceda-synth detecteert kolomtypes automatisch via SDV's `detect_from_dataframe`. In de UI kun je dit per kolom overschrijven:

| App-label | SDV-type | Gebruik voor |
|---|---|---|
| Categorisch | `categorical` | Codes, geslacht, opleiding |
| Numeriek (geheel) | `numerical` | Jaren, aantallen |
| Numeriek (decimaal) | `numerical` | Cijfers, ratio's |
| Datum | `datetime` | Datumkolommen |
| ID / vrije tekst | `id` | Sleutels, studentnummers |

## Primaire sleutel

Markeer een kolom als primaire sleutel als die unieke rij-identifiers bevat. SDV genereert dan nieuwe unieke waarden in plaats van bestaande te kopiëren.

## Geavanceerd: YAML-schema

Voor batchverwerking via de CLI kun je een schema-bestand meegeven:

```yaml
name: naam_van_dataset

columns:
  student_id:
    dtype: string
    role: primary_key
  inschrijvingsjaar:
    dtype: integer
  inschrijfdatum:
    dtype: date
    datetime_format: "%Y%m%d"
  geslacht:
    dtype: categorical
    categories: ["1", "2"]
```

```bash
ceda-synth synthesize data.csv output.csv --schema schema.yaml --rows 1000
```

Zonder `--schema` detecteert de CLI kolomtypes automatisch, net als de app.

### Datumkolommen (`datetime_format`)

Geef bij een `date`-kolom het formaat mee in [strftime](https://strftime.org/)-notatie. Zonder `datetime_format` gaat SDV uit van ISO 8601 (`YYYY-MM-DD`) en mislukt het parsen van DUO-datums (`YYYYMMDD`). Veelgebruikte waarden: `"%Y%m%d"` (20190101), `"%Y-%m-%d"` (2019-01-01), `"%d-%m-%Y"` (01-01-2019).

In de **app** is een schema niet nodig: zodra je een kolom op `Datum` zet, detecteert de app het formaat uit de data en geeft het automatisch door aan SDV.

### Domein van numerieke kolommen

SDV houdt numerieke waarden automatisch binnen de range van je trainingsdata en respecteert gehele getallen (geen `1,2` aanmeldingen) — dit staat standaard aan. Categorische kolommen kunnen per definitie geen waarde buiten de bestaande categorieën krijgen.

Hardere regels die SDV niet uit de data afleidt — een vast domein strakker dan de data, of cross-kolom-logica zoals `inschrijfjaar ≤ uitschrijfjaar` — staan op de roadmap.

### Reproduceerbaarheid (`--seed`)

Geef `--seed <getal>` mee om reproduceerbare output te krijgen: dezelfde seed met dezelfde inputdata levert een identieke synthetische dataset.

```bash
ceda-synth synthesize data.csv output.csv --seed 42
```

Zonder `--seed` is elke run anders. In de app staat dezelfde optie onder **Geavanceerd — reproduceerbaarheid**.

## Bekende schema's

Voorgeconfigureerde schema's voor CEDA-datasets (1CHO, CROHO) staan in `schemas/` zodra ze beschikbaar zijn.
