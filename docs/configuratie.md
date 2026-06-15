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
    min: 2015
    max: 2024
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

Geef bij een `date`-kolom het formaat mee in [strftime](https://strftime.org/)-notatie. Zonder `datetime_format` gaat SDV uit van ISO 8601 (`YYYY-MM-DD`) en mislukt het parsen van DUO-datums (`YYYYMMDD`). Veelgebruikte waarden: `"%Y%m%d"` (20190101), `"%Y-%m-%d"` (2019-01-01), `"%d-%m-%Y"` (01-01-2019). De app herkent het patroon en toont het in de kolomtype-hints.

### Bereik (`min` / `max`)

Met `min` en `max` leg je het domein van een numerieke kolom vast. Waarden die SDV daarbuiten genereert worden na het samplen naar de grens geklemd — zo komen er geen negatieve aantallen of jaren buiten bereik uit.

### Reproduceerbaarheid (`--seed`)

Geef `--seed <getal>` mee om reproduceerbare output te krijgen: dezelfde seed met dezelfde inputdata levert een identieke synthetische dataset.

```bash
ceda-synth synthesize data.csv output.csv --seed 42
```

Zonder `--seed` is elke run anders. In de app staat dezelfde optie onder **Geavanceerd — reproduceerbaarheid**.

## Bekende schema's

Voorgeconfigureerde schema's voor CEDA-datasets (1CHO, CROHO) staan in `schemas/` zodra ze beschikbaar zijn.
