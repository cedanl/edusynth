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
  geslacht:
    dtype: categorical
    categories: ["1", "2"]
```

```bash
ceda-synth synthesize data.csv output.csv --schema schema.yaml --rows 1000
```

Zonder `--schema` detecteert de CLI kolomtypes automatisch, net als de app.

## Bekende schema's

Voorgeconfigureerde schema's voor CEDA-datasets (1CHO, CROHO) staan in `schemas/` zodra ze beschikbaar zijn.
