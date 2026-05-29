# Configuratie

Alle dataset-specifieke configuratie staat in een schema YAML-bestand. Er is geen configuratie hardcoded in de package.

## Schema-structuur

```yaml
name: naam_van_dataset
description: Optionele beschrijving.

columns:
  kolomnaam:
    dtype: string | integer | float | categorical | date
    role: primary_key | null          # optioneel
    min: 0                            # voor integer/float
    max: 100                          # voor integer/float
    categories: ["a", "b", "c"]      # voor categorical
    probabilities: [0.5, 0.3, 0.2]   # voor categorical
    nullable: false
    description: "Optionele toelichting"
```

## Dtypes

| dtype | SDV-type | Gebruik voor |
|---|---|---|
| `categorical` | categorical | Codes, geslacht, regio |
| `integer` | numerical | Jaren, aantallen |
| `float` | numerical | Cijfers, ratio's |
| `string` | id | Vrije tekst, IDs |
| `date` | datetime | Datumkolommen |

## Bekende schema's

Voorgeconfigureerde schema's voor CEDA-datasets staan in `schemas/` zodra ze beschikbaar zijn.
