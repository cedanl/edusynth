# Configuratie

## Kolomtypes in de app

edu-synth detecteert kolomtypes automatisch via SDV's `detect_from_dataframe`. In de UI kun je dit per kolom overschrijven:

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
edu-synth synthesize data.csv output.csv --schema schema.yaml --rows 1000
```

Zonder `--schema` detecteert de CLI kolomtypes automatisch, net als de app.

### Cross-kolom constraints

SDV bewaakt per kolom al de range en het type, maar leidt geen logische verbanden _tussen_ kolommen af. Die dwing je af tijdens de synthese (geen correctie achteraf), op twee manieren:

- **In de app** — bij stap 2 (Genereren) onder **Logische regels (optioneel)**: kies via dropdowns een volgorde tussen twee kolommen (≤, <, ≥, >) of selecteer kolommen die alleen in bestaande combinaties mogen voorkomen. Geen YAML nodig.
- **In de CLI/batch** — via een optioneel `constraints`-blok in het schema.

Beide paden vertalen naar dezelfde SDV-constraints. Het schema-blok ziet er zo uit:

```yaml
constraints:
  - type: inequality          # low ≤ high
    low: inschrijvingsjaar
    high: uitschrijvingsjaar
    strict: false             # optioneel; false = gelijk mag (standaard), true = strikt <
  - type: fixed_combinations  # alleen combinaties die in de data voorkomen
    columns: [instellingscode, opleidingscode]
```

| Type | Wat het afdwingt | Verplichte sleutels |
|---|---|---|
| `inequality` | `low ≤ high` (of strikt `<` met `strict: true`) | `low`, `high` |
| `fixed_combinations` | Alleen in de data voorkomende combinaties van de kolommen | `columns` |

Botsen de regels met de data of bestaat een kolom niet, dan stopt de synthese met een duidelijke melding.

### Datumkolommen (`datetime_format`)

Geef bij een `date`-kolom het formaat mee in [strftime](https://strftime.org/)-notatie. Zonder `datetime_format` gaat SDV uit van ISO 8601 (`YYYY-MM-DD`) en mislukt het parsen van DUO-datums (`YYYYMMDD`). Veelgebruikte waarden: `"%Y%m%d"` (20190101), `"%Y-%m-%d"` (2019-01-01), `"%d-%m-%Y"` (01-01-2019).

In de **app** is een schema niet nodig: zodra je een kolom op `Datum` zet, detecteert de app het formaat uit de data en geeft het automatisch door aan SDV.

### Domein van numerieke kolommen

SDV houdt numerieke waarden automatisch binnen de range van je trainingsdata en respecteert gehele getallen (geen `1,2` aanmeldingen) — dit staat standaard aan. Categorische kolommen kunnen per definitie geen waarde buiten de bestaande categorieën krijgen.

Cross-kolom-logica zoals `inschrijvingsjaar ≤ uitschrijvingsjaar` of geldige categorie-combinaties dwing je af via [cross-kolom constraints](#cross-kolom-constraints) — in de app (Logische regels) of in het schema (CLI/batch). Een vast domein strakker dan de data staat nog op de roadmap.

### Verdeling per numerieke kolom (`distribution`)

GaussianCopula modelleert elke numerieke kolom met een marginale verdeling. De standaard (`norm`) past slecht op scheve of zero-inflated kolommen — denk aan een bedrag dat bij de meeste rijen 0 is met een lange staart. De gefitte normaal trekt daar onmogelijke waarden en de afstand tot de echte data loopt op.

edu-synth detecteert zulke kolommen automatisch (hoge scheefheid of één dominante waarde, met genoeg unieke waarden) en zet daar `gaussian_kde` op, die de echte vorm volgt. De overige kolommen blijven op `norm`.

- **In de app** — bij stap 2 onder **Verdelingen**: aanbevolen kolommen staan met ⭐ en hebben `gaussian_kde` voorgeselecteerd. Pas per kolom aan of laat staan.
- **In de CLI/batch** — geef per kolom een `distribution` mee in het schema. Dit overschrijft de auto-detectie.

```yaml
columns:
  capital_gain:
    dtype: integer
    distribution: gaussian_kde
```

Geldige waarden: `norm`, `beta`, `truncnorm`, `uniform`, `gamma`, `gaussian_kde`. `gaussian_kde` is het meest flexibel maar kost meer rekentijd en geheugen; daarom past edu-synth het gericht toe in plaats van overal.

### Reproduceerbaarheid (`--seed`)

Geef `--seed <getal>` mee om reproduceerbare output te krijgen: dezelfde seed met dezelfde inputdata levert een identieke synthetische dataset.

```bash
edu-synth synthesize data.csv output.csv --seed 42
```

Zonder `--seed` is elke run anders. In de app staat dezelfde optie onder **Geavanceerd — reproduceerbaarheid**.

## Bekende schema's

Voorgeconfigureerde schema's voor CEDA-datasets (1CHO, CROHO) staan in `schemas/` zodra ze beschikbaar zijn.
