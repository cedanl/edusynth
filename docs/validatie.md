# Validatie

Na het genereren toont ceda-synth een validatierapport om te beoordelen hoe goed de synthetische data de echte data nabootst.

## Statistische getrouwheid

### Categorische kolommen — Total Variation (TV) afstand

Meet hoe ver de verhoudingen tussen categorieën afwijken. Bereik `[0, 1]`, lager is beter.

| Score | Betekenis |
|---|---|
| `< 0.1` | Uitstekend — distributies zijn nagenoeg identiek |
| `0.1 – 0.2` | Goed — kleine afwijkingen |
| `> 0.2` | Let op — de verdeling wijkt merkbaar af |

### Numerieke kolommen — Wasserstein-1 afstand

Meet de gemiddelde verschuiving tussen de twee verdelingen. De schaal hangt af van de kolom (een afstand van 5 is anders voor een kolom in jaren dan voor een kolom in euro's).

## Distributieplots

De app toont naast de scores ook histogrammen en staafdiagrammen van echte vs. synthetische data per kolom — zo zie je direct waar afwijkingen zitten.

## Privacy — DCR / NNDR

De app schat het re-identificatierisico via een holdout-vergelijking:

1. De echte data wordt gesplitst in een train- (80%) en holdout-set (20%).
2. **DCR** (Distance to Closest Record): hoe dicht zit elke synthetische rij bij de dichtstbijzijnde trainingsrij?
3. Als baseline wordt dezelfde afstand gemeten voor de holdout-set.
4. Liggen synthetische rijen even ver van de trainingsdata als de holdout-set, dan gedragen ze zich als onbekende data — laag risico.

De **DCR-ratio** vat dit samen:

| DCR-ratio | Risico |
|---|---|
| `> 0.9` | Laag — synthetisch gedraagt zich als onbekende data |
| `0.7 – 0.9` | Matig — beoordeel quasi-identifiers vóór publicatie |
| `< 0.7` | Hoog — synthetisch zit te dicht op de trainingsdata |

### Welke kolommen tellen mee?

Zowel numerieke als categorische kolommen worden meegenomen. In onderwijsdata zijn de quasi-identifiers (`geslacht`, `instellingscode`, `opleidingscode`, `woonplaats`) juist categorisch — die mogen niet ontbreken in de berekening. Numerieke kolommen worden geschaald, categorische worden one-hot gecodeerd.

Kolommen met heel veel unieke waarden (vrije tekst, namen, identifiers) worden uitgesloten en de app waarschuwt erover — beoordeel die handmatig. De primary key telt nooit mee; die is vervangen door nieuwe anonieme ID's.

!!! warning "Geen formele privacygarantie"
    DCR/NNDR is een statistische schatting, geen formele garantie. Een DPIA blijft vereist vóór publicatie. Gebruik de synthetische data niet als privacyveilig zonder aanvullende beoordeling.

## Zelf valideren met SDV

```python
from sdmetrics.reports.single_table import QualityReport

report = QualityReport()
report.generate(real_df, synth_df, metadata.to_dict())
report.get_visualization(property_name="Column Shapes")
```
