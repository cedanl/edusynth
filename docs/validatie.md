# Validatie

Na het genereren toont edu-synth een validatierapport om te beoordelen hoe goed de synthetische data de echte data nabootst.

## Statistische getrouwheid

### Categorische kolommen — Total Variation (TV) afstand

Meet hoe ver de verhoudingen tussen categorieën afwijken. Bereik `[0, 1]`, lager is beter.

| Score | Betekenis |
|---|---|
| `< 0.1` | Uitstekend — distributies zijn nagenoeg identiek |
| `0.1 – 0.2` | Goed — kleine afwijkingen |
| `> 0.2` | Let op — de verdeling wijkt merkbaar af |

### Numerieke kolommen — genormaliseerde Wasserstein-1 afstand

Meet de gemiddelde verschuiving tussen de twee verdelingen. De ruwe Wasserstein-afstand hangt af van de kolomschaal (een afstand van 5 betekent iets anders voor jaren dan voor euro's), daarom wordt de waarde gedeeld door de **IQR** (interkwartielafstand) van de echte kolom. Het resultaat — de **score** — ligt op dezelfde schaal als de TV-afstand, zodat alle kolommen vergelijkbaar zijn en even zwaar meetellen in het eindoordeel.

| Score | Betekenis |
|---|---|
| `< 0.1` | Uitstekend |
| `0.1 – 0.2` | Goed |
| `> 0.2` | Let op — de verdeling wijkt merkbaar af |

Het rapport toont de score als primaire waarde en de ruwe Wasserstein-afstand ernaast.

## Geavanceerde kwaliteitsscore (sdmetrics)

De TV- en Wasserstein-scores zijn een snelle vuistregel. Voor een **uitgebreider** oordeel toont de app onder _Geavanceerde kwaliteitsscore (sdmetrics)_ ook de officiële [sdmetrics](https://docs.sdv.dev/sdmetrics) `QualityReport`:

- **Overall quality score** — één samenvattende score `[0, 1]`, hoger is beter.
- **Column Shapes** — verdeling per kolom (TVComplement voor categorisch, KSComplement voor numeriek).
- **Column Pair Trends** — samenhang tussen kolomparen, inclusief **categorisch × categorisch** (ContingencySimilarity) en categorisch × numeriek. Dit dekt verbanden die de eenvoudige Pearson-correlatie mist.

Kolomparen met een zwakke samenhang in de echte data (onder de sdmetrics-associatiedrempel) krijgen score *NaN* en blijven buiten beschouwing. Dat is geen fout: er is dan weinig verband om te bewaren. Bij meer dan 5000 rijen rekent de app op een steekproef.

Deze score komt uit een breed gebruikte, gestandaardiseerde methode (sdmetrics).

## Gebruiksoordeel — een vuistregel, geen norm

Boven de details toont de app een kort **gebruiksoordeel** (bijv. "Hoge statistische kwaliteit") met een bruikbaarheidsindicatie. Dit oordeel is bewust geformuleerd in termen van statistische kwaliteit en bruikbaarheid.

!!! note "Operationele vuistregel"
    Het oordeel is een operationele vuistregel op basis van afstandsmetrieken (TV, genormaliseerde Wasserstein), niet ontleend aan een vastgestelde norm. Beoordeel zelf of de kwaliteit volstaat voor het beoogde gebruik.

## Validatierapport exporteren (JSON)

In de tab _Download & Reproductie_ staat naast de CSV een knop **Download validation_report.json**. Dit machine-leesbare rapport bundelt alle scores die anders alleen in de UI zichtbaar zijn, plus de synthese-parameters:

- `generated_at`, `sdv_version`, `synthesizer`, `n_training_rows`, `n_generated_rows`, `random_seed`, `intended_use`
- `column_stats` — per kolom de afstand, score, metriek en of die binnen de drempel valt
- `sdmetrics` — overall score, Column Shapes en Column Pair Trends (indien beschikbaar)
- `privacy` — DCR-ratio, NNDR-mediaan en risiconiveau (indien beschikbaar)
- `usage_recommendation` en de bijbehorende disclaimer

Zo leg je het volledige oordeel reproduceerbaar vast.

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
