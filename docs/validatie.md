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

## Privacy

!!! warning "Geen privacygarantie"
    De huidige validatie meet alleen statistische gelijkenis, niet privacyrisico. DCR (Distance to Closest Record) en NNDR-validatie worden in een latere versie toegevoegd. Gebruik de synthetische data niet als privacyveilig zonder aanvullende beoordeling.

## Zelf valideren met SDV

```python
from sdmetrics.reports.single_table import QualityReport

report = QualityReport()
report.generate(real_df, synth_df, metadata.to_dict())
report.get_visualization(property_name="Column Shapes")
```
