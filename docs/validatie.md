# Validatie

edusynth valideert synthetische data op twee assen.

## Statistische getrouwheid

Hoe goed bootsen de distributies de echte data na?

- **Total Variation (TV) afstand** voor categorische kolommen — `[0, 1]`, lager is beter
- **Wasserstein-1 afstand** voor numerieke kolommen — schaalonafhankelijk, lager is beter

## Privacy

!!! warning "Nog niet geïmplementeerd"
    DCR (Distance to Closest Record) en NNDR-validatie worden in een latere versie toegevoegd. Gebruik de huidige validatie niet als privacygarantie.

## Drempelwaarden

De standaard TV-drempel is `0.2`. Kolommen boven deze drempel worden gemarkeerd als `ok=False` in het rapport.

## Gebruik

```python
from edusynth import evaluate

report = evaluate(real_df, synth_df)
report.print()          # Rich-tabel in de terminal
df = report.to_dataframe()  # pandas DataFrame voor verdere analyse
report.passed()         # True als alle kolommen onder de drempel zitten
```
