# edusynth

**edusynth** genereert privacyveilige synthetische versies van Nederlandse onderwijsdatasets.

## Wat doet het?

Geef het je echte data en een schema — het leert de statistische structuur en genereert nieuwe rijen die er op lijken zonder echte persoonsgegevens te bevatten.

## Installatie

```bash
pip install edusynth
```

## Quickstart

```python
import pandas as pd
from edusynth import fit, sample, evaluate

real = pd.read_csv("inschrijvingen.csv")
model = fit(real, schema_path="schemas/1cijferho.yaml")
synth = sample(model, n_rows=1000)

report = evaluate(real, synth)
report.print()
```

Of via de CLI:

```bash
edusynth synthesize inschrijvingen.csv schemas/1cijferho.yaml output.csv --rows 1000
edusynth validate inschrijvingen.csv output.csv
```
