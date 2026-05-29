# Aan de slag

## Installatie

```bash
pip install edusynth
```

## Stap 1 — Schema voorbereiden

Een schema beschrijft de kolommen van je dataset. Maak een YAML-bestand:

```yaml
name: mijn_dataset
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
    probabilities: [0.48, 0.52]
```

## Stap 2 — Model trainen

```python
import pandas as pd
from edusynth import fit

real = pd.read_csv("data.csv")
model = fit(real, schema_path="schema.yaml")
```

## Stap 3 — Data genereren

```python
from edusynth import sample

synth = sample(model, n_rows=1000)
synth.to_csv("synthetisch.csv", index=False)
```

## Stap 4 — Valideren

```python
from edusynth import evaluate

report = evaluate(real, synth)
report.print()
```
