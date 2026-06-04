# ceda-synth

**ceda-synth** is een Streamlit-applicatie die [SDV (Synthetic Data Vault)](https://github.com/sdv-dev/SDV) toegankelijk maakt voor data-analisten bij Nederlandse hogeronderwijsinstellingen.

Upload je dataset, genereer een synthetische versie en bekijk direct of de statistische structuur klopt — zonder SDV zelf te hoeven configureren.

## Wat doet het?

1. **Uploaden** — sleep een CSV of Parquet-bestand in de app
2. **Genereren** — ceda-synth traint een SDV Gaussian Copula model op jouw data en genereert een synthetische versie
3. **Valideren** — de app toont per kolom hoe dicht de synthetische distributies bij de echte liggen
4. **Reproduceren** — wil je hetzelfde in je eigen Python-omgeving doen? De app genereert een SDV-codeblok dat je direct kunt kopiëren

## Starten

```bash
pip install ceda-synth
ceda-synth app
```

De browser opent automatisch op `http://localhost:8501`.

## Zelf doen met SDV

ceda-synth is een dunne laag. Alles wat de app doet, kun je ook direct met SDV:

```python
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
import pandas as pd

real = pd.read_csv("inschrijvingen.csv")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real)

synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(real)
synth = synthesizer.sample(num_rows=1000)
```

Zie de [SDV-documentatie](https://docs.sdv.dev) voor geavanceerde opties.
