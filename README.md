# edu-synth

![Demo](docs/assets/demo.gif)

Applicatielaag bovenop [SDV](https://github.com/sdv-dev/SDV) voor het genereren van synthetische versies van Nederlandse onderwijsdatasets.

Doelgroep: data-analisten en onderzoekers bij Nederlandse hogeronderwijsinstellingen die synthetische data willen genereren zonder zelf SDV te hoeven configureren.

Installeren en starten (vereist [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/cedanl/edusynth
cd edusynth
uv sync
uv run edu-synth app
```

```bash
edu-synth synthesize data.csv output.csv --schema schema.yaml --rows 1000
edu-synth validate data.csv output.csv
```

Documentatie: [cedanl.github.io/edusynth](https://cedanl.github.io/edusynth)
