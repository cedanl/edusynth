# edu-synth

![Demo](docs/assets/demo.gif)

Applicatielaag bovenop [SDV](https://github.com/sdv-dev/SDV) voor het genereren van privacyveilige synthetische versies van Nederlandse onderwijsdatasets.

Doelgroep: data-analisten en onderzoekers bij Nederlandse hogeronderwijsinstellingen die synthetische data willen genereren zonder zelf SDV te hoeven configureren.

```bash
pip install edu-synth
```

Of lokaal:

```bash
uv sync
uv run edu-synth app
```

```bash
edu-synth synthesize data.csv schema.yaml output.csv --rows 1000
edu-synth validate data.csv output.csv
```

Documentatie: [cedanl.github.io/edu-synth](https://cedanl.github.io/edu-synth)
