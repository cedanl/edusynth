# ceda-synth

Applicatielaag bovenop [SDV](https://github.com/sdv-dev/SDV) voor het genereren van privacyveilige synthetische versies van Nederlandse onderwijsdatasets.

Doelgroep: data-analisten en onderzoekers bij Nederlandse hogeronderwijsinstellingen die synthetische data willen genereren zonder zelf SDV te hoeven configureren.

```bash
pip install ceda-synth
```

```bash
ceda-synth synthesize data.csv schema.yaml output.csv --rows 1000
ceda-synth validate data.csv output.csv
```

Documentatie: [cedanl.github.io/ceda-synth](https://cedanl.github.io/ceda-synth)
