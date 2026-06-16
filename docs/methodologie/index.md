# Methodologie

edu-synth is een applicatielaag bovenop [SDV (Synthetic Data Vault)](https://github.com/sdv-dev/SDV). De syntheselogica zit volledig in SDV — edu-synth voegt een gebruiksvriendelijke interface en validatierapportage toe.

## Synthesemodel: Gaussian Copula

Voor enkelvoudige tabellen gebruikt edu-synth SDV's `GaussianCopulaSynthesizer`:

1. **Transformatie** — elke kolom wordt via de empirische CDF naar een uniforme verdeling omgezet
2. **Correlatieschatting** — de Spearman-correlatiematrix tussen kolommen wordt geschat
3. **Sampling** — gecorreleerde steekproeven worden gegenereerd via Cholesky-decompositie
4. **Terugprojectie** — de inverse CDF zet waarden terug naar de originele schaal

Dit model is stabiel, snel en goed interpreteerbaar — ideaal voor enkelvoudige onderwijstabellen.

## Wanneer werkt het goed?

- Enkelvoudige tabellen met > 500 rijen
- Kolommen met relatief lineaire onderlinge verbanden
- Stationaire data (geen tijdsafhankelijkheid per rij)

## Wanneer werkt het minder goed?

- **Kleine datasets (< 500 rijen)** — correlatieramingen worden instabiel
- **Sterke niet-lineaire verbanden** — Gaussian Copula mist complexe interacties
- **Longitudinale data met studentpaden** — hiervoor schakelt de app over op SDV's `PARSynthesizer`. Geef je een upload met meerdere rijen per entiteit over de tijd, dan kies je een sequence key (ID per entiteit) en sequence index (tijdkolom); PAR behoudt de volgorde binnen elke entiteit.

## Relatie tot SDV

edu-synth maakt SDV toegankelijk maar vervangt het niet. Voor maatwerk, complexe tabelrelaties of geavanceerde privacyvalidatie raden we aan direct met SDV te werken:

- [SDV-documentatie](https://docs.sdv.dev)
- [SDV GitHub](https://github.com/sdv-dev/SDV)
- [sdmetrics](https://github.com/sdv-dev/SDMetrics) voor uitgebreide validatie
