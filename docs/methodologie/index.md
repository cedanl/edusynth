# Methodologie

edu-synth is een applicatielaag bovenop [SDV (Synthetic Data Vault)](https://github.com/sdv-dev/SDV). De syntheselogica zit volledig in SDV — edu-synth voegt een gebruiksvriendelijke interface en validatierapportage toe.

## Synthesemodel: Gaussian Copula

Voor enkelvoudige tabellen gebruikt edu-synth SDV's `GaussianCopulaSynthesizer`:

1. **Transformatie** — elke kolom wordt via de empirische CDF naar een uniforme verdeling omgezet
2. **Correlatieschatting** — de Spearman-correlatiematrix tussen kolommen wordt geschat
3. **Sampling** — gecorreleerde steekproeven worden gegenereerd via Cholesky-decompositie
4. **Terugprojectie** — de inverse CDF zet waarden terug naar de originele schaal

Dit model is stabiel, snel en goed interpreteerbaar — ideaal voor enkelvoudige onderwijstabellen.

### Marginale verdeling per kolom

Stap 1 (transformatie) gaat uit van een marginale verdeling per numerieke kolom. De standaard is een normaalverdeling. Die past slecht op **scheve of zero-inflated kolommen** — bijvoorbeeld een bedrag dat bij de meeste rijen 0 is met een lange staart. De gefitte normaal trekt dan onmogelijke waarden en de kolom scoort slecht in de validatie.

edu-synth detecteert zulke kolommen automatisch (hoge scheefheid of één dominante waarde, mits genoeg unieke waarden) en zet daar `gaussian_kde` op. KDE volgt de empirische vorm in plaats van een vaste aanname, wat de afstand tot de echte data fors verkleint. Op de benchmark-demodataset `adult` daalt de slechtst scorende kolom (`capital-gain`, 92 % nullen) van een afstand ~4,0 naar ~0,1.

KDE kost meer rekentijd en geheugen, dus edu-synth past het **gericht** toe op de kolommen die het nodig hebben — niet globaal. Je kunt de keuze per kolom overschrijven; zie [Configuratie › Verdeling per numerieke kolom](../configuratie.md#verdeling-per-numerieke-kolom-distribution).

### Waarom geen CTGAN/TVAE?

SDV biedt ook neurale synthesizers (CTGAN, TVAE). Gemeten tegen de benchmark-harness scoorden die op deze datasets **slechter** dan Gaussian Copula, terwijl ze fors trager zijn en een zware afhankelijkheid (PyTorch) meebrengen. Voor de typische onderwijstabel weegt dat niet op; gerichte distributie-tuning levert de winst zonder die kosten.

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
