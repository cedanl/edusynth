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
- **Longitudinale data met studentpaden** — voor data met meerdere rijen per entiteit over de tijd gebruikt de app een aparte sequentiële synthesizer (zie hieronder).

## Longitudinale synthese

Heeft een upload meerdere rijen per entiteit over de tijd (bv. één rij per student per studiejaar), dan kies je een **sequence key** (ID per entiteit) en een **sequence index** (tijdkolom). De app gebruikt dan een lichte, niet-neurale sequentiële synthesizer bovenop dezelfde Gaussian Copula:

1. **Plat maken** — de data wordt naar wide-formaat gezet: één rij per entiteit, met per tijdstap een kolom (`status_j1`, `status_j2`, `ec_j1`, …). Zo worden de verbanden tussen tijdstappen (doorstroomkansen) gewone kolom-correlaties die de Gaussian Copula al modelleert.
2. **Genereren** — de copula leert de gezamenlijke verdeling inclusief een expliciete reekslengte, en trekt nieuwe wide-rijen.
3. **Terugzetten** — elke synthetische rij wordt teruggevouwen naar het originele long-formaat. Twee regels houden de reeksen geldig: een reeks stopt bij een **eindstaat** (een categorie die in de echte data nooit een opvolger heeft, bv. gediplomeerd/uitgestroomd — automatisch afgeleid), en de lengte van reeksen zónder eindstaat komt uit de meegemodelleerde reekslengte.

Deze copula-aanpak is de aanbevolen default. Hij fit én samplet in seconden op CPU, behoudt de doorstroomkansen en genereert zowel categorische staten als numerieke kolommen per tijdstap.

De benchmark-harness dekt dit pad met een vaste doorstroom-dataset: per run meet ze de temporele kwaliteit (overgangsmatrix-afstand, autocorrelatie, verdeling van trajectlengtes) plus de fit-tijd, met een regressie-baseline net als voor het tabulaire spoor. De fit-tijd wordt gerapporteerd maar niet gegate — wall-clock verschilt per machine.

### PAR als optie

Onder *Synthesizer kiezen (geavanceerd)* staat SDV's `PARSynthesizer` (deep learning, LSTM) als alternatief. PAR is structureel trager — op CPU minuten in plaats van seconden — en scoort op kleine onderwijsdatasets vaak matiger, maar kan soms complexere temporele patronen leren. Kies PAR alleen als de copula tekortschiet. Het aantal epochs is instelbaar en de training toont een voortgangsbalk in procenten.

## Relatie tot SDV

edu-synth maakt SDV toegankelijk maar vervangt het niet. Voor maatwerk, complexe tabelrelaties of geavanceerde privacyvalidatie raden we aan direct met SDV te werken:

- [SDV-documentatie](https://docs.sdv.dev)
- [SDV GitHub](https://github.com/sdv-dev/SDV)
- [sdmetrics](https://github.com/sdv-dev/SDMetrics) voor uitgebreide validatie
