# Validatie

Na het genereren toont edu-synth een validatierapport om te beoordelen hoe goed de synthetische data de echte data nabootst.

## Statistische getrouwheid

### Categorische kolommen — Total Variation (TV) afstand

Meet hoe ver de verhoudingen tussen categorieën afwijken. Bereik `[0, 1]`, lager is beter.

| Score | Betekenis |
|---|---|
| `< 0.1` | Uitstekend — distributies zijn nagenoeg identiek |
| `0.1 – 0.2` | Goed — kleine afwijkingen |
| `> 0.2` | Let op — de verdeling wijkt merkbaar af |

### Numerieke kolommen — genormaliseerde Wasserstein-1 afstand

Meet de gemiddelde verschuiving tussen de twee verdelingen. De ruwe Wasserstein-afstand hangt af van de kolomschaal (een afstand van 5 betekent iets anders voor jaren dan voor euro's), daarom wordt de waarde gedeeld door de **IQR** (interkwartielafstand) van de echte kolom. Het resultaat — de **score** — ligt op dezelfde schaal als de TV-afstand, zodat alle kolommen vergelijkbaar zijn en even zwaar meetellen in het eindoordeel.

| Score | Betekenis |
|---|---|
| `< 0.1` | Uitstekend |
| `0.1 – 0.2` | Goed |
| `> 0.2` | Let op — de verdeling wijkt merkbaar af |

Het rapport toont de score als primaire waarde en de ruwe Wasserstein-afstand ernaast.

Ja/nee-kolommen (booleaans) worden als categorisch behandeld en via de TV-afstand vergeleken, niet via Wasserstein.

### Datum- en id-kolommen

- **Datums** worden naar een tijdstip omgezet en als numerieke kolom gescoord (Wasserstein), niet als categorie. Een datumkolom heeft veel unieke waarden; als categorie zou die bijna altijd "mislukken" terwijl de synthese prima is. De datumvergelijking voorkomt dat valse alarm.
- **Id-kolommen** (identifiers, primary key) blijven buiten de score. Een id heeft geen verdeling om na te bootsen; een afwijking erop zegt niets over de kwaliteit.

Hiervoor gebruikt de app de kolomtypes uit de configuratie. Staat een datumkolom nog op _categorisch_, pas het type dan aan onder _Kolomtypes aanpassen_ — anders wordt de datum alsnog als categorie vergeleken.

## Geavanceerde kwaliteitsscore (sdmetrics)

De TV- en Wasserstein-scores zijn een snelle vuistregel. Voor een **uitgebreider** oordeel toont de app onder _Geavanceerde kwaliteitsscore (sdmetrics)_ ook de officiële [sdmetrics](https://docs.sdv.dev/sdmetrics) `QualityReport`:

- **Overall quality score** — één samenvattende score `[0, 1]`, hoger is beter.
- **Column Shapes** — verdeling per kolom (TVComplement voor categorisch, KSComplement voor numeriek).
- **Column Pair Trends** — samenhang tussen kolomparen, inclusief **categorisch × categorisch** (ContingencySimilarity) en categorisch × numeriek. Dit dekt verbanden die de eenvoudige Pearson-correlatie mist.

Kolomparen met een zwakke samenhang in de echte data (onder de sdmetrics-associatiedrempel) krijgen score *NaN* en blijven buiten beschouwing. Dat is geen fout: er is dan weinig verband om te bewaren. Bij meer dan 5000 rijen rekent de app op een steekproef.

Deze score komt uit een breed gebruikte, gestandaardiseerde methode (sdmetrics).

## Samenhang tussen kolommen

Naast de verdeling per kolom controleert de app of de **verbanden tussen kolommen** behouden blijven. Direct onder de scorecards staat een vergelijking van de Pearson-correlatie tussen alle numerieke kolomparen in de echte versus de synthetische data. Paren waarbij het verschil groter is dan `0.1` worden getoond.

Een **tekenomslag** is het belangrijkste signaal: een verband dat in de echte data positief is maar in de synthetische data negatief is omgeklapt. Wie op zo'n verband een analyse of model bouwt, trekt een omgekeerde conclusie. Hoe zwaar dit telt hangt af van de **sterkte van het echte verband**: een omslag tussen twee zwakke correlaties (bijv. +0.20 → −0.11) is grotendeels ruis — zeker bij weinig rijen — en mag het oordeel niet domineren.

De samenhang krijgt een eigen scorecard (**Samenhang**) en weegt mee in het eindoordeel:

| Samenhang | Betekenis |
|---|---|
| Goed bewaard | Geen kolomparen wijken meer dan `0.1` af |
| Enkele afwijkingen | Een matig verband (`\|echt\| ≥ 0.2`) wijkt merkbaar af |
| Verband omgeklapt | Een **sterk** verband (`\|echt\| ≥ 0.4`) is van teken gewisseld, of een verschuiving `> 0.5` |

Wijkt alleen een zwak verband (`\|echt\| < 0.2`) af, dan blijft de samenhang _Goed bewaard_ — dat is ruis, geen kwaliteitsprobleem.

Heeft de dataset minder dan twee numerieke kolommen, dan is deze controle niet berekenbaar; dat verlaagt het oordeel niet. Voor categorisch × categorisch en categorisch × numeriek dekt de sdmetrics _Column Pair Trends_ (hierboven) de samenhang aanvullend af.

## Temporele getrouwheid (longitudinale data)

Longitudinale data heeft meerdere rijen per entiteit (bijvoorbeeld één rij per student per studiejaar), geordend op een tijd-index. De metrieken hierboven vergelijken kolommen los van elkaar en zeggen niets over of het gedrag **over de tijd** klopt. Voor sequentiële data komen daar drie signalen bij, elk op dezelfde `[0, ~]`-schaal als de TV- en Wasserstein-score (drempel `0.2`):

| Signaal | Voor welke kolom | Wat het meet |
|---|---|---|
| **Overgangsmatrix** | Categorische staten | Behoud van de doorstroomkansen: gaat de synthese van staat naar staat met dezelfde kansen als de echte data (bijv. _jaar 1 → jaar 2_ vs. _jaar 1 → uitval_)? |
| **Autocorrelatie** | Numeriek | Behoud van de samenhang tussen opeenvolgende tijdstappen (lag-1), gemiddeld over de sequenties |
| **Sequentielengte** | Alle | Kloppen de lengtes van de sequenties (aantal rijen per entiteit)? |

De overgangsmatrix-afstand is een TV-afstand per bronstaat, gewogen naar hoe vaak die bronstaat in de echte data voorkomt — een veelvoorkomende overgang telt zwaarder dan een zeldzame. De sequentielengte-verdeling wordt vergeleken met de genormaliseerde Wasserstein-afstand, net als numerieke kolommen. Id-kolommen, de sequentie-key en de tijd-index zelf blijven buiten beschouwing.

Bij longitudinale synthese toont de app deze signalen als scorecard **Tijdsgedrag** met een gewone-taal-oordeel, op dezelfde plek waar tabulaire data de scorecard Samenhang krijgt. Het tijdsgedrag telt mee in het overall bruikbaarheidsoordeel, en de temporele scores komen mee in de `validation_report.json`-export onder `temporal`. De tabulaire samenhang-sectie (correlaties over de platgeslagen kolommen) wordt bij longitudinale data weggeklapt — die is grotendeels ruis naast de temporele metrieken.

Naast de scores toont de tab _Distributies_ bij longitudinale data een **Tijd**-sectie met de metriek als beeld, echt vs. synthetisch:

- **Overgangsmatrix-heatmaps** (categorische kolom) — echt en synthetisch naast elkaar; cellen die verschillen zijn de doorstroomkansen die de synthese mist.
- **Staatverdeling over de tijd** (categorische kolom) — hoe de verdeling over de statussen per tijdstap verloopt.
- **Gemiddelde per tijdstap** (numerieke kolom) — wanneer in de tijd de synthetische reeks van de echte afwijkt.

Standaard toont de sectie de sterkst afwijkende kolom; via een keuzemenu bekijk je een andere kolom.

## Gebruiksoordeel — een vuistregel, geen norm

Boven de details toont de app een kort **gebruiksoordeel** (bijv. "Hoge statistische kwaliteit") met een bruikbaarheidsindicatie. Dit oordeel is bewust geformuleerd in termen van statistische kwaliteit en bruikbaarheid.

Het eindoordeel combineert de deeloordelen (verdeling, samenhang, privacy — en bij longitudinale data ook tijdsgedrag) zónder solo-veto: één enkele zwakke dimensie verlaagt naar **Bruikbaar met voorbehoud**, niet meteen naar **Niet aanbevolen**. Een dataset die op 94% van de kolommen goed scoort wordt zo niet afgekeurd op één afwijking. **Niet aanbevolen** verschijnt alleen bij een privacy-risico (altijd zwaarwegend) of wanneer twee deeloordelen tegelijk hoog zijn.

!!! note "Operationele vuistregel"
    Het oordeel is een operationele vuistregel op basis van afstandsmetrieken (TV, genormaliseerde Wasserstein), niet ontleend aan een vastgestelde norm. Beoordeel zelf of de kwaliteit volstaat voor het beoogde gebruik.

### Verbeteradvies bij een matig of onvoldoende oordeel

Is het oordeel _Bruikbaar met voorbehoud_ of _Niet aanbevolen_, dan toont de app onder het oordeel een blok **Wat kun je verbeteren?**. Dat koppelt de slechtst scorende kolommen aan een waarschijnlijke oorzaak en een concrete actie:

| Gemeten signaal | Advies |
|---|---|
| Kolom lijkt verkeerd getypeerd | Pas het type aan onder _Kolomtypes aanpassen_ |
| Kolom verliest meerdere pieken (multimodaal) | Kies een andere verdeling (`gaussian_kde`) onder _Verdelingen_ |
| Categorische kolom met veel unieke waarden | Laat de kolom weg of groepeer waarden |
| Te weinig rijen (< 500) | Gebruik meer data voor stabielere synthese |
| Hoog privacyrisico | Markeer identifiers als _ID_ of genereer minder rijen |
| Afwijkend tijdsgedrag (longitudinaal) | Synthesizer-afhankelijk: draaide de synthese op de lichte copula, dan raadt de app de PAR-synthesizer aan; draaide hij al op PAR, dan epochs verhogen of meer entiteiten gebruiken |

Het tijdsgedrag-advies is **synthesizer-bewust**: het raadt PAR alleen aan wanneer de synthese níét al op PAR draaide, zodat de suggestie nooit circulair is. Bij een goed oordeel verschijnt het blok niet. De adviezen zijn suggesties op basis van de gemeten scores — geen garantie dat één aanpassing het oordeel omdraait.

## Validatierapport exporteren (JSON)

In de tab _Download & Reproductie_ staat naast de CSV een knop **Download validation_report.json**. Dit machine-leesbare rapport bundelt alle scores die anders alleen in de UI zichtbaar zijn, plus de synthese-parameters:

- `generated_at`, `sdv_version`, `synthesizer`, `n_training_rows`, `n_generated_rows`, `random_seed`, `intended_use`
- `column_stats` — per kolom de afstand, score, metriek en of die binnen de drempel valt
- `sdmetrics` — overall score, Column Shapes en Column Pair Trends (indien beschikbaar)
- `privacy` — DCR-ratio, NNDR-mediaan en risiconiveau (indien beschikbaar)
- `usage_recommendation` en de bijbehorende disclaimer

Zo leg je het volledige oordeel reproduceerbaar vast.

## Validatierapport exporteren (PDF)

Naast de JSON staat in dezelfde tab een knop **Download rapport (PDF)**. Het PDF is een leesbaar rapport om te delen of archiveren en bevat het oordeel, de scorekaarten, de verdeling per kolom, het tijdsgedrag-detail (bij longitudinale data), de privacymaten en de reproductie-parameters. Het PDF put uit dezelfde bron als de JSON, dus beide blijven consistent.

Het PDF wordt headless gegenereerd met [reportlab](https://www.reportlab.com/) — pure Python, zonder systeemafhankelijkheden, dus het werkt ook bij een lokale `pip install` op Windows en macOS.

## Distributieplots

De app toont naast de scores ook histogrammen en staafdiagrammen van echte vs. synthetische data per kolom — zo zie je direct waar afwijkingen zitten. Bij longitudinale data komt daar de **Tijd**-sectie bij (zie hierboven) met de overgangs- en trajectbeelden over de tijd.

## Privacy — DCR / NNDR

De app schat het re-identificatierisico via een holdout-vergelijking:

1. De echte data wordt gesplitst in een train- (80%) en holdout-set (20%).
2. **DCR** (Distance to Closest Record): hoe dicht zit elke synthetische rij bij de dichtstbijzijnde trainingsrij?
3. Als baseline wordt dezelfde afstand gemeten voor de holdout-set.
4. Liggen synthetische rijen even ver van de trainingsdata als de holdout-set, dan gedragen ze zich als onbekende data — laag risico.

De **DCR-ratio** vat dit samen:

| DCR-ratio | Risico |
|---|---|
| `> 0.9` | Laag — synthetisch gedraagt zich als onbekende data |
| `0.7 – 0.9` | Matig — beoordeel quasi-identifiers vóór publicatie |
| `< 0.7` | Hoog — synthetisch zit te dicht op de trainingsdata |

De **NNDR** (Nearest Neighbor Distance Ratio) staat ernaast als aanvullend signaal: de afstand van een synthetische rij tot de dichtstbijzijnde échte rij, gedeeld door de afstand tot de op-één-na dichtstbijzijnde. Hoger is beter; een lage waarde betekent dat een synthetische rij vlak op één echte rij zit terwijl de rest verder weg is. Het risico-oordeel leunt op de DCR-ratio; NNDR dient ter controle.

### Welke kolommen tellen mee?

Zowel numerieke als categorische kolommen worden meegenomen. In onderwijsdata zijn de quasi-identifiers (`geslacht`, `instellingscode`, `opleidingscode`, `woonplaats`) juist categorisch — die mogen niet ontbreken in de berekening. Numerieke kolommen worden geschaald, categorische worden one-hot gecodeerd.

Kolommen met heel veel unieke waarden (vrije tekst, namen, identifiers) worden uitgesloten en de app waarschuwt erover — beoordeel die handmatig. De primary key telt nooit mee; die is vervangen door nieuwe anonieme ID's.

!!! warning "Geen formele privacygarantie"
    DCR/NNDR is een statistische schatting, geen formele garantie. Een DPIA blijft vereist vóór publicatie. Gebruik de synthetische data niet als privacyveilig zonder aanvullende beoordeling.

## Zelf valideren met SDV

```python
from sdmetrics.reports.single_table import QualityReport

report = QualityReport()
report.generate(real_df, synth_df, metadata.to_dict())
report.get_visualization(property_name="Column Shapes")
```
