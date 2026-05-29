# Methodologie

edusynth gebruikt [SDV (Synthetic Data Vault)](https://sdv.dev) als statistische engine.

## Huidige aanpak

Voor enkelvoudige tabellen wordt de **Gaussian Copula** gebruikt:

1. Transformeer elke kolom naar een uniforme verdeling via de empirische CDF
2. Schat de Spearman-correlatiematrix tussen kolommen
3. Genereer gecorreleerde steekproeven via de Cholesky-decompositie
4. Inverteer de CDF terug naar de originele schaal

## Aannames

- Kolommen zijn stationair (geen tijdsafhankelijkheid)
- Correlaties zijn lineair genoeg om via Spearman te schatten
- De dataset is representatief voor de populatie

## Wanneer vertrouw je het niet?

- Datasets met sterke niet-lineaire afhankelijkheden
- Kleine datasets (< 500 rijen) — correlaties zijn dan instabiel
- Longitudinale data met studentpaden — gebruik daarvoor een sequentieel model (nog niet geïmplementeerd)
