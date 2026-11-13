# Données

Ce dossier n'est pas versionné (≈ 7 Go). Pour le reconstruire :

1. Télécharger `slicing-simulations.zip` (271 Mo) depuis Zenodo :
   [10.5281/zenodo.10610616](https://doi.org/10.5281/zenodo.10610616)
2. Le placer dans `data/raw/` et le décompresser (≈ 6,6 Go une fois extrait).
3. Lancer `bash code/run_all.sh` : l'extraction produit `data/processed/`
   (`samples`, `flows`, `links`, `queues` en Parquet, puis `graphs.pkl`).

Structure attendue :

```
data/
├── raw/
│   ├── slicing-simulations.zip
│   └── 2.2.1-10kprocessed/        # archives results_slicing_WFQ_*.tar.gz + graphs/
└── processed/                     # généré par code/extraction.py et code/preparation.py
```
