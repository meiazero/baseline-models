# baseline-models

Reproducible training of the three baseline models used in the SUPPLEO /
Entropy-Guided Router paper, trained from scratch on the AllClear Brazil
biome subset (Amazônia, Cerrado, Pantanal).

## Models

| Model       | Paper                          | Repo                         |
|-------------|--------------------------------|------------------------------|
| UnCRtainTS  | Ebel et al., CVPRW 2023        | `UnCRtainTS/`                |
| CTGAN       | Huang & Wu, ICIP 2022          | `CTGAN/`                     |
| U-TILISE    | Stucker et al., TGRS 2023      | `U-TILISE/`                  |

Each model is kept as-is (upstream clone). Training uses dataset adapters
that feed the AllClear Brazil subset in the native format of each model.

## Layout

```
baseline-models/
├── adapters/        dataset adapters: AllClearDataset -> per-model format
├── configs/         training configs (YAML) per model
├── train/           thin training entrypoints per model
├── scripts/         SLURM sbatch scripts (gpuq, 1x A100)
├── results/         trained weights, logs, metrics
└── {CTGAN,UnCRtainTS,U-TILISE}/   upstream model repos (untouched)
```

## Dataset

Training uses the Brazil biome subset of AllClear:

- Train: `train_tx3_s2-s1_100pct_ama-cer-pan.json` (8,048 samples, 953 ROIs)
- Val:   `val_tx3_s2-s1-landsat_100pct_ama-cer-pan.json` (landsat channels ignored)
- Test:  `test_tx3_s2-s1_100pct_ama-cer-pan.json` (1,454 samples, 165 ROIs)

JSONs live in `../allclear/metadata/datasets/`. Image tiles are referenced
by absolute paths inside the JSONs.

## Reproduction

Submit all three training jobs on Apollo (CENAPAD-UFC):

```bash
cd scripts
./submit_all.sh
```

Each job runs on `gpuq` with one A100, writing checkpoints and logs to
`../results/<model>/`.

After training, evaluate the trained checkpoints on the Brazil test split
using `allclear/allclear/benchmark.py` and fill the reproduced numbers into
`../draft_v2-LNLM/tables/table_baselines.tex`.
