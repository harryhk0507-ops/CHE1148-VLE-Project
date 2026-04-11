# Predicting Mole Fraction in Binary Vapor–Liquid Systems

**Harry Chin, Chenhe Zhang, Kuangdi Zhu** 
Team Name: Binary VLE Fighters
Target Workshop: Machine Learning and the Physical Sciences  
CHE1148

---

## Overview

In this project, different machine learning models were developed for predicting the **mole fraction of binary vapor–liquid mixtures** from thermodynamic conditions, molecular descriptors, and at least one measured thermophysical property. Data is sourced from the NIST ThermoML Archive and ThermoData Engine (TDE).

The task is formulated as an inverse mapping problem: given temperature, pressure, phase, component identities, and one additional measured property, predict the mole fraction of the first listed component.

Four models are implemented and evaluated:
- Simple linear regression (baseline)
- XGBoost
- Simple MLP
- Mixture-aware multimodal neural network (SMOLE-BERT + RDKit descriptors)

The multimodal model achieves the best performance (R² = 0.8175, MAE = 0.0696, MSE = 0.0174), outperforming the other baselines.

---

## Repository Structure

```
CHE1148-VLE-Project/
│
├── README.md
├── environment.yml                   # Conda environment
├── .gitignore
│
├── data/
│   └── raw/
│       └── data.parquet              # Unprocessed dataset (2,643,425 rows, 70 columns)
│                                     # Intermediate and Processed Data not attached here due to size
│                                     # Run notebooks 01–05 to regenerate Processed Dataset
│
├── models/                           # Saved model weights and hyperparameters
│   ├── linear_baseline.pkl
│   ├── mlp_baseline.pth
│   ├── multimodal.pth
│   └── xgboost_best_params_final.json
│
├── notebooks/                        # Numbered in pipeline order
│   ├── 01_Data_processing_part_1.ipynb
│   ├── 02_Data_processing_part_2.ipynb
│   ├── 03_Add_Rdkit_Smiles_Mols.ipynb
│   ├── 04_Add_Descriptastorus.ipynb
│   ├── 05_dataset_splitting.ipynb
│   ├── 06_Linear_baseline.ipynb
│   ├── 07_XGBoost.ipynb
│   ├── 08_NN_baseline.ipynb
│   ├── 09_Precompute_SMOLE_BERT.ipynb
│   └── 10_multi_modal.ipynb
│
└── report/
    ├── data for plotting figures/    # Training loss history CSVs for the baseline and multimodal MLP models
    │   ├── training_loss_history_baseline.csv
    │   └── training_loss_history_multimodal.csv
    └── figures/                      # Figures used in the final report
        ├── Multimodal_Architecture.pdf
        ├── training_curves_mlp.png
        └── training_curves_multimodal.png
```

---

## Setup

To reproduce this project locally:

```bash
conda env create -f environment.yml
conda activate CHE1148
```

> Note: Some notebooks (especially SMOLE-BERT precomputation in `09` and multimodal training in `10`) are computationally intensive and are best run on a GPU runtime.

---

## Pipeline

Run notebooks in numbered order:

| Step | Notebook | Description |
|------|----------|-------------|
| 1–2 | `01–02_Data_processing` | Filter raw TDE data, quality control, sparsity reduction |
| 3 | `03_Add_Rdkit_Smiles_Mols` | Obtain SMILES strings from PubChemPy, Mol objects from RDKit|
| 4 | `04_Add_Descriptastorus` | Compute molecular descriptors from SMILES strings |
| 5 | `05_dataset_splitting` | Stratified train/val/test split (80/10/10) by mixture rarity |
| 6 | `06_Linear_baseline` | Train and evaluate simple linear regression baseline |
| 7 | `07_XGBoost` | Train and evaluate XGBoost |
| 8 | `08_NN_baseline` | Train and evaluate simple MLP baseline |
| 9 | `09_Precompute_SMOLE_BERT` | Precompute SMOLE-BERT embeddings for mixture components |
| 10 | `10_multi_modal` | Train and evaluate full multimodal model |

---

## Results

All metrics reported on the held-out test set with bootstrapped standard deviations (1000 resamples).

| Model | MSE | MAE | R² | Kendall's τ |
|-------|-----|-----|----|-------------|
| Linear | 10733 ± 7849 | 1.923 ± 1.013 | ~0 | 0.236 ± 0.006 |
| XGBoost | 0.0190 ± 0.0005 | 0.0831 ± 0.0011 | 0.800 ± 0.006 | 0.747 ± 0.004 |
| Simple MLP | 0.0200 ± 0.0005 | 0.0817 ± 0.0011 | 0.790 ± 0.005 | 0.736 ± 0.004 |
| **Multimodal MLP** | **0.0174 ± 0.0005** | **0.0696 ± 0.0011** | **0.818 ± 0.005** | **0.768 ± 0.004** |

---

## Data

Raw data is sourced from:
- [NIST ThermoML Archive](https://trc.nist.gov/ThermoML/)
- [NIST ThermoData Engine (TDE)](https://www.nist.gov/srd/nist-standard-reference-database-103b)

The initial unified dataset contained **2,643,425 rows and 70 columns**. After filtering and preprocessing, the cleaned dataset contains **106,383 rows and 412 features**. CSV files containing interim and processed data are not tracked in this repository due to file size constraints.

---

## Dependencies

Key packages:
- `pytorch`, `scikit-learn`, `xgboost`, `optuna`
- `rdkit`, `descriptastorus`, `pubchempy`
- `pandas`, `numpy`, `matplotlib`

See `environment.yml` for the full list.
