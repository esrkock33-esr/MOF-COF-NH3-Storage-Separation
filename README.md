# Large-Scale Screening of MOFs and COFs for NH₃ Storage and Separation

This repository contains the data and code for the following study:

> Esra Kocak, Gokhan Onder Aksu, Seda Keskin. *Large-Scale Screening of Metal-Organic Frameworks and Covalent Organic Frameworks for NH₃ Storage and Separation.* Manuscript submitted for publication (2026).

Department of Chemical and Biological Engineering, Koç University, Istanbul, Turkey.

---

## Summary of the study

We screened more than 28,000 synthesized and hypothetical metal-organic frameworks (MOFs) and covalent organic frameworks (COFs) for NH₃ storage and NH₃/N₂/H₂ separation. We used molecular simulations together with machine learning (ML). Four material families were studied:

| Material family | Source database | Database size | Single-component NH₃ simulations | NH₃/N₂/H₂ mixture simulations |
|---|---|---:|---:|---:|
| CoRE MOFs | CoRE MOF DB (Zhao et al., 2025) | 43,439 | 1,860 | 1,619 |
| hypoMOFs | Wilmer et al. (2012) | 137,953 | 5,253 | 5,293 |
| CoRE COFs | CoRE COF DB (Tong et al., 2017) | 1,242 | 1,086 | 1,072 |
| hypoCOFs | Mercado et al. (2018) | 69,840 | 19,886 | 19,886 |

NH₃ uptakes and APSs of the remaining 48,496 hypoCOFs were predicted with the hypoCOF ML models, so 68,382 hypoCOFs were covered in total.

### Workflow

1. **Database curation.** Duplicates and MOFs with open metal sites were removed. Pore limiting diameter (PLD), largest cavity diameter (LCD), accessible surface area (S<sub>acc</sub>), porosity, and pore volume (PV) were computed with Zeo++ 0.3. Materials with zero S<sub>acc</sub>, or with PLDs smaller than the kinetic diameter of NH₃ (2.6 Å, single-component) or N₂ (3.64 Å, mixture) were excluded. CoRE MOFs and hypoMOFs had to pass MOSAEC, MOFChecker, MOFClassifier (CLscore > 0.6), and the Chen-Manz method. Materials with Q⁰<sub>st,NH3</sub> > 60 kJ/mol were removed. A representative hypoCOF subset was selected by PLD and PV ranking.
2. **Molecular simulations.** Henry's constants and isosteric heats of adsorption at infinite dilution of NH₃, N₂, and H₂ were computed with Widom insertion. GCMC simulations (RASPA 2.0.50) gave single-component NH₃ uptakes at 1 bar, 298 K and NH₃/N₂/H₂ (0.01/0.495/0.495) mixture uptakes at 1 and 0.1 bar, 298 K. The mixture results were used for a vacuum swing adsorption (VSA) process with 1 bar adsorption and 0.1 bar desorption pressures.
3. **Machine learning.** For each material family, two models were trained: one for single-component NH₃ uptake and one for log APS. Pipelines were selected with TPOT (scikit-learn and XGBoost regressors) using the settings in `TPOT Input Parameters.csv`. Descriptors were structural (PLD, LCD, S<sub>acc</sub>, PV, porosity), chemical (atomic percentages), and energetic (K<sub>H,NH3</sub> for uptake models; K<sub>H,NH3</sub>/K<sub>H,N2</sub> and K<sub>H,NH3</sub>/K<sub>H,H2</sub> for APS models). SHAP analysis was used to find the most important descriptors.
4. **Transferability and unseen predictions.** Uptake models were applied to other material families without retraining. The hypoCOF models were used to predict the 48,496 hypoCOFs that were not simulated.
5. **Top materials.** The top materials (R% > 80%, highest APS) were examined for pore features, functional groups, stability (MOFSimplify), and synthesizability (SAscore, SCScore, linkage chemistry).

### Main findings

- hypoCOFs reached the highest simulated NH₃ uptake (53.1 mol/kg at 1 bar, 298 K). The highest uptakes in all four families were found in materials with pores of about 10 Å.
- NH₃ uptakes of MOFs depend on structural, chemical, and energetic descriptors together. NH₃ uptakes of COFs are mainly governed by NH₃ affinity (K<sub>H,NH3</sub>).
- For NH₃/N₂/H₂ separation, CoRE MOFs and hypoCOFs reached the highest APSs (8,928 and 6,159 mol/kg) with R% > 80%.
- The ratios of Henry's constants (K<sub>H,NH3</sub>/K<sub>H,H2</sub> and K<sub>H,NH3</sub>/K<sub>H,N2</sub>) are the leading descriptors in all APS models.
- Top CoRE MOFs have narrow pores (PLD 3.9-4.1 Å) with ring N atoms and amide groups. Top hypoCOFs have larger pores with phenolic O-H and carboxyl groups.
- Uptake models transfer within MOFs and within COFs, but not across the MOF-COF boundary.

---

## Repository structure

```
MOF-COF-NH3-Storage-Separation/
├── Data/
│   ├── NH3_Uptake/                         # single-component NH3 simulation results
│   │   ├── CoREMOF_NH3_Uptake.xlsx
│   │   ├── hypoMOF_NH3_Uptake.xlsx
│   │   ├── CoRECOF_NH3_Uptake.xlsx
│   │   └── hypoCOF_NH3_Uptake.xlsx
│   └── NH3_N2_H2_Mixture_logAPS/           # NH3/N2/H2 mixture simulation results
│       ├── CoREMOF_Mixture_logAPS.xlsx
│       ├── hypoMOF_Mixture_logAPS.xlsx
│       ├── CoRECOF_Mixture_logAPS.xlsx
│       └── hypoCOF_Mixture_logAPS.xlsx
├── Models-Shap/
│   ├── UptakeModels-Shap/                  # NH3 uptake models (Figure 4, Table S4)
│   │   ├── CoREMOF/
│   │   ├── hypoMOF/
│   │   ├── CoRECOF/
│   │   └── hypoCOF/
│   └── logAPSModel-Shap/                   # log APS models (Figure 7, Table S5)
│       ├── CoREMOF/
│       ├── hypoMOF/
│       ├── CoRECOF/
│       └── hypoCOF/
├── Unseen/
│   ├── unseen uptake/                      # transferability of uptake models (Figure S10)
│   │   ├── (a) CoREMOF-to-CoRECOF/
│   │   ├── (b)hypoMOF-to-hypoCOF/
│   │   ├── (c) CoRE MOF-to-hypoMOF/
│   │   ├── (d) CoRE COF-to-hypoCOF/
│   │   └── plot_Unseen_unseen uptake.py
│   └── SI_hypoCOF_unseen/                  # predictions for 48,496 unsimulated hypoCOFs (Figures S12-S13)
├── TPOT Input Parameters.csv              # TPOT settings used for all ML models
├── LICENSE
└── README.md
```

### Where to find each figure and table

| Paper / SI item | Location |
|---|---|
| Simulated data behind Figures 2, 3, 5, 6 and Tables S6-S13 | `Data/` |
| Figure 4 (uptake parity plots and SHAP), Table S4 | `Models-Shap/UptakeModels-Shap/` |
| Figure 7 (log APS parity plots and SHAP), Table S5 | `Models-Shap/logAPSModel-Shap/` |
| Figure S10 (transferability of uptake models) | `Unseen/unseen uptake/` |
| Figures S12 and S13 (ML predictions for unsimulated hypoCOFs) | `Unseen/SI_hypoCOF_unseen/` |

MOFSimplify predictions (Tables S10-S11), t-SNE maps (Figure S11), and the top-material and synthesizability analyses (Figures 8-9, S14-S17) are not part of this repository.

---

## Folder details

### `Data/`

Full simulation results for every material, one Excel file per material family. The first column is the material name as given in the source database (`FileName` for CoRE MOFs and hypoMOFs, `ID` for CoRE COFs, `Name` or `FileName` for hypoCOFs).

- **`NH3_Uptake/`**: structural properties, K<sub>H,NH3</sub>, Q⁰<sub>st,NH3</sub>, atomic percentages, and the simulated single-component NH₃ uptake at 1 bar, 298 K.
- **`NH3_N2_H2_Mixture_logAPS/`**: mixture uptakes at 1 and 0.1 bar, NH₃ selectivity, working capacity, APS, R%, Henry's constants and isosteric heats of NH₃, N₂, and H₂, Henry's constant ratios, atomic percentages, and log APS.

| File | Rows |
|---|---:|
| `CoREMOF_NH3_Uptake.xlsx` / `CoREMOF_Mixture_logAPS.xlsx` | 1,860 / 1,619 |
| `hypoMOF_NH3_Uptake.xlsx` / `hypoMOF_Mixture_logAPS.xlsx` | 5,253 / 5,293 |
| `CoRECOF_NH3_Uptake.xlsx` / `CoRECOF_Mixture_logAPS.xlsx` | 1,086 / 1,072 |
| `hypoCOF_NH3_Uptake.xlsx` / `hypoCOF_Mixture_logAPS.xlsx` | 19,886 / 19,886 |

### `Models-Shap/`

One subfolder per material family under `UptakeModels-Shap/` (target: `NH3 - 1 bar (mol/kg)`) and `logAPSModel-Shap/` (target: `logAPS`). Each subfolder contains:

| File pattern | Content |
|---|---|
| `<family>_Uptake.xlsx`, `<family>_Mixture_logAPS.xlsx` | ML input table: descriptors and the target column only (no material names) |
| `TPOT_Pipeline-Gen10-Pop50-Random42-Cv5-*.py` | Best pipeline exported by TPOT (settings in `TPOT Input Parameters.csv`). The script refits it on an 80/20 train/test split (random_state = 42), prints R², RMSE, MAE, and SRCC, writes the simulated-vs-predicted table, and draws the parity plot. The log APS scripts draw parity plots for both log APS and APS. |
| `*_Simulated_vs_Predicted.xlsx` | Output of the TPOT script: descriptors, `Set` (Train/Test), simulated value, and ML-predicted value |
| `shapanalysis*.py` | Refits the same pipeline and draws the SHAP summary plot for the test set |
| `*_SHAP_importance.csv` | Mean absolute SHAP value of each descriptor (`Rank`, `Feature`, `mean_abs_SHAP`, `share_%`, `cumulative_%`). Descriptors were added to the SHAP panels of Figures 4 and 7 until they covered at least 90% of the total. |

Selected models and test-set accuracy:

| Family | NH₃ uptake model (final estimator) | Test R² | log APS model (final estimator) | Test R² |
|---|---|---:|---|---:|
| CoRE MOF | Random Forest | 0.922 | XGBoost | 0.985-1 |
| hypoMOF | Random Forest | 0.893 | Random Forest | 0.985-1 |
| CoRE COF | Random Forest | 0.652 | XGBoost | 0.985-1 |
| hypoCOF | XGBoost | 0.856 | XGBoost | 0.985-1 |

The full pipelines with stacked estimators and hyperparameters are given in Tables S4 and S5 and in the scripts.

### `Unseen/`

**`unseen uptake/`**: tests whether an uptake model trained on one family can predict another family without retraining.

| Subfolder | Model (training data) | Predicted family |
|---|---|---|
| `(a) CoREMOF-to-CoRECOF/` | CoRE MOF | CoRE COF |
| `(b)hypoMOF-to-hypoCOF/` | hypoMOF | hypoCOF |
| `(c) CoRE MOF-to-hypoMOF/` | CoRE MOF | hypoMOF |
| `(d) CoRE COF-to-hypoCOF/` | CoRE COF | hypoCOF |

Each subfolder has the training table of the model, the simulated uptakes of the predicted family (`*_SimulatedUptake.xlsx`), the TPOT script, and two output tables: `*_Uptake_Simulated_vs_Predicted.xlsx` (the model on its own data) and `*_to_*_unseen_Simulated_vs_Predicted.xlsx` (simulated uptake `Simulated_NH3_Uptake_1bar` and prediction `Unseen_Result` for the other family). `plot_Unseen_unseen uptake.py` reads the four unseen tables and draws the panels of Figure S10.

**`SI_hypoCOF_unseen/`**: predictions for the 48,496 hypoCOFs that were not simulated.

| File | Content |
|---|---|
| `HypoCOF_Uptake.xlsx`, `HypoCOF_Mixture.xlsx` | Training tables of the hypoCOF uptake and log APS models (19,886 simulated hypoCOFs) |
| `Remaining_48496_Single_features.xlsx`, `Remaining_48496_Mixture_features.xlsx` | Descriptors of the unsimulated hypoCOFs (`Structure` column gives the name) |
| `hypoCOF_remaining_predict.py` | Refits the hypoCOF uptake and log APS pipelines on the 80% training split and predicts the unsimulated hypoCOFs. TPOT is not rerun. |
| `Remaining_48496_Single_prediction_NEW.xlsx` | Output: descriptors and `ML_Predicted_NH3_Uptake` (mol/kg) |
| `Remaining_48496_Mixture_prediction_NEW.xlsx` | Output: descriptors, `ML_Predicted_logAPS`, and `ML_Predicted_APS` (mol/kg) |
| `plot_SI_S12_S13_hypoCOF.py` | Draws simulated (gray) and ML-predicted (green) uptakes and APSs against PLD and PV (Figures S12 and S13). Output files: `Figure_S12_hypoCOF_uptake_PLD_PV` and `Figure_S13_hypoCOF_APS_PLD_PV` (PNG, PDF, TIFF). |

### `TPOT Input Parameters.csv`

TPOT settings used for all NH₃ uptake and log APS models (semicolon-separated):

| Parameter | Value |
|---|---|
| Generations | 10 |
| Population size | 50 |
| Cross-validation folds (cv) | 5 |
| Verbosity | 2 |
| Random state | 42 |
| Training set | 80% |
| Test set | 20% |

The same values appear in the script names (`Gen10-Pop50-Random42-Cv5`).

---

## Column definitions

| Column | Description | Unit |
|---|---|---|
| `PLD (Å)` | Pore limiting diameter | Å |
| `LCD (Å)` | Largest cavity diameter | Å |
| `Sacc (m2/g)` | Accessible surface area (probe radius 1.82 Å) | m²/g |
| `Porosity` | Porosity (void fraction) | - |
| `Pore Volume (cm3/g)` | Pore volume | cm³/g |
| `KNH3`, `KN2`, `KH2 (mol/kg/Pa)` | Henry's constants at infinite dilution, 298 K | mol/kg/Pa |
| `Qst-NH3`, `Qst-N2`, `Qst-H2 (kJ/mol)` | Isosteric heats of adsorption at infinite dilution | kJ/mol |
| `KNH3/KN2`, `KNH3/KH2` | Ratios of Henry's constants | - |
| `%C`, `%H`, `%N`, `%O`, `%Halogen`, `%Metal`, `%Metalloid` | Atomic percentages | % |
| `%Ametal` | Nonmetal atomic percentage (`%Nonmetal` in Table S2) | % |
| `NH3 - 1 bar (mol/kg)` | In `NH3_Uptake` files: simulated single-component NH₃ uptake at 1 bar, 298 K. In mixture files: NH₃ uptake from the mixture at 1 bar | mol/kg |
| `N2 - 1 bar`, `H2 - 1 bar (mol/kg)` | N₂ and H₂ uptakes from the mixture at 1 bar | mol/kg |
| `NH3 - 0.1 bar (mol/kg)` | NH₃ uptake from the mixture at 0.1 bar | mol/kg |
| `Sads` | NH₃/(N₂+H₂) mixture selectivity at 1 bar | - |
| `∆N (mol/kg)` | NH₃ working capacity between 1 and 0.1 bar | mol/kg |
| `APS (mol/kg)` | Adsorbent performance score | mol/kg |
| `R%` | Percent regenerability | % |
| `logAPS` | log₁₀(APS), the target of the APS models | - |

Material families without a given element type do not have that column (for example, COF files have no `%Metal` column).

Performance metrics for the VSA process (adsorption at 1 bar, desorption at 0.1 bar):

- S<sub>NH3</sub> = (x<sub>NH3</sub> / (x<sub>N2</sub> + x<sub>H2</sub>)) / (y<sub>NH3</sub> / (y<sub>N2</sub> + y<sub>H2</sub>)), where x and y are mole fractions in the adsorbed and bulk phases
- ΔN<sub>NH3</sub> = N<sub>NH3,ads</sub> − N<sub>NH3,des</sub>
- APS = S<sub>NH3</sub> × ΔN<sub>NH3</sub>
- R% = ΔN<sub>NH3</sub> / N<sub>NH3,ads</sub> × 100

---

## Requirements

- Python 3
- pandas, numpy, scipy, openpyxl
- scikit-learn, xgboost
- TPOT 0.12.x (the scripts import `tpot.builtins` and `tpot.export_utils`)
- shap
- matplotlib, seaborn, Pillow

```bash
pip install pandas numpy scipy openpyxl scikit-learn xgboost "tpot<1.0" shap matplotlib seaborn pillow
```

The figure scripts in `Unseen/` use the Times New Roman font and stop if it is not installed. Another font can be set with the `TEST_FONT` environment variable.

## How to run

All scripts use absolute Windows paths (`C:/Users/.../MOF-COF-NH3-Storage-Separation/...`). Change these paths to your local copy before running.

1. **NH₃ uptake models (Figure 4).** In each `Models-Shap/UptakeModels-Shap/<family>/` folder, run the `TPOT_Pipeline-*.py` script, then the `shapanalysis*.py` script.
2. **log APS models (Figure 7).** Do the same in each `Models-Shap/logAPSModel-Shap/<family>/` folder.
3. **Transferability (Figure S10).** Run the `TPOT_Pipeline-*.py` script in each of the four `Unseen/unseen uptake/` subfolders, then run `plot_Unseen_unseen uptake.py`.
4. **Unsimulated hypoCOFs (Figures S12-S13).** In `Unseen/SI_hypoCOF_unseen/`, run `hypoCOF_remaining_predict.py` once, then `plot_SI_S12_S13_hypoCOF.py`.

---

## Notes

- Crystal structure (CIF) files and simulation input files are not included. The structures can be obtained from the original databases listed above.
- Some files appear in more than one folder so that each folder can be run on its own.

## Citation

If you use these data or scripts, please cite:

> E. Kocak, G. O. Aksu, S. Keskin. Large-Scale Screening of Metal-Organic Frameworks and Covalent Organic Frameworks for NH₃ Storage and Separation. Manuscript submitted for publication, 2026.

The full reference will be updated after publication.

Corresponding author: Seda Keskin (skeskin@ku.edu.tr)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
