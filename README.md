# IoT Network Anomaly Detection

This repository contains the Python scripts and experimental results supporting the paper **“Machine Learning-Based Anomaly Detection in IoT Network Traffic: A Comparative Study.”**

**Author:** Sandra Ristić  
**Affiliation:** Faculty of Engineering, University of Kragujevac, Serbia

The study compares Logistic Regression, Linear SVM, Random Forest, and XGBoost for detecting malicious IoT network traffic. The experiments use 11 common flow-level features from the ToN-IoT and IoT-23 datasets.

The evaluation includes duplicate-aware classification, cross-dataset generalization, leave-one-capture-out validation, external scenario testing, and paired Torii cross-capture analysis.

The original third-party datasets are publicly available from their respective sources and are not included in this repository.

## Repository Contents

- `scripts/` — nine numbered Python scripts for data preparation, model evaluation, analysis, and visualization.
- `validated_results/` — retained reference results used in the manuscript, including corrected external validation and leave-one-capture-out results.
- `figures/` — five manuscript figures generated from the retained reference results.
- `requirements.txt` — pinned direct Python dependencies.

Additional sensitivity experiments examining conflicting labels, duplicate training records, and feature removal were conducted separately. Their results are discussed in the manuscript.

## Environment

The reference experimental environment uses:

| Software | Version |
|---|---|
| Python | 3.13.5 |
| pandas | 2.2.3 |
| NumPy | 2.3.5 |
| scikit-learn | 1.8.0 |
| XGBoost | 3.1.3 |
| Matplotlib | 3.10.8 |

The direct dependencies are pinned in `requirements.txt`. This file does not provide a complete lock of all transitive dependencies.

Run the commands below from the repository root.

### Create the Python environment

Using `uv`:

```bash
uv venv --python 3.13.5 .venv-repro
uv pip install --python .venv-repro/bin/python -r requirements.txt
```

If `.venv-repro` already contains the required environment, it can be reused.

The commands below invoke the environment's Python interpreter directly, so activation is not required.

Alternatively, create a standard virtual environment using Python 3.13.5 and install the dependencies with pip.

## Input Data

Create a directory named `data/` in the repository root and place the required original dataset files directly inside it.

The `data/` directory is excluded from Git.

### Required input files

| Local filename | Dataset or capture |
|---|---|
| `train_test_network.csv` | ToN-IoT network data |
| `2018-09-21-11-40-22-192.168.2.3-zeek-conn-log.labeled.txt` | Amazon Echo, core benign |
| `2018-10-25-14-06-32-192.168.1.132-zeek-conn.log.labeled.txt` | Philips Hue, core benign |
| `2018-12-21-15-50-14-192.168.1.195-zeek-conn-log.labeled.txt` | Mirai, core malware |
| `2018-05-19-20-57-19-192.168.2.5-zeek-conn-log.labeled.txt` | Muhstik |
| `2018-07-31-15-15-09-192.168.100.113-zeek-conn-log.txt` | Hakai |
| `2018-10-02-13-12-30-192.168.100.103-zeek-conn-log.txt` | Torii 20-1 |
| `2018-10-03-15-22-32-192.168.100.113-zeek-conn-log.txt` | Torii 21-1 |
| `2019-01-10-14-34-38-192.168.1.197-zeek-conn-log.txt` | Trojan/FileDownload |
| `2019-07-03-16-41-09-192.168.1.158-zeek-conn-log.txt` | Somfy Door Lock, benign |

### Dataset Sources

The original datasets can be obtained from the sources cited in the manuscript:

- **ToN-IoT:** Moustafa [1].
- **IoT-23:** Garcia et al. [2].
- **IoT-23 capture documentation:** Stratosphere Laboratory [17].

Official IoT-23 dataset information is available at:

https://www.stratosphereips.org/datasets-iot23

The filenames above identify the local input files used in the study. The scripts expect the corresponding tab-separated Zeek connection-log format, including the original label structure.

The Somfy Door Lock capture does not contain label columns in the local export and is treated as benign according to the official IoT-23 documentation.

For reproducibility, avoid substituting different dataset versions or modifying file formats without checking the parsers and expected record counts.

## Running the Experiments

The following commands reproduce the main experimental workflow.

### 1. Prepare the core datasets

```bash
.venv-repro/bin/python scripts/01_prepare_robust_data.py
```

This script processes the original ToN-IoT and IoT-23 input files in `data/` and generates the cleaned feature tables used in the core experiments.

Expected retained counts:

| Dataset | Unique retained patterns |
|---|---:|
| ToN-IoT | 92,439 |
| Core IoT-23 | 6,146 |

The preparation stage also generates duplicate-audit outputs, including:

- `duplicate_leakage_audit.csv`
- `duplicate_overlap_per_seed.csv`

The duplicate audit uses the original records and five stratified 80/20 train-test splits.

### 2. Run within-dataset evaluation

The experiments use five random seeds:

`11, 23, 42, 77, 101`

Run the models on both datasets:

```bash
for dataset in ton iot; do
  for seed in 11 23 42 77 101; do
    .venv-repro/bin/python scripts/02_run_one_seed.py "$dataset" "$seed"
  done
done
```

Combine the results:

```bash
.venv-repro/bin/python scripts/03_collect_five_seed_results.py
```

The scripts generate per-seed results, combined results, and summaries reporting mean performance and sample standard deviation.

The main evaluation metrics are accuracy, precision, recall, F1-score, Matthews correlation coefficient (MCC), and PR-AUC.

### 3. Run cross-dataset evaluation

```bash
.venv-repro/bin/python scripts/04_run_cross_unseen.py
```

This script evaluates how models trained on one dataset perform when applied to another dataset using the common 11-feature representation.

The analysis includes bidirectional ToN-IoT and IoT-23 transfer, held-out attack evaluation, and attack-pattern summaries.

Results are saved under `data/`.

### 4. Run external scenario validation

```bash
.venv-repro/bin/python scripts/05_expanded_scenario_validation.py
```

Models are trained using the original three core IoT-23 captures and evaluated on additional independent captures.

The evaluated scenarios include Muhstik, Hakai, Torii, Trojan/FileDownload, and Somfy Door Lock.

The experiment records classification metrics, attack-specific recall, exact feature overlap, and Linear SVM permutation importance.

Results are written to `reproduced_outputs/`, including:

- `external_scenario_validation_full.csv`
- `external_attack_detail_recall.csv`
- `external_overlap_audit.csv`
- `external_permutation_importance_linear_svm.csv`

The feature-importance analysis uses five permutation repeats, F1-score, and `random_state=42`.

### 5. Run leave-one-capture-out validation

```bash
.venv-repro/bin/python scripts/06_leave_one_capture_out.py
```

This experiment evaluates generalization across nine IoT-23 capture scenarios.

Each capture is held out in turn, while the remaining captures are used for training. Matching feature patterns are excluded from training to prevent exact train-test overlap.

The experiment evaluates complete capture scenarios rather than random record-level partitions.

The corrected nine-capture results supersede the earlier leave-one-capture-out reference table.

### 6. Run paired Torii cross-capture validation

```bash
.venv-repro/bin/python scripts/07_torii_pair_validation.py
```

This script evaluates transfer between Torii captures 20-1 and 21-1 in both directions.

The analysis includes:

- Exact cross-capture overlap.
- Duplicate-aware transfer results.
- False-positive and false-negative analysis.
- Attack-specific error analysis.
- Linear SVM permutation importance.

Each held-out capture contains only nine unique malicious patterns, so the results should be interpreted as exploratory.

## Regenerating Figures

The five figures used in the manuscript can be regenerated without repeating model training.

```bash
.venv-repro/bin/python scripts/08_make_final_figures.py
```

The script reads the retained reference tables from `validated_results/` and saves five PNG figures at **600 dpi**.

### Generate figures from newly computed results

```bash
.venv-repro/bin/python scripts/08_make_final_figures.py --source rerun
```

This option reads the locally reproduced results and writes the figures to `figures_rerun/`.

Run the relevant preparation and evaluation scripts before using this option.

The script does not silently substitute retained reference files when rerun outputs are missing.

## Comparing Reproduced Results

Use the comparison script to examine differences between selected newly generated outputs and the retained reference results:

```bash
.venv-repro/bin/python scripts/09_compare_added_analyses.py
```

The script compares the duplicate audit and external permutation-importance results.

Differences are reported by metric and row. Missing outputs or differences produce a nonzero exit status.

The retained reference files are not modified.

This comparison does not validate every experiment in the repository.

## Additional Sensitivity Experiments

Three additional analyses were conducted to examine methodological choices raised during review of the manuscript.

### Conflicting labels

Feature vectors associated with both benign and malicious labels were identified and examined.

A sensitivity experiment compared models trained with conflicting feature-label patterns removed or retained.

The comparison used identical non-conflicting test patterns across five random seeds, with exact feature overlap between training and testing prevented.

### Duplicate training records

Models were compared with and without duplicate training records while keeping the test sets unchanged.

This analysis examined the influence of repeated records during training separately from the effect of train-test overlap.

### Feature ablation

The influence of `conn_state` and `proto` was examined using Muhstik and Hakai external captures.

Four feature configurations were compared:

- All 11 features.
- Without `conn_state`.
- Without `proto`.
- Without both features.

The analysis examined changes in F1-score, precision, recall, false-positive rate, and confusion-matrix counts.

The source files and outputs of these sensitivity experiments should be retained alongside the main reproducibility materials. They are not part of the nine numbered scripts listed above unless separately added to the public repository.

## Metrics and Interpretation

Malicious traffic is treated as the positive class.

| Metric | Interpretation |
|---|---|
| Accuracy | Fraction of correctly classified patterns |
| Precision | Fraction of predicted malicious patterns that are actually malicious |
| Recall | Fraction of malicious patterns correctly detected |
| F1-score | Harmonic mean of precision and recall |
| MCC | Correlation between predicted and true binary labels |
| PR-AUC | Average precision across classification thresholds |
| FPR | False positives divided by all actual benign patterns |

`PR_AUC` and `PR-AUC` both refer to average precision, calculated using `average_precision_score`.

For Linear SVM, decision scores are used where probability estimates are unavailable.

Malicious recall is undefined for benign-only captures and is stored as an empty or NaN value where applicable.

In leave-one-capture-out summaries, the best model is selected by F1-score, with recall used to break ties. The reported best recall belongs to the selected model.

For benign-only captures, the summaries report maximum FPR across models.

Results based on very small malicious samples, especially Torii and Trojan/FileDownload, should not be interpreted as general performance estimates across all IoT attacks.

## Verification and Reproducibility Notes

The classification experiments were run by the author using the study input files.

The duplicate audit matched the historical reference results at an absolute tolerance of `1e-10`.

External classification metrics, attack-detail recall, and overlap counts also matched the retained results.

For external permutation importance, the Hakai results matched the earlier reference. Three nonzero Muhstik importance values differed from the historical table.

The current importance results and Figure 5 use newly calculated values based on five permutation repeats, F1 scoring, and `random_state=42`.

The feature ranking and qualitative interpretation remain unchanged. The precise reason for the historical numerical difference has not been established.

Scripts 05 and 06 recalculate feature hashes after concatenating capture data to maintain consistent column types when checking cross-capture overlap.

The corrected leave-one-capture-out results include:

| Evaluation | Result |
|---|---:|
| Core malware selected-model recall | 0.597747 |
| Muhstik selected-model recall | 0.641431 |
| Philips Hue maximum FPR | 0.078431 |
| Torii 21-1 selected-model recall | 0.777778 |

The reference environment is documented above.

Some external-classification outputs also matched when using Python 3.10.12, pandas 2.3.3, NumPy 2.2.6, scikit-learn 1.7.2, and XGBoost 3.2.0.

This does not guarantee identical results across all environments.

Runtime depends on hardware and system load. Seeded splits and permutations may depend on the original input row order, which should be preserved.

The duplicate audit reports sample standard deviation using `ddof=1`. Permutation-importance variability uses the standard deviation returned by scikit-learn.

Script 09 compares selected analyses only and should not be treated as a complete validation of every experimental result.

## Code and Data Availability

The Python scripts and retained experimental results supporting the manuscript are provided in this repository.

The original ToN-IoT and IoT-23 datasets are publicly available from their official sources and are not redistributed here.

Instructions for obtaining and preparing the required input files are provided above. Researchers wishing to reproduce the experiments should use the specified dataset versions, filenames, and preprocessing steps.

When using this work, please cite the original dataset sources and the accompanying manuscript.
