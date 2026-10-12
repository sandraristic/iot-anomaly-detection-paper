# IoT Network Anomaly Detection

This repository contains the Python scripts and results for **Machine Learning-Based Anomaly Detection in IoT Network Traffic: A Comparative Study**.

Author: Sandra Ristić, Faculty of Engineering, University of Kragujevac, Serbia.

The experiments compare Logistic Regression, Linear SVM, Random Forest and XGBoost using 11 common flow features from ToN-IoT and selected IoT-23 captures. Raw third-party datasets are not included.

## Contents

- `scripts`: nine numbered analysis, plotting and comparison scripts.
- `validated_results`: retained manuscript reference tables, including the corrected external and leave-one-capture-out results.
- `figures`: manuscript figures regenerated from the current reference tables; script 08 exports five PNG figures at 600 dpi.
- `requirements.txt`: pinned direct dependencies.

## Environment

The reference environment uses Python **3.13.5**, pandas **2.2.3**, NumPy **2.3.5**, scikit-learn **1.8.0**, XGBoost **3.1.3**, and Matplotlib **3.10.8**. Direct dependencies are pinned; this is not a complete transitive dependency lock.

Run commands from the repository root. With `uv` installed:

```bash
uv venv --python 3.13.5 .venv-repro
uv pip install --python .venv-repro/bin/python -r requirements.txt
```

If `.venv-repro` already contains this environment, reuse it. The commands below invoke its Python directly; activation is unnecessary. Alternatively, create a standard virtual environment with an installed Python 3.13.5 and install `requirements.txt` using pip.

## Input data

Create `data` in the repository root and put the following files directly inside it. Paths are resolved relative to the scripts, so `/mnt/data` and `IOT_RAW_DIR` are not used.

| Local filename | Dataset or capture |
| --- | --- |
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

Obtain the data from the sources cited in manuscript references [1], [2] and [7]. These filenames identify the local exports used in the study. The parsers expect those tab-separated Zeek log layouts, including their label formatting. The Somfy export has no label columns and is explicitly treated as benign. Do not substitute a different dataset version or reformat the logs without checking parsing and record counts.

The `data` directory is excluded from Git. Derived pickles and local rerun outputs inside it are also excluded.

## Run the experiments in order

### 1. Prepare the core datasets

```bash
.venv-repro/bin/python scripts/01_prepare_robust_data.py
```

Creates `data` with the cleaned ToN-IoT and IoT-23 feature tables. Expected retained counts for the study inputs are 92,439 ToN-IoT patterns and 6,146 core IoT-23 patterns.

### 2. Run all five seeds on both datasets

```bash
for dataset in ton iot; do
  for seed in 11 23 42 77 101; do
    .venv-repro/bin/python scripts/02_run_one_seed.py "$dataset" "$seed"
  done
done
.venv-repro/bin/python scripts/03_collect_five_seed_results.py
```

Outputs the ten seed CSVs, the combined results and the mean/sample-standard-deviation summary under `data`.

### 3. Run cross-dataset and held-out attack evaluation

```bash
.venv-repro/bin/python scripts/04_run_cross_unseen.py
```

Writes cross-dataset results, attack pattern counts and held-out attack results under `data`.

### 4. Run the capture-level experiments

```bash
.venv-repro/bin/python scripts/05_expanded_scenario_validation.py
.venv-repro/bin/python scripts/06_leave_one_capture_out.py
.venv-repro/bin/python scripts/07_torii_pair_validation.py
```

| Script | Experiment | Local output directory |
| --- | --- | --- |
| 05 | Core-to-external validation, attack-detail recall and overlap audit | `reproduced_outputs` |
| 06 | Nine-capture leave-one-capture-out validation | `data` |
| 07 | Paired Torii transfer, overlap audit, errors and permutation importance | `data` |

Script 01 also writes `duplicate_leakage_audit.csv` and `duplicate_overlap_per_seed.csv` using the original rows and five stratified 80/20 splits. Script 05 now writes four CSVs, including external Linear SVM permutation importance (five repeats, F1 scoring, seed 42). Both added analyses were run on the study inputs. The duplicate audit matched the historical table. The external importance reference uses the latest author-run output.

## Regenerate Figures 1–5

```bash
.venv-repro/bin/python scripts/08_make_final_figures.py
```

No raw data or model training is needed for this step. Script 08 reads the **saved tables in `validated_results`**, not the local rerun directories, and overwrites five PNG figures at 600 dpi.

To plot newly computed results without replacing the references:

```bash
.venv-repro/bin/python scripts/08_make_final_figures.py --source rerun
```

This reads the local output directories and writes to `figures_rerun/`. Run scripts 01–05 first; plotting does not silently fall back to retained CSVs.

To compare the two newly implemented analyses with the retained tables:

```bash
.venv-repro/bin/python scripts/09_compare_added_analyses.py
```

Differences are printed per metric and row; a nonzero exit status means differences or missing outputs. Reference files remain unchanged. Investigate differences before updating the manuscript or reference tables.

## Metrics and interpretation

Malicious traffic is the positive class. `PR_AUC` and `PR-AUC` both denote average precision, calculated with `average_precision_score`; Linear SVM uses decision scores. FPR is FP / (FP + TN). Malicious recall is undefined for benign-only captures and is stored as an empty/NaN value where applicable.

In the leave-one-capture-out summary, the best model is selected by F1-score, with recall as the tie-breaker. `Best recall` is the selected model's recall. For benign-only captures the summary reports the maximum FPR across models. Torii and Trojan malicious sample sizes are very small and do not support broad generalization claims.

## Verification and reference results

The classification experiments were run by the author on the study inputs. The added duplicate audit matched the historical reference at absolute tolerance 1e-10. External classification metrics, attack-detail recall and overlap counts also matched the retained tables. Hakai permutation importance matched; the three nonzero Muhstik importance values differed from the historical table. The current importance CSV and Figure 5 use the newly calculated values from five permutation repeats with F1 scoring and random_state=42. The feature ranking and qualitative interpretation are unchanged. The precise cause of the historical importance difference was not established because the original generator was unavailable.

Scripts 05 and 06 recalculate hashes after concatenating capture features, ensuring consistent column dtypes when identifying cross-capture overlap. The corrected nine-capture results supersede the earlier leave-one-capture-out table. The updated summary includes core-malware recall 0.597747, Muhstik selected-model recall 0.641431, Philips Hue maximum FPR 0.078431, and Torii 21-1 selected-model recall 0.777778.

The results were verified using the author's supplied output files; raw datasets are not redistributed in this repository. The reference environment is stated above. Earlier external-classification outputs also matched when run with Python 3.10.12, pandas 2.3.3, NumPy 2.2.6, scikit-learn 1.7.2 and XGBoost 3.2.0; this does not guarantee identical outputs across other environments. Any retained environment.txt is a historical record rather than an environment manifest for every reference file.

Runtime measurements depend on hardware and system load. The duplicate audit uses sample standard deviation across seeds (`ddof=1`); permutation-importance variability uses the standard deviation returned by scikit-learn. Preserve the original input row order because it affects seeded splits and permutations. Script 09 compares only the duplicate audit and external permutation importance, rather than validating every experiment.

## Code and data availability

Raw third-party files are not redistributed. Cite the manuscript and the original dataset sources when using this work. 
