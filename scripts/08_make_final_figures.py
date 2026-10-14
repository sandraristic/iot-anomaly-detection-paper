
from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / 'validated_results'

parser = argparse.ArgumentParser(
    description='Plot saved references or newly computed outputs.'
)
parser.add_argument(
    '--source',
    choices=['reference', 'rerun'],
    default='reference'
)
args = parser.parse_args()

CORE = (
    RESULTS / 'core_results'
    if args.source == 'reference'
    else BASE / 'data' / 'robustness_checks'
)

EXTERNAL = (
    RESULTS / 'expanded_validation'
    if args.source == 'reference'
    else BASE / 'reproduced_outputs'
)

OUT = BASE / (
    'figures' if args.source == 'reference' else 'figures_rerun'
)
OUT.mkdir(parents=True, exist_ok=True)


def save_figure(filename, dpi=600):
    """Save the current figure in PNG, JPG and TIFF formats."""
    fig = plt.gcf()

    for extension in ('png', 'jpg', 'tiff'):
        output_path = OUT / f'{filename}.{extension}'

        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches='tight',
            facecolor='white'
        )

        print(f'Saved: {output_path}')

    plt.close(fig)


# --------------------------------------------------
# Figure 1
# --------------------------------------------------

audit = pd.read_csv(
    CORE / 'duplicate_leakage_audit.csv'
)

ax = audit.set_index('Dataset')[
    'Mean random-split test overlap pct'
].plot(
    kind='bar',
    figsize=(5.8, 3.8)
)

ax.set_ylabel(
    'Test rows with exact feature match in training (%)'
)
ax.set_ylim(0, 100)
ax.set_title(
    'Exact Feature-Vector Overlap in Conventional Random Splits'
)

plt.xticks(rotation=0)
plt.tight_layout()

save_figure('figure1_overlap')


# --------------------------------------------------
# Figure 2
# --------------------------------------------------

summary = pd.read_csv(
    CORE / 'five_seed_duplicate_aware_summary.csv'
)

summary['Dataset'] = summary['Dataset'].replace({
    'iot': 'IoT-23',
    'ton': 'ToN-IoT',
})

model_order = [
    'Linear SVM',
    'Logistic Regression',
    'Random Forest',
    'XGBoost'
]

summary['Model'] = pd.Categorical(
    summary['Model'],
    categories=model_order,
    ordered=True
)

summary = summary.sort_values('Model')

piv = summary.pivot(
    index='Model',
    columns='Dataset',
    values='F1_mean'
).reindex(model_order)

err = summary.pivot(
    index='Model',
    columns='Dataset',
    values='F1_std'
).reindex(model_order)

ax = piv.plot(
    kind='bar',
    yerr=err,
    figsize=(7.6, 4.3),
    capsize=3
)

ax.set_ylim(0.85, 1.005)
ax.set_ylabel('F1-score')
ax.set_xlabel('Model')
ax.set_title(
    'Duplicate-Aware Within-Dataset F1 (mean ± SD, 5 seeds)'
)

plt.xticks(
    rotation=15,
    ha='right'
)
plt.tight_layout()

save_figure('figure2_within_f1')


# --------------------------------------------------
# Figure 3
# --------------------------------------------------

cross = pd.read_csv(
    CORE / 'cross_dataset_duplicate_aware.csv'
)

cross['Model'] = pd.Categorical(
    cross['Model'],
    categories=model_order,
    ordered=True
)

cross = cross.sort_values('Model')

ax = cross.pivot(
    index='Model',
    columns='Direction',
    values='F1'
).reindex(model_order).plot(
    kind='bar',
    figsize=(7.6, 4.3)
)

ax.set_ylim(0, 1.0)
ax.set_ylabel('F1-score')
ax.set_xlabel('Model')
ax.set_title(
    'Cross-Dataset F1 on Duplicate-Aware Data'
)

plt.xticks(
    rotation=15,
    ha='right'
)
plt.tight_layout()

save_figure('figure3_cross_f1')


# --------------------------------------------------
# Figure 4
# --------------------------------------------------

external = pd.read_csv(
    EXTERNAL / 'external_scenario_validation_full.csv'
)

scenario_order = [
    'Muhstik capture',
    'Hakai capture',
    'Torii capture',
    'Trojan capture'
]

external = external[
    external['External scenario'].isin(scenario_order)
].copy()

external['External scenario'] = pd.Categorical(
    external['External scenario'],
    categories=scenario_order,
    ordered=True
)

model_plot_order = [
    'Linear SVM',
    'Logistic Regression',
    'Random Forest',
    'XGBoost'
]

external['Model'] = pd.Categorical(
    external['Model'],
    categories=model_plot_order,
    ordered=True
)

ax = external.pivot(
    index='External scenario',
    columns='Model',
    values='Recall'
).reindex(scenario_order)[model_plot_order].plot(
    kind='bar',
    figsize=(7.8, 4.5)
)

ax.set_ylim(0, 1.05)
ax.set_ylabel('Malicious recall')
ax.set_xlabel('External scenario')
ax.set_title(
    'External IoT-23 Scenario Validation'
)

plt.xticks(
    rotation=15,
    ha='right'
)
plt.tight_layout()

save_figure('figure4_external_scenario_recall')


# --------------------------------------------------
# Figure 5
# --------------------------------------------------

imp = pd.read_csv(
    EXTERNAL / 'external_permutation_importance_linear_svm.csv'
)

show_features = [
    'conn_state',
    'proto',
    'service'
]

imp = imp[
    imp['Feature'].isin(show_features)
].copy()

scenario_plot_order = [
    'Hakai capture',
    'Muhstik capture'
]

imp['Scenario'] = pd.Categorical(
    imp['Scenario'],
    categories=scenario_plot_order,
    ordered=True
)

ax = imp.pivot(
    index='Feature',
    columns='Scenario',
    values='F1 drop'
).reindex(show_features)[scenario_plot_order].plot(
    kind='bar',
    figsize=(7.4, 4.3)
)

ax.set_ylabel(
    'Mean F1 decrease after permutation'
)
ax.set_xlabel('Feature')
ax.set_title(
    'Linear SVM Feature Sensitivity on External Captures'
)

plt.xticks(rotation=0)
plt.tight_layout()

save_figure('figure5_external_feature_sensitivity')


print()
print('All five figures generated successfully.')
print(f'Output directory: {OUT}')
print('Formats: PNG, JPG, TIFF')
print('Resolution: 600 dpi')
