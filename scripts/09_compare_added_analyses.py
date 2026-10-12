from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
checks = [
    ('duplicate audit', BASE/'validated_results/core_results/duplicate_leakage_audit.csv',
     BASE/'data/robustness_checks/duplicate_leakage_audit.csv', ['Dataset']),
    ('external permutation importance', BASE/'validated_results/expanded_validation/external_permutation_importance_linear_svm.csv',
     BASE/'reproduced_outputs/external_permutation_importance_linear_svm.csv', ['Scenario', 'Feature']),
]
failed = False
for name, reference, rerun, keys in checks:
    print('\n' + name.upper())
    if not rerun.is_file():
        print('MISSING:', rerun)
        failed = True
        continue
    a = pd.read_csv(reference).set_index(keys).sort_index()
    b = pd.read_csv(rerun).set_index(keys).sort_index()
    if not a.index.equals(b.index) or set(a.columns) != set(b.columns):
        print('FAIL: row keys or columns differ')
        failed = True
        continue
    differences = []
    for col in a.columns:
        same = np.isclose(a[col], b[col], atol=1e-10, rtol=0, equal_nan=True)
        for key in a.index[~same]:
            differences.append({'Row': str(key), 'Metric': col,
                                'Reference': a.loc[key, col], 'Rerun': b.loc[key, col],
                                'Difference': b.loc[key, col] - a.loc[key, col]})
    if differences:
        print(pd.DataFrame(differences).to_string(index=False))
        failed = True
    else:
        print('MATCH (absolute tolerance 1e-10)')
print('\nReferences were not modified.')
raise SystemExit(1 if failed else 0)
