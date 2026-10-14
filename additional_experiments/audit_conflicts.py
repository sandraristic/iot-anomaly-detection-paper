from pathlib import Path
from zipfile import ZipFile
import pandas as pd

ZIP = Path(__file__).resolve().parent.parent / 'data.zip'
OUT = Path(__file__).resolve().parent
FEATURES = ['proto', 'service', 'duration', 'src_bytes', 'dst_bytes', 'conn_state',
            'missed_bytes', 'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes']
with ZipFile(ZIP) as archive:
    df = pd.read_csv(archive.open('data/train_test_network.csv'), low_memory=False)
key = pd.util.hash_pandas_object(df[FEATURES].astype('string').fillna('__MISSING__'), index=False)
frame = df.assign(vector_hash=key.values)
counts = frame.groupby(['vector_hash', 'label']).size().unstack(fill_value=0)
for label in (0, 1):
    if label not in counts.columns:
        counts[label] = 0
conflicts = counts[(counts[0] > 0) & (counts[1] > 0)].copy()
conflicts = conflicts.rename(columns={0: 'benign_rows', 1: 'malicious_rows'})
conflicts['total_rows'] = conflicts['benign_rows'] + conflicts['malicious_rows']
conflicts = conflicts.sort_values('total_rows', ascending=False).reset_index()
conflicts.insert(0, 'conflict_group', range(1, len(conflicts)+1))
conflicts.to_csv(OUT / 'ton_conflict_group_summary.csv', index=False)
print(f"Total rows: {len(df):,}")
print(f"Conflicting groups: {len(conflicts)}")
print(f"Rows in conflicting groups: {int(conflicts.total_rows.sum()):,}")
print(f"Benign rows in conflicting groups: {int(conflicts.benign_rows.sum()):,}")
print(f"Malicious rows in conflicting groups: {int(conflicts.malicious_rows.sum()):,}")
print(f"CSV: {OUT / 'ton_conflict_group_summary.csv'}")
