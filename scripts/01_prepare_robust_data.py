from pathlib import Path
import re, numpy as np, pandas as pd
from pandas.util import hash_pandas_object
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parents[1] / 'data'
OUT=BASE/'robustness_checks'
OUT.mkdir(parents=True, exist_ok=True)

FEATURES=['proto','service','duration','src_bytes','dst_bytes','conn_state','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes']
NUM=['duration','src_bytes','dst_bytes','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes']
CAT=['proto','service','conn_state']

T=pd.read_csv(BASE/'train_test_network.csv',low_memory=False)
X=T[FEATURES].copy(); y=T.label.astype(int)
for c in NUM: X[c]=pd.to_numeric(X[c],errors='coerce')
for c in CAT: X[c]=X[c].replace('-',np.nan)

def clean(X,y):
    h=hash_pandas_object(X,index=False).astype('uint64')
    d=X.copy(); d['_y']=y.values; d['_g']=h.values
    n=d.groupby('_g')['_y'].nunique()
    bad=set(n[n>1].index)
    d=d[~d['_g'].isin(bad)].drop_duplicates(subset=FEATURES,keep='first').reset_index(drop=True)
    return d[FEATURES],d['_y'].astype(int),bad

Xd,yd,badT=clean(X,y)
pd.concat([Xd,yd.rename('target')],axis=1).to_pickle(OUT/'ton_unique.pkl')

fields=['ts','uid','id.orig_h','id.orig_p','id.resp_h','id.resp_p','proto','service','duration','orig_bytes','resp_bytes','conn_state','local_orig','local_resp','missed_bytes','history','orig_pkts','orig_ip_bytes','resp_pkts','resp_ip_bytes','tunnel_parents','label','detailed_label']

def parse(p):
    rows=[]
    with open(p,errors='replace') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            parts=line.rstrip('\n').split('\t')
            if len(parts)<21: continue
            ch=re.split(r'\s{2,}',parts[20].strip(),maxsplit=2)
            if len(ch)==3: tunnel,label,detail=ch
            else:
                toks=parts[20].split()
                tunnel=toks[0] if toks else '-'
                label=toks[1] if len(toks)>1 else None
                detail=' '.join(toks[2:]) if len(toks)>2 else '-'
            row=parts[:20]+[tunnel,label,detail]
            if len(row)==23: rows.append(row)
    return pd.DataFrame(rows,columns=fields)

I=pd.concat([parse(p) for p in [
    BASE/'2018-09-21-11-40-22-192.168.2.3-zeek-conn-log.labeled.txt',
    BASE/'2018-12-21-15-50-14-192.168.1.195-zeek-conn-log.labeled.txt',
    BASE/'2018-10-25-14-06-32-192.168.1.132-zeek-conn.log.labeled.txt'
]],ignore_index=True)

rn={'orig_bytes':'src_bytes','resp_bytes':'dst_bytes','orig_pkts':'src_pkts','orig_ip_bytes':'src_ip_bytes','resp_pkts':'dst_pkts','resp_ip_bytes':'dst_ip_bytes'}
Xi=I.rename(columns=rn)[FEATURES].copy()
yi=I.label.astype(str).str.lower().map(lambda x:0 if x=='benign' else 1).astype(int)
for c in NUM: Xi[c]=pd.to_numeric(Xi[c].replace('-',np.nan),errors='coerce')
for c in CAT: Xi[c]=Xi[c].replace('-',np.nan)

Id,yId,badI=clean(Xi,yi)
pd.concat([Id,yId.rename('target')],axis=1).to_pickle(OUT/'iot_unique.pkl')


h=hash_pandas_object(Xi,index=False).astype('uint64')
u=Xi.copy(); u['_y']=yi.values; u['_g']=h.values; u['_detail']=I.detailed_label.replace('-','Benign').values
u=u[~u['_g'].isin(badI)].drop_duplicates(subset=FEATURES,keep='first').reset_index(drop=True)
u.to_pickle(OUT/'iot_unique_detail.pkl')

print('ToN',len(Xd),yd.value_counts().to_dict(),'IoT',len(Id),yId.value_counts().to_dict(),'attack',u._detail.value_counts().to_dict())

def duplicate_audit(features, labels, dataset):
    labels = np.asarray(labels, dtype=int)
    hashes = hash_pandas_object(features, index=False).to_numpy()
    groups = pd.DataFrame({'hash': hashes, 'label': labels})
    conflicts = groups.groupby('hash')['label'].nunique()
    conflict_hashes = conflicts[conflicts > 1].index
    pairs = features.reset_index(drop=True).copy()
    pairs['_label'] = labels
    excess = len(pairs) - len(pairs.drop_duplicates())
    overlaps = []
    split_rows = []
    for seed in [11, 23, 42, 77, 101]:
        train, test = train_test_split(
            np.arange(len(features)), test_size=0.2,
            stratify=labels, random_state=seed)
        matched = np.isin(hashes[test], hashes[train])
        percentage = 100.0 * matched.mean()
        overlaps.append(percentage)
        split_rows.append({'Dataset': dataset, 'Seed': seed,
                           'Test rows': len(test),
                           'Overlapping test rows': int(matched.sum()),
                           'Test overlap pct': percentage})
    return {
        'Dataset': dataset, 'Rows': len(features),
        'Unique feature vectors': len(features.drop_duplicates()),
        'Excess exact feature-label duplicates': excess,
        'Duplicate excess pct': 100.0 * excess / len(features),
        'Conflicting feature-vector groups': len(conflict_hashes),
        'Records in conflicting groups': int(groups['hash'].isin(conflict_hashes).sum()),
        'Mean random-split test overlap pct': float(np.mean(overlaps)),
        'Std random-split test overlap pct': float(np.std(overlaps, ddof=1)),
    }, split_rows

_audit_rows, _split_rows = [], []
for _features, _labels, _name in [(X, y, 'ToN-IoT'), (Xi, yi, 'IoT-23')]:
    _audit, _splits = duplicate_audit(_features, _labels, _name)
    _audit_rows.append(_audit)
    _split_rows.extend(_splits)
pd.DataFrame(_audit_rows).to_csv(OUT/'duplicate_leakage_audit.csv', index=False)
pd.DataFrame(_split_rows).to_csv(OUT/'duplicate_overlap_per_seed.csv', index=False)
print(pd.DataFrame(_audit_rows).to_string(index=False))
