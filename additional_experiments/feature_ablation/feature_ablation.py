from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, average_precision_score, confusion_matrix
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

import argparse
from sklearn.base import clone
from sklearn.metrics import confusion_matrix

ap=argparse.ArgumentParser(description="Ablation of proto and conn_state for Muhstik and Hakai external evaluation")
ap.add_argument('--data-dir', type=Path, default=Path('data'))
ap.add_argument('--out-dir', type=Path, default=Path('additional_experiments/feature_ablation'))
args=ap.parse_args()
RAW=args.data_dir
OUT=args.out_dir
OUT.mkdir(parents=True, exist_ok=True)

FILES={
'2018-09-21-11-40-22-192.168.2.3-zeek-conn-log.labeled.txt':'Benign Amazon Echo',
'2018-10-25-14-06-32-192.168.1.132-zeek-conn.log.labeled.txt':'Benign Philips Hue',
'2018-12-21-15-50-14-192.168.1.195-zeek-conn-log.labeled.txt':'Mirai capture',
'2018-05-19-20-57-19-192.168.2.5-zeek-conn-log.labeled.txt':'Muhstik capture',
'2018-07-31-15-15-09-192.168.100.113-zeek-conn-log.txt':'Hakai capture',
'2018-10-02-13-12-30-192.168.100.103-zeek-conn-log.txt':'Torii capture',
'2019-01-10-14-34-38-192.168.1.197-zeek-conn-log.txt':'Trojan capture',
'2019-07-03-16-41-09-192.168.1.158-zeek-conn-log.txt':'Benign Somfy Door Lock',
}
CORE=['Benign Amazon Echo','Benign Philips Hue','Mirai capture']
EXTERNAL=['Muhstik capture','Hakai capture','Torii capture','Trojan capture','Benign Somfy Door Lock']
CORE_FIELDS=['ts','uid','id.orig_h','id.orig_p','id.resp_h','id.resp_p','proto','service','duration','orig_bytes','resp_bytes','conn_state','local_orig','local_resp','missed_bytes','history','orig_pkts','orig_ip_bytes','resp_pkts','resp_ip_bytes','tunnel_parents']
FEATURES=['proto','service','duration','src_bytes','dst_bytes','conn_state','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes']
NUM=['duration','src_bytes','dst_bytes','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes']
CAT=['proto','service','conn_state']

def parse(path, default_label=None):
    rows=[]
    with open(path,errors='replace') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            parts=line.rstrip('\n').split('\t')
            if len(parts)<21: continue
            tail=parts[20]
            m=re.match(r'^(.*?)\s{3,}([A-Za-z]+)\s{3,}(.*)$',tail)
            if m: tunnel,label,detail=m.groups()
            else: tunnel,label,detail=tail,default_label,'-'
            rows.append(parts[:20]+[tunnel,label,detail])
    return pd.DataFrame(rows,columns=CORE_FIELDS+['label','detailed_label'])

def make_features(d):
    X=pd.DataFrame({
        'proto':d.proto,'service':d.service,'duration':d.duration,
        'src_bytes':d.orig_bytes,'dst_bytes':d.resp_bytes,'conn_state':d.conn_state,
        'missed_bytes':d.missed_bytes,'src_pkts':d.orig_pkts,'src_ip_bytes':d.orig_ip_bytes,
        'dst_pkts':d.resp_pkts,'dst_ip_bytes':d.resp_ip_bytes})
    for c in NUM: X[c]=pd.to_numeric(X[c].replace({'-':np.nan,'(empty)':np.nan,'':np.nan}),errors='coerce')
    for c in CAT: X[c]=X[c].replace({'-':np.nan,'(empty)':np.nan,'':np.nan})
    return X

def prep(scale):
    ns=[('imputer',SimpleImputer(strategy='median'))]
    if scale: ns.append(('scaler',StandardScaler()))
    return ColumnTransformer([
        ('num',Pipeline(ns),NUM),
        ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),CAT)])

def models():
    return {
        'Logistic Regression':Pipeline([('prep',prep(True)),('model',LogisticRegression(class_weight='balanced',solver='liblinear',max_iter=300,random_state=42))]),
        'Linear SVM':Pipeline([('prep',prep(True)),('model',LinearSVC(class_weight='balanced',dual='auto',max_iter=3000,random_state=42))]),
        'Random Forest':Pipeline([('prep',prep(False)),('model',RandomForestClassifier(n_estimators=60,max_depth=18,class_weight='balanced_subsample',n_jobs=-1,random_state=42))]),
        'XGBoost':Pipeline([('prep',prep(False)),('model',XGBClassifier(n_estimators=80,max_depth=6,learning_rate=0.08,subsample=0.9,colsample_bytree=0.9,eval_metric='logloss',random_state=42,n_jobs=-1))])}

selected = ['Benign Amazon Echo', 'Benign Philips Hue', 'Mirai capture', 'Muhstik capture', 'Hakai capture']
frames = []
for filename, scenario in FILES.items():
    if scenario not in selected:
        continue
    p = RAW / filename
    if not p.is_file():
        raise FileNotFoundError(f'Missing data file: {p}')
    data = parse(p)
    data['scenario'] = scenario
    data['label_norm'] = data.label.astype(str).str.strip().str.lower().eq('malicious').astype(int)
    feat = make_features(data)
    df = pd.concat([feat, data[['scenario','label_norm']].reset_index(drop=True)], axis=1)
    df['_fhash'] = pd.util.hash_pandas_object(df[FEATURES], index=False).astype('uint64')

    df = df.drop_duplicates(subset=['scenario', '_fhash', 'label_norm']).reset_index(drop=True)
    frames.append(df)

all_data = pd.concat(frames, ignore_index=True)
core = all_data[all_data.scenario.isin(CORE)].copy()
rows = []
variants = {
    'all_11_features': [],
    'without_conn_state': ['conn_state'],
    'without_proto': ['proto'],
    'without_both': ['conn_state', 'proto'],
}

for scenario in ['Muhstik capture', 'Hakai capture']:
    test = all_data[all_data.scenario == scenario].copy()

    train = core[~core._fhash.isin(set(test._fhash))].copy()
    print(f'{scenario}: train={len(train)} test={len(test)} malicious={int(test.label_norm.sum())}', flush=True)
    for variant, excluded in variants.items():
        features = [f for f in FEATURES if f not in excluded]
        nums = [f for f in NUM if f in features]
        cats = [f for f in CAT if f in features]
        for model_name, original in models().items():
            estimator = clone(original.named_steps['model'])
            transformer = ColumnTransformer([
                ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))] +
                    ([('scaler', StandardScaler())] if model_name in ('Logistic Regression','Linear SVM') else [])), nums),
                ('cat', Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                                  ('onehot',OneHotEncoder(handle_unknown='ignore'))]), cats),
            ])
            pipeline = Pipeline([('prep', transformer), ('model', estimator)])
            pipeline.fit(train[features], train.label_norm.astype(int))
            actual = test.label_norm.astype(int)
            predicted = pipeline.predict(test[features])
            tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
            fpr = float(fp / (fp + tn)) if fp + tn else float('nan')
            rows.append({
                'Scenario':scenario, 'Variant':variant, 'Model':model_name,
                'Features':len(features), 'Train_n':len(train), 'Test_n':len(test),
                'Malicious_n':int(actual.sum()), 'Benign_n':int((actual == 0).sum()),
                'Accuracy':accuracy_score(actual,predicted),
                'Precision':precision_score(actual,predicted,zero_division=0),
                'Recall':recall_score(actual,predicted,zero_division=0),
                'F1':f1_score(actual,predicted,zero_division=0),
                'FPR':fpr,'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)
            })
            print(scenario, variant, model_name, f'F1={rows[-1]["F1"]:.4f}', flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'feature_ablation_results.csv', index=False)

results = pd.DataFrame(rows)
reference = results[results.Variant == 'all_11_features'][['Scenario','Model','F1','Recall','FPR']].rename(
    columns={'F1':'Baseline_F1','Recall':'Baseline_Recall','FPR':'Baseline_FPR'})
comparison = results.merge(reference, on=['Scenario','Model'])
comparison['Delta_F1'] = comparison.F1 - comparison.Baseline_F1
comparison['Delta_Recall'] = comparison.Recall - comparison.Baseline_Recall
comparison['Delta_FPR'] = comparison.FPR - comparison.Baseline_FPR
comparison.to_csv(OUT / 'feature_ablation_comparison.csv', index=False)
print(f'Saved results in {OUT.resolve()}')
