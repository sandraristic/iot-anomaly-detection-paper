import argparse
import io
import re
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

FEATS=['proto','service','duration','src_bytes','dst_bytes','conn_state','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes']
NUM=['duration','src_bytes','dst_bytes','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes']
CAT=['proto','service','conn_state']
SEEDS=[11,23,42,77,101]
CORE=['2018-09-21-11-40-22-192.168.2.3-zeek-conn-log.labeled.txt',
      '2018-12-21-15-50-14-192.168.1.195-zeek-conn-log.labeled.txt',
      '2018-10-25-14-06-32-192.168.1.132-zeek-conn.log.labeled.txt']

def parse_iot23(path):
    rows=[]
    with open(path,encoding='utf-8',errors='replace') as f:
        for raw in f:
            if not raw.strip() or raw.startswith('#'): continue
            parts=raw.rstrip('\n').split('\t')
            if len(parts)<21: continue
            toks=re.split(r'\s+',parts[20].strip())
            label=toks[1] if len(toks)>1 else None
            if label is not None: rows.append(parts[:20]+[label])
    cols=['ts','uid','id.orig_h','id.orig_p','id.resp_h','id.resp_p','proto','service','duration',
          'orig_bytes','resp_bytes','conn_state','local_orig','local_resp','missed_bytes',
          'history','orig_pkts','orig_ip_bytes','resp_pkts','resp_ip_bytes','label']
    return pd.DataFrame(rows,columns=cols)

def load_clean_conflict_free_raw(data_dir,dataset):
    if dataset=='ton':
        d=pd.read_csv(data_dir/'train_test_network.csv',low_memory=False)
        X=d[FEATS].copy(); y=d['label'].astype(int)
    else:
        d=pd.concat([parse_iot23(data_dir/name) for name in CORE],ignore_index=True)
        X=d.rename(columns={'orig_bytes':'src_bytes','resp_bytes':'dst_bytes','orig_pkts':'src_pkts',
                    'orig_ip_bytes':'src_ip_bytes','resp_pkts':'dst_pkts','resp_ip_bytes':'dst_ip_bytes'})[FEATS].copy()
        y=d['label'].astype(str).str.lower().eq('malicious').astype(int)
    for c in NUM: X[c]=pd.to_numeric(X[c].replace('-',np.nan),errors='coerce')
    for c in CAT: X[c]=X[c].replace('-',np.nan)
    z=X.copy(); z['target']=y.to_numpy(); z['group_id']=hash_pandas_object(X,index=False).to_numpy()
    conflict_ids=z.groupby('group_id')['target'].nunique().loc[lambda s:s>1].index
    z=z.loc[~z.group_id.isin(conflict_ids)].copy()
    distinct=z.drop_duplicates(subset=FEATS).copy()
    print(f'{dataset}: original={len(X)}, after conflict removal={len(z)}, distinct feature vectors={len(distinct)}, removed conflict groups={len(conflict_ids)}',flush=True)
    return z,distinct

def pipeline(scale):
    numeric=[('imputer',SimpleImputer(strategy='median'))]
    if scale: numeric.append(('scale',StandardScaler()))
    return ColumnTransformer([('num',Pipeline(numeric),NUM),('cat',Pipeline([
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('onehot',OneHotEncoder(handle_unknown='ignore'))]),CAT)])

def models(seed):
    return [
        ('Logistic Regression',LogisticRegression(class_weight='balanced',solver='liblinear',max_iter=300,random_state=seed),True),
        ('Linear SVM',LinearSVC(class_weight='balanced',dual='auto',max_iter=3000,random_state=seed),True),
        ('Random Forest',RandomForestClassifier(n_estimators=60,max_depth=18,class_weight='balanced_subsample',random_state=seed,n_jobs=-1),False),
        ('XGBoost',XGBClassifier(n_estimators=80,max_depth=6,learning_rate=.08,subsample=.9,colsample_bytree=.9,random_state=seed,n_jobs=-1,eval_metric='logloss'),False)]

def run_dataset(raw,unique,dataset,seeds):
    results=[]
    for seed in seeds:
        a,b=train_test_split(unique,test_size=.2,random_state=seed,stratify=unique.target)
        train_ids=set(a.group_id)
        original_train=raw.loc[raw.group_id.isin(train_ids)]
        unique_train=a
        test=b
        assert not set(original_train.group_id).intersection(set(test.group_id))
        for condition,train in [('retain_duplicates',original_train),('remove_duplicates',unique_train)]:
            for name,model,scale in models(seed):
                prep=pipeline(scale)
                Xtr=prep.fit_transform(train[FEATS])
                Xt=prep.transform(test[FEATS])
                model.fit(Xtr,train.target)
                pred=model.predict(Xt)
                row=dict(Dataset=dataset,Seed=seed,Condition=condition,Model=name,
                         Train_rows=len(train),Train_unique_groups=train.group_id.nunique(),Test_unique_rows=len(test),
                         F1=f1_score(test.target,pred,zero_division=0),
                         Accuracy=accuracy_score(test.target,pred),
                         Precision=precision_score(test.target,pred,zero_division=0),
                         Recall=recall_score(test.target,pred,zero_division=0),
                         MCC=matthews_corrcoef(test.target,pred))
                results.append(row)
                print(dataset,seed,condition,name,'F1=',f'{row["F1"]:.4f}',flush=True)
    return results

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,default=Path('additional_experiments/duplicate_analysis'))
    ap.add_argument('--dataset',choices=['both','ton','iot'],default='both')
    ap.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    args=ap.parse_args()
    if not args.data_dir.is_dir(): ap.error(f'No directory: {args.data_dir}')
    args.out_dir.mkdir(exist_ok=True,parents=True)
    all_rows=[]
    for name in (['ton','iot'] if args.dataset=='both' else [args.dataset]):
        raw,unique=load_clean_conflict_free_raw(args.data_dir,name)
        all_rows.extend(run_dataset(raw,unique,name,args.seeds))
        pd.DataFrame(all_rows).to_csv(args.out_dir/'duplicate_sensitivity_per_seed.csv',index=False)
    df=pd.DataFrame(all_rows)
    summary=df.groupby(['Dataset','Condition','Model'],as_index=False).agg(
        F1_mean=('F1','mean'),F1_sd=('F1','std'),Precision_mean=('Precision','mean'),
        Recall_mean=('Recall','mean'),Accuracy_mean=('Accuracy','mean'),MCC_mean=('MCC','mean'),
        Train_rows_mean=('Train_rows','mean'),Test_rows_mean=('Test_unique_rows','mean'))
    summary.to_csv(args.out_dir/'duplicate_sensitivity_summary.csv',index=False)
    print('Saved results in',args.out_dir.resolve())
if __name__=='__main__': main()
