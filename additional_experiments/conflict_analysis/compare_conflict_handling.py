import argparse
from pathlib import Path
from zipfile import ZipFile
from contextlib import contextmanager
import re
import io
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

FEATS = ['proto','service','duration','src_bytes','dst_bytes','conn_state','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes']
NUM = ['duration','src_bytes','dst_bytes','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes']
CAT = ['proto','service','conn_state']
SEEDS = [11, 23, 42, 77, 101]
CORE = ['2018-09-21-11-40-22-192.168.2.3-zeek-conn-log.labeled.txt',
        '2018-12-21-15-50-14-192.168.1.195-zeek-conn-log.labeled.txt',
        '2018-10-25-14-06-32-192.168.1.132-zeek-conn.log.labeled.txt']

def parse_iot23(stream):
    rows=[]
    for raw in io.TextIOWrapper(stream, encoding='utf-8', errors='replace'):
        if raw.startswith('#') or not raw.strip(): continue
        parts=raw.rstrip('\n').split('\t')
        if len(parts)<21: continue
        ch=re.split(r'\s{2,}',parts[20].strip(),maxsplit=2)
        if len(ch)==3: label=ch[1]
        else:
            toks=parts[20].split()
            label=toks[1] if len(toks)>1 else None
        if label is not None: rows.append(parts[:20]+[label])
    names=['ts','uid','id.orig_h','id.orig_p','id.resp_h','id.resp_p','proto','service',
           'duration','orig_bytes','resp_bytes','conn_state','local_orig','local_resp',
           'missed_bytes','history','orig_pkts','orig_ip_bytes','resp_pkts','resp_ip_bytes','label']
    return pd.DataFrame(rows,columns=names)

@contextmanager
def open_input(source, name):
    """Read the same data from an extracted directory or an original ZIP."""
    if isinstance(source, ZipFile):
        with source.open('data/' + name) as f:
            yield f
    else:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing input file: {path}')
        with path.open('rb') as f:
            yield f

def load_dataset(source, which):
    if which=='ton':
        with open_input(source, 'train_test_network.csv') as f:
            df=pd.read_csv(f,low_memory=False)
        X=df[FEATS].copy()
        y=df['label'].astype(int)
    else:
        pieces=[]
        for name in CORE:
            with open_input(source, name) as f: pieces.append(parse_iot23(f))
        df=pd.concat(pieces,ignore_index=True)
        X=df.rename(columns={'orig_bytes':'src_bytes','resp_bytes':'dst_bytes','orig_pkts':'src_pkts',
              'orig_ip_bytes':'src_ip_bytes','resp_pkts':'dst_pkts','resp_ip_bytes':'dst_ip_bytes'})[FEATS].copy()
        y=df.label.astype(str).str.lower().eq('malicious').astype(int)
    for c in NUM: X[c]=pd.to_numeric(X[c].replace('-',np.nan),errors='coerce')
    for c in CAT: X[c]=X[c].replace('-',np.nan)
    h=hash_pandas_object(X,index=False).astype('uint64')
    frame=X.copy()
    frame['target']=y.to_numpy()
    frame['group_id']=h.to_numpy()
    group_label_count=frame.groupby('group_id').target.nunique()
    bad=set(group_label_count.index[group_label_count>1])
    frame['conflict']=frame.group_id.isin(bad)

    frame=frame.drop_duplicates(subset=FEATS+['target']).reset_index(drop=True)
    print(f'{which}: raw={len(X)}, conflict_groups={len(bad)}, raw_conflict_rows={int(h.isin(bad).sum())}, unique_feature_label_rows={len(frame)}',flush=True)
    return frame

def prep(scale):
    num=[('imputer',SimpleImputer(strategy='median'))]
    if scale: num.append(('scale',StandardScaler()))
    return ColumnTransformer([
        ('num',Pipeline(num),NUM),
        ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                         ('onehot',OneHotEncoder(handle_unknown='ignore'))]),CAT)
    ])

def models(seed):
    return [
        ('Logistic Regression',LogisticRegression(max_iter=300,class_weight='balanced',solver='liblinear',random_state=seed),True),
        ('Linear SVM',LinearSVC(class_weight='balanced',dual='auto',max_iter=3000,random_state=seed),True),
        ('Random Forest',RandomForestClassifier(n_estimators=60,max_depth=18,class_weight='balanced_subsample',n_jobs=-1,random_state=seed),False),
        ('XGBoost',XGBClassifier(n_estimators=80,max_depth=6,learning_rate=.08,subsample=.9,colsample_bytree=.9,n_jobs=-1,random_state=seed,eval_metric='logloss'),False)
    ]

def metrics(y,pred):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(Accuracy=accuracy_score(y,pred), Precision=precision_score(y,pred,zero_division=0),
                Recall=recall_score(y,pred,zero_division=0),F1=f1_score(y,pred,zero_division=0),
                MCC=matthews_corrcoef(y,pred),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),Test_n=len(y))

def run(frame,dataset,seed):
    group_label=frame.groupby('group_id').target.mean()
    group_ids=group_label.index.to_numpy()
    strat=(group_label.to_numpy()>=0.5).astype(int)
    train_groups,test_groups=train_test_split(group_ids,test_size=.2,random_state=seed,stratify=strat)
    is_train=frame.group_id.isin(train_groups)
    rows=[]
    for condition in ('remove_conflicts','retain_conflicts'):
        train=frame.loc[is_train].copy()
        if condition=='remove_conflicts': train=train.loc[~train.conflict]
        test=frame.loc[~is_train].copy()
        clean_test=test.loc[~test.conflict]
        for model_name,est,scaled in models(seed):
            p=prep(scaled)
            A=p.fit_transform(train[FEATS]); B=p.transform(clean_test[FEATS]); C=None
            est.fit(A,train.target)
            pred=est.predict(B)
            rows.append(dict(Dataset=dataset,Seed=seed,Condition=condition,Evaluation='common_nonconflict_test',Model=model_name,
                             Train_n=len(train),Conflicting_train_rows=int(train.conflict.sum()),**metrics(clean_test.target,pred)))
            if condition=='retain_conflicts':
                C=p.transform(test[FEATS]); p_all=est.predict(C)
                rows.append(dict(Dataset=dataset,Seed=seed,Condition=condition,Evaluation='all_test_including_conflicts',Model=model_name,
                                 Train_n=len(train),Conflicting_train_rows=int(train.conflict.sum()),**metrics(test.target,p_all)))
            print(dataset,seed,condition,model_name,'F1(common)=',round(rows[-(2 if condition=='retain_conflicts' else 1)]['F1'],4),flush=True)
    return rows

def main():
    ap=argparse.ArgumentParser()
    inputs = ap.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--data-dir', type=Path, help='Folder with extracted original inputs (e.g. data/)')
    inputs.add_argument('--data-zip', type=Path, help='Original ZIP archive (optional alternative)')
    ap.add_argument('--out-dir',type=Path,default=Path('additional_experiments/conflict_analysis'))
    ap.add_argument('--dataset',choices=['ton','iot','both'],default='both')
    ap.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    results=[]
    @contextmanager
    def get_source():
        if args.data_dir is not None:
            if not args.data_dir.is_dir():
                raise NotADirectoryError(f'Data directory not found: {args.data_dir}')
            yield args.data_dir
        else:
            with ZipFile(args.data_zip) as z:
                yield z

    with get_source() as source:
        for ds in (['ton','iot'] if args.dataset=='both' else [args.dataset]):
            frame=load_dataset(source,ds)
            for seed in args.seeds:
                results.extend(run(frame,ds,seed))
                pd.DataFrame(results).to_csv(args.out_dir/'conflict_sensitivity_per_seed.csv',index=False)
    df=pd.DataFrame(results)
    df.groupby(['Dataset','Condition','Evaluation','Model'],as_index=False).agg(
        F1_mean=('F1','mean'),F1_sd=('F1','std'),Precision_mean=('Precision','mean'),
        Recall_mean=('Recall','mean'),MCC_mean=('MCC','mean'),Test_n_mean=('Test_n','mean')
    ).to_csv(args.out_dir/'conflict_sensitivity_summary.csv',index=False)
    print('Wrote:',args.out_dir.resolve())

if __name__=='__main__': main()
