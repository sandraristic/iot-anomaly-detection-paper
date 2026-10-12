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

BASE = Path(__file__).resolve().parents[1] / 'data'
OUT = BASE / 'torii_pair_validation'
OUT.mkdir(parents=True, exist_ok=True)
FILES={
    'Torii 20-1':BASE/'2018-10-02-13-12-30-192.168.100.103-zeek-conn-log.txt',
    'Torii 21-1':BASE/'2018-10-03-15-22-32-192.168.100.113-zeek-conn-log.txt',
}
CORE_FIELDS=['ts','uid','id.orig_h','id.orig_p','id.resp_h','id.resp_p','proto','service','duration','orig_bytes','resp_bytes','conn_state','local_orig','local_resp','missed_bytes','history','orig_pkts','orig_ip_bytes','resp_pkts','resp_ip_bytes','tunnel_parents']
FEATURES=['proto','service','duration','src_bytes','dst_bytes','conn_state','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes']
NUM=['duration','src_bytes','dst_bytes','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes']
CAT=['proto','service','conn_state']

def parse(path):
    rows=[]
    with open(path,errors='replace') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts=line.rstrip('\n').split('\t')
            if len(parts)<21:
                continue
            tail=parts[20]
            m=re.match(r'^(.*?)\s{3,}([A-Za-z]+)\s{3,}(.*)$',tail)
            if m:
                tunnel,label,detail=m.groups()
            else:
                tunnel,label,detail=tail,None,'-'
            rows.append(parts[:20]+[tunnel,label,detail])
    return pd.DataFrame(rows,columns=CORE_FIELDS+['label','detailed_label'])

def make_features(d):
    X=pd.DataFrame({
        'proto':d.proto,'service':d.service,'duration':d.duration,
        'src_bytes':d.orig_bytes,'dst_bytes':d.resp_bytes,'conn_state':d.conn_state,
        'missed_bytes':d.missed_bytes,'src_pkts':d.orig_pkts,'src_ip_bytes':d.orig_ip_bytes,
        'dst_pkts':d.resp_pkts,'dst_ip_bytes':d.resp_ip_bytes})
    for c in NUM:
        X[c]=pd.to_numeric(X[c].replace({'-':np.nan,'(empty)':np.nan,'':np.nan}),errors='coerce')
    for c in CAT:
        X[c]=X[c].replace({'-':np.nan,'(empty)':np.nan,'':np.nan})
    return X

def unique_capture(name,path):
    d=parse(path)
    d['label_norm']=d.label.astype(str).str.strip().str.lower().map(lambda x:0 if x=='benign' else 1)
    X=make_features(d)
    z=pd.concat([X,d[['label_norm','detailed_label']].reset_index(drop=True)],axis=1)
    z['_fhash']=pd.util.hash_pandas_object(z[FEATURES],index=False).astype('uint64')
    conflict=set(z.groupby('_fhash').label_norm.nunique().loc[lambda s:s>1].index)
    clean=z[~z._fhash.isin(conflict)].copy()
    u=clean.groupby(['_fhash','label_norm'],as_index=False).agg(
        detailed_label=('detailed_label',lambda s:'|'.join(sorted(set(map(str,s))))),
        raw_count=('label_norm','size'),
        **{c:(c,'first') for c in FEATURES})
    u['capture']=name
    return d,u,conflict

def prep(scale):
    ns=[('imputer',SimpleImputer(strategy='median'))]
    if scale:
        ns.append(('scaler',StandardScaler()))
    return ColumnTransformer([
        ('num',Pipeline(ns),NUM),
        ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),CAT)])

def models(seed=42):
    return {
        'Logistic Regression':Pipeline([('prep',prep(True)),('model',LogisticRegression(class_weight='balanced',solver='liblinear',max_iter=300,random_state=seed))]),
        'Linear SVM':Pipeline([('prep',prep(True)),('model',LinearSVC(class_weight='balanced',dual='auto',max_iter=3000,random_state=seed))]),
        'Random Forest':Pipeline([('prep',prep(False)),('model',RandomForestClassifier(n_estimators=60,max_depth=18,class_weight='balanced_subsample',n_jobs=-1,random_state=seed))]),
        'XGBoost':Pipeline([('prep',prep(False)),('model',XGBClassifier(n_estimators=80,max_depth=6,learning_rate=0.08,subsample=0.9,colsample_bytree=0.9,eval_metric='logloss',random_state=seed,n_jobs=-1))])}

caps={}
summary=[]
for name,path in FILES.items():
    raw,u,conf=unique_capture(name,path)
    caps[name]=u
    summary.append({'Capture':name,'Raw rows':len(raw),'Unique clean patterns':len(u),'Unique benign':int((u.label_norm==0).sum()),'Unique malicious':int((u.label_norm==1).sum()),'Conflict feature groups':len(conf)})
pd.DataFrame(summary).to_csv(OUT/'torii_pair_capture_summary.csv',index=False)

u20=caps['Torii 20-1']; u21=caps['Torii 21-1']
all_overlap=set(u20._fhash)&set(u21._fhash)
mal_overlap=set(u20.loc[u20.label_norm==1,'_fhash'])&set(u21.loc[u21.label_norm==1,'_fhash'])
pd.DataFrame([{
    'Shared exact feature patterns':len(all_overlap),
    'Shared malicious feature patterns':len(mal_overlap),
    'Torii 20-1 unique patterns':len(u20),
    'Torii 21-1 unique patterns':len(u21),
}]).to_csv(OUT/'torii_pair_overlap_audit.csv',index=False)

results=[]; errors=[]; importance=[]
for src,tgt in [('Torii 20-1','Torii 21-1'),('Torii 21-1','Torii 20-1')]:
    tr=caps[src].copy(); te=caps[tgt].copy()
    overlap=set(tr._fhash)&set(te._fhash)
    tr=tr[~tr._fhash.isin(overlap)].copy()
    for name,p in models().items():
        ytr=tr.label_norm.astype(int); y=te.label_norm.astype(int)
        p.fit(tr[FEATURES],ytr)
        pred=p.predict(te[FEATURES])
        if hasattr(p,'predict_proba'):
            try: score=p.predict_proba(te[FEATURES])[:,1]
            except Exception: score=p.decision_function(te[FEATURES])
        else:
            score=p.decision_function(te[FEATURES])
        cm=confusion_matrix(y,pred,labels=[0,1])
        results.append({
            'Direction':f'{src} -> {tgt}','Model':name,'Train unique after overlap removal':len(tr),'Train malicious':int(ytr.sum()),
            'Test unique':len(te),'Test malicious':int(y.sum()),'Exact overlap removed from training':len(overlap),
            'Accuracy':accuracy_score(y,pred),'Precision':precision_score(y,pred,zero_division=0),'Recall':recall_score(y,pred,zero_division=0),
            'F1':f1_score(y,pred,zero_division=0),'MCC':matthews_corrcoef(y,pred),'PR-AUC':average_precision_score(y,score),
            'TN':int(cm[0,0]),'FP':int(cm[0,1]),'FN':int(cm[1,0]),'TP':int(cm[1,1])})
        te2=te.copy(); te2['pred']=pred
        for _,row in te2[(te2.label_norm==1)&(te2.pred==0)].iterrows():
            errors.append({'Direction':f'{src} -> {tgt}','Model':name,'Error':'False negative','Detailed label':row.detailed_label,'Raw multiplicity':int(row.raw_count),**{c:row[c] for c in FEATURES}})
        for _,row in te2[(te2.label_norm==0)&(te2.pred==1)].iterrows():
            errors.append({'Direction':f'{src} -> {tgt}','Model':name,'Error':'False positive','Detailed label':row.detailed_label,'Raw multiplicity':int(row.raw_count),**{c:row[c] for c in FEATURES}})
        if name=='Linear SVM':
            pi=permutation_importance(p,te[FEATURES],y,scoring='f1',n_repeats=20,random_state=42,n_jobs=-1)
            for f,m,s in zip(FEATURES,pi.importances_mean,pi.importances_std):
                importance.append({'Direction':f'{src} -> {tgt}','Model':name,'Feature':f,'Mean F1 decrease':m,'Std':s})

pd.DataFrame(results).to_csv(OUT/'torii_pair_transfer_results.csv',index=False)
pd.DataFrame(errors).to_csv(OUT/'torii_pair_error_analysis.csv',index=False)
pd.DataFrame(importance).sort_values(['Direction','Mean F1 decrease'],ascending=[True,False]).to_csv(OUT/'torii_pair_permutation_importance_linear_svm.csv',index=False)
print(pd.DataFrame(results).to_string(index=False))
