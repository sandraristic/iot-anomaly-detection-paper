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

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / 'data'
OUT = BASE / 'reproduced_outputs'
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

frames=[]
for fn,scen in FILES.items():
    default='Benign' if scen=='Benign Somfy Door Lock' else None
    d=parse(RAW/fn,default)
    d['scenario']=scen; d['source_file']=fn
    d['label_norm']=d.label.astype(str).str.strip().str.lower().map(lambda x:0 if x=='benign' else 1)
    X=make_features(d)
    z=pd.concat([X,d[['label_norm','detailed_label','scenario','source_file']].reset_index(drop=True)],axis=1)
    z['_fhash']=pd.util.hash_pandas_object(z[FEATURES],index=False).astype('uint64')
    frames.append(z)
all_df=pd.concat(frames,ignore_index=True)
all_df['_fhash'] = pd.util.hash_pandas_object(
    all_df[FEATURES], index=False
).astype('uint64')
su=all_df.groupby(['scenario','_fhash','label_norm'],as_index=False).agg(
    detailed_label=('detailed_label',lambda s:'|'.join(sorted(set(map(str,s))))),
    count=('label_norm','size'), **{c:(c,'first') for c in FEATURES})

base=su[su.scenario.isin(CORE)].copy()
results=[]; detail_rows=[]; overlap=[]; importance=[]
for hold in EXTERNAL:
    te=su[su.scenario==hold].copy(); tr=base.copy()
    ov=tr._fhash.isin(set(te._fhash))
    overlap.append([hold,len(te),int(te.label_norm.sum()),len(tr),int(ov.sum()),int((~ov).sum())])
    tr=tr[~ov]
    for name,p in models().items():
        p.fit(tr[FEATURES],tr.label_norm.astype(int))
        pred=p.predict(te[FEATURES]); y=te.label_norm.astype(int)
        if hasattr(p,'predict_proba'):
            try: score=p.predict_proba(te[FEATURES])[:,1]
            except Exception: score=p.decision_function(te[FEATURES])
        else: score=p.decision_function(te[FEATURES])
        cm=confusion_matrix(y,pred,labels=[0,1])
        results.append({
            'External scenario':hold,'Model':name,'Unique test':len(te),'Benign':int((y==0).sum()),'Malicious':int(y.sum()),
            'Accuracy':accuracy_score(y,pred),'Precision':precision_score(y,pred,zero_division=0),
            'Recall':recall_score(y,pred,zero_division=0) if y.sum()>0 else np.nan,
            'F1':f1_score(y,pred,zero_division=0) if y.sum()>0 else np.nan,
            'MCC':matthews_corrcoef(y,pred) if len(np.unique(y))>1 else np.nan,
            'PR-AUC':average_precision_score(y,score) if y.sum()>0 else np.nan,
            'FPR':float(pred[y==0].mean()) if (y==0).sum()>0 else np.nan,
            'TN':int(cm[0,0]),'FP':int(cm[0,1]),'FN':int(cm[1,0]),'TP':int(cm[1,1])})

        if name == 'Linear SVM' and hold in ['Muhstik capture', 'Hakai capture']:
            pi = permutation_importance(
                p, te[FEATURES], y, scoring='f1', n_repeats=5,
                random_state=42, n_jobs=-1)
            for feature, mean, std in zip(FEATURES, pi.importances_mean, pi.importances_std):
                importance.append({'Scenario': hold, 'Feature': feature,
                                   'F1 drop': float(mean), 'Std': float(std)})
        te2=te.copy(); te2['pred']=pred
        for dl,g in te2[te2.label_norm==1].groupby('detailed_label'):
            detail_rows.append({'External scenario':hold,'Model':name,'Detailed label':dl,'Unique malicious':len(g),'Detected':int(g.pred.sum()),'Recall':float(g.pred.mean())})

pd.DataFrame(results).to_csv(OUT/'external_scenario_validation_full.csv',index=False)
pd.DataFrame(detail_rows).to_csv(OUT/'external_attack_detail_recall.csv',index=False)
pd.DataFrame(overlap,columns=['Scenario','Test rows','Malicious','Train before','Overlap removed','Train after']).to_csv(OUT/'external_overlap_audit.csv',index=False)
pd.DataFrame(importance, columns=['Scenario','Feature','F1 drop','Std']).to_csv(OUT/'external_permutation_importance_linear_svm.csv', index=False)
print('Wrote',OUT)
