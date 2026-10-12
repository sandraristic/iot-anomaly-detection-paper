from pathlib import Path
import re, numpy as np, pandas as pd, time
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, average_precision_score, confusion_matrix
from xgboost import XGBClassifier

BASE = Path(__file__).resolve().parents[1] / 'data'
OUT = BASE / 'scenario_loso_v19'
OUT.mkdir(parents=True, exist_ok=True)
FILES={
    'Amazon Echo (benign)':(BASE/'2018-09-21-11-40-22-192.168.2.3-zeek-conn-log.labeled.txt', True),
    'Philips Hue (benign)':(BASE/'2018-10-25-14-06-32-192.168.1.132-zeek-conn.log.labeled.txt', True),
    'Core malware capture':(BASE/'2018-12-21-15-50-14-192.168.1.195-zeek-conn-log.labeled.txt', True),
    'Muhstik':(BASE/'2018-05-19-20-57-19-192.168.2.5-zeek-conn-log.labeled.txt', True),
    'Hakai':(BASE/'2018-07-31-15-15-09-192.168.100.113-zeek-conn-log.txt', True),
    'Torii 20-1':(BASE/'2018-10-02-13-12-30-192.168.100.103-zeek-conn-log.txt', True),
    'Torii 21-1':(BASE/'2018-10-03-15-22-32-192.168.100.113-zeek-conn-log.txt', True),
    'Trojan/FileDownload':(BASE/'2019-01-10-14-34-38-192.168.1.197-zeek-conn-log.txt', True),
    'Somfy Door Lock (benign)':(BASE/'2019-07-03-16-41-09-192.168.1.158-zeek-conn-log.txt', False),
}
CORE_FIELDS=['ts','uid','id.orig_h','id.orig_p','id.resp_h','id.resp_p','proto','service','duration','orig_bytes','resp_bytes','conn_state','local_orig','local_resp','missed_bytes','history','orig_pkts','orig_ip_bytes','resp_pkts','resp_ip_bytes','tunnel_parents']
FEATURES=['proto','service','duration','src_bytes','dst_bytes','conn_state','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes']
NUM=['duration','src_bytes','dst_bytes','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes']
CAT=['proto','service','conn_state']

def parse(path, labeled=True):
    rows=[]
    with open(path,errors='replace') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            parts=line.rstrip('\n').split('\t')
            if len(parts)<21: continue
            tail=parts[20]
            if labeled:
                m=re.match(r'^(.*?)\s{3,}([A-Za-z]+)\s{3,}(.*)$',tail)
                if m:
                    tunnel,label,detail=m.groups()
                else:
                    tunnel,label,detail=tail,None,'-'
            else:
                tunnel,label,detail=tail,'Benign','-'
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

def unique_capture(name,path,labeled):
    d=parse(path,labeled)
    if d.label.isna().any():
        d=d[d.label.notna()].copy()
    d['label_norm']=d.label.astype(str).str.strip().str.lower().map(lambda x:0 if x=='benign' else 1)
    X=make_features(d).reset_index(drop=True)
    z=pd.concat([X,d[['label_norm','detailed_label']].reset_index(drop=True)],axis=1)
    z['_fhash']=pd.util.hash_pandas_object(z[FEATURES],index=False).astype('uint64')
    conflict=set(z.groupby('_fhash').label_norm.nunique().loc[lambda s:s>1].index)
    clean=z[~z._fhash.isin(conflict)].copy()
    u=clean.groupby(['_fhash','label_norm'],as_index=False).agg(
        detailed_label=('detailed_label',lambda s:'|'.join(sorted(set(map(str,s))))),
        raw_count=('label_norm','size'),
        **{c:(c,'first') for c in FEATURES})
    u['capture']=name
    return d,u,len(conflict)

def prep(scale):
    ns=[('imputer',SimpleImputer(strategy='median'))]
    if scale: ns.append(('scaler',StandardScaler()))
    return ColumnTransformer([
        ('num',Pipeline(ns),NUM),
        ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),CAT)
    ])

def models(seed=42):
    return {
        'Logistic Regression':Pipeline([('prep',prep(True)),('model',LogisticRegression(class_weight='balanced',solver='liblinear',max_iter=300,random_state=seed))]),
        'Linear SVM':Pipeline([('prep',prep(True)),('model',LinearSVC(class_weight='balanced',dual='auto',max_iter=3000,random_state=seed))]),
        'Random Forest':Pipeline([('prep',prep(False)),('model',RandomForestClassifier(n_estimators=60,max_depth=18,class_weight='balanced_subsample',n_jobs=-1,random_state=seed))]),
        'XGBoost':Pipeline([('prep',prep(False)),('model',XGBClassifier(n_estimators=80,max_depth=6,learning_rate=0.08,subsample=0.9,colsample_bytree=0.9,eval_metric='logloss',random_state=seed,n_jobs=-1))])}

caps={}; summary=[]
for name,(path,labeled) in FILES.items():
    raw,u,nconf=unique_capture(name,path,labeled)
    caps[name]=u
    summary.append({'Scenario':name,'Raw rows':len(raw),'Unique clean patterns':len(u),'Unique benign':int((u.label_norm==0).sum()),'Unique malicious':int((u.label_norm==1).sum()),'Conflict feature groups':nconf})
pd.DataFrame(summary).to_csv(OUT/'capture_summary.csv',index=False)
combined = pd.concat(list(caps.values()), ignore_index=True)
combined['_fhash'] = pd.util.hash_pandas_object(
    combined[FEATURES], index=False
).astype('uint64')

caps = {
    name: combined[combined['capture'] == name].copy()
    for name in caps
}
allres=[]; detail=[]
for held,te in caps.items():
    tr=pd.concat([u for name,u in caps.items() if name!=held],ignore_index=True)
    test_hash=set(te._fhash)
    overlap_before=int(tr._fhash.isin(test_hash).sum())
    tr=tr[~tr._fhash.isin(test_hash)].copy()
    gconf=set(tr.groupby('_fhash').label_norm.nunique().loc[lambda s:s>1].index)
    tr=tr[~tr._fhash.isin(gconf)].copy()
    tr=tr.sort_values('_fhash').drop_duplicates(['_fhash','label_norm'])
    ytr=tr.label_norm.astype(int); y=te.label_norm.astype(int)
    for mname,p in models().items():
        t0=time.perf_counter(); p.fit(tr[FEATURES],ytr); train_t=time.perf_counter()-t0
        pred=p.predict(te[FEATURES])
        if hasattr(p,'predict_proba'):
            try: score=p.predict_proba(te[FEATURES])[:,1]
            except Exception: score=p.decision_function(te[FEATURES])
        else: score=p.decision_function(te[FEATURES])
        cm=confusion_matrix(y,pred,labels=[0,1]); tn,fp,fn,tp=[int(x) for x in cm.ravel()]
        row={'Held-out scenario':held,'Model':mname,'Train unique':len(tr),'Train malicious':int(ytr.sum()),'Test unique':len(te),'Test benign':int((y==0).sum()),'Test malicious':int((y==1).sum()),'Exact source rows removed due to test overlap':overlap_before,'Train conflict hashes removed':len(gconf),'Accuracy':accuracy_score(y,pred),'TN':tn,'FP':fp,'FN':fn,'TP':tp,'FPR':fp/(fp+tn) if fp+tn else np.nan,'Train time':train_t}
        if y.nunique()>1:
            row.update({'Precision':precision_score(y,pred,zero_division=0),'Recall':recall_score(y,pred,zero_division=0),'F1':f1_score(y,pred,zero_division=0),'MCC':matthews_corrcoef(y,pred),'PR-AUC':average_precision_score(y,score)})
        else:
            row.update({'Precision':np.nan,'Recall':np.nan,'F1':np.nan,'MCC':np.nan,'PR-AUC':np.nan})
        allres.append(row)
        if int((y==1).sum())>0:
            tt=te.copy(); tt['pred']=pred
            for det,g in tt[tt.label_norm==1].groupby('detailed_label'):
                detail.append({'Held-out scenario':held,'Model':mname,'Attack detail':det,'Unique malicious patterns':len(g),'Detected':int(g.pred.sum()),'Recall':float(g.pred.mean())})

res=pd.DataFrame(allres)
res.to_csv(OUT/'leave_one_capture_out_results.csv',index=False)
pd.DataFrame(detail).to_csv(OUT/'leave_one_capture_attack_detail.csv',index=False)
rows=[]
for held,g in res.groupby('Held-out scenario'):
    nmal=int(g['Test malicious'].iloc[0]); nben=int(g['Test benign'].iloc[0])
    if nmal>0:
        best=g.sort_values(['F1','Recall'],ascending=False).iloc[0]
        rows.append({'Scenario':held,'Unique malicious':nmal,'Unique benign':nben,'Best model':best.Model,'Best recall':best.Recall,'Best F1':best.F1,'Best MCC':best.MCC,'Min recall across models':g.Recall.min(),'Max recall across models':g.Recall.max()})
    else:
        rows.append({'Scenario':held,'Unique malicious':0,'Unique benign':nben,'Best model':'-','Best recall':np.nan,'Best F1':np.nan,'Best MCC':np.nan,'Min recall across models':np.nan,'Max recall across models':np.nan,'Max FPR across models':g.FPR.max()})
pd.DataFrame(rows).to_csv(OUT/'scenario_summary_compact.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
