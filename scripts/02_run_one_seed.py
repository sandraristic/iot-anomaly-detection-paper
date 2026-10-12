import sys,time
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,average_precision_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

BASE = Path(__file__).resolve().parents[1] / 'data' / 'robustness_checks'
dataset=sys.argv[1]
seed=int(sys.argv[2])
FEATURES=['proto','service','duration','src_bytes','dst_bytes','conn_state','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes']
NUM=['duration','src_bytes','dst_bytes','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes']
CAT=['proto','service','conn_state']

df=pd.read_pickle(BASE/('ton_unique.pkl' if dataset=='ton' else 'iot_unique.pkl'))
X=df[FEATURES]; y=df.target.astype(int)

def pre(scale):
    ns=[('imp',SimpleImputer(strategy='median'))]
    if scale: ns.append(('scale',StandardScaler()))
    return ColumnTransformer([
        ('num',Pipeline(ns),NUM),
        ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),CAT)
    ])

Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=seed)

t=time.perf_counter(); ps=pre(True); A=ps.fit_transform(Xtr); B=ps.transform(Xte); print('scaled prep',time.perf_counter()-t, A.shape, flush=True)
t=time.perf_counter(); pt=pre(False); C=pt.fit_transform(Xtr); D=pt.transform(Xte); print('tree prep',time.perf_counter()-t,C.shape, flush=True)

mods=[
    ('Logistic Regression',LogisticRegression(max_iter=300,class_weight='balanced',solver='liblinear',random_state=seed),A,B),
    ('Linear SVM',LinearSVC(class_weight='balanced',dual='auto',max_iter=3000,random_state=seed),A,B),
    ('Random Forest',RandomForestClassifier(n_estimators=60,max_depth=18,class_weight='balanced_subsample',n_jobs=-1,random_state=seed),C,D),
    ('XGBoost',XGBClassifier(n_estimators=80,max_depth=6,learning_rate=.08,subsample=.9,colsample_bytree=.9,n_jobs=-1,random_state=seed,eval_metric='logloss'),C,D)
]
rows=[]
for n,m,a,b in mods:
    t=time.perf_counter(); m.fit(a,ytr); dur=time.perf_counter()-t
    pred=m.predict(b)
    sc=m.predict_proba(b)[:,1] if hasattr(m,'predict_proba') else m.decision_function(b)
    tn,fp,fn,tp=confusion_matrix(yte,pred,labels=[0,1]).ravel()
    rows.append({
        'Dataset':dataset,'Seed':seed,'Model':n,
        'Accuracy':accuracy_score(yte,pred),
        'Precision':precision_score(yte,pred,zero_division=0),
        'Recall':recall_score(yte,pred,zero_division=0),
        'F1':f1_score(yte,pred,zero_division=0),
        'MCC':matthews_corrcoef(yte,pred),
        'PR_AUC':average_precision_score(yte,sc),
        'TN':tn,'FP':fp,'FN':fn,'TP':tp,'Train_s':dur
    })
    print(n,dur,rows[-1]['F1'], flush=True)

out=BASE/f'{dataset}_seed_{seed}.csv'
pd.DataFrame(rows).to_csv(out,index=False)
print(out)
