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

P = Path(__file__).resolve().parents[1] / 'data' / 'robustness_checks'
FEATURES=['proto','service','duration','src_bytes','dst_bytes','conn_state','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes']
NUM=['duration','src_bytes','dst_bytes','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes']
CAT=['proto','service','conn_state']

def pre(scale):
    ns=[('imp',SimpleImputer(strategy='median'))]
    if scale: ns.append(('scale',StandardScaler()))
    return ColumnTransformer([
        ('num',Pipeline(ns),NUM),
        ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),CAT)
    ])

def mods(seed=42):
    return {
        'Logistic Regression':LogisticRegression(max_iter=300,class_weight='balanced',solver='liblinear',random_state=seed),
        'Linear SVM':LinearSVC(class_weight='balanced',dual='auto',max_iter=3000,random_state=seed),
        'Random Forest':RandomForestClassifier(n_estimators=60,max_depth=18,class_weight='balanced_subsample',n_jobs=-1,random_state=seed),
        'XGBoost':XGBClassifier(n_estimators=80,max_depth=6,learning_rate=.08,subsample=.9,colsample_bytree=.9,n_jobs=-1,random_state=seed,eval_metric='logloss')
    }

def met(m,Xtr,ytr,Xte,yte):
    m.fit(Xtr,ytr); pred=m.predict(Xte)
    sc=m.predict_proba(Xte)[:,1] if hasattr(m,'predict_proba') else m.decision_function(Xte)
    tn,fp,fn,tp=confusion_matrix(yte,pred,labels=[0,1]).ravel()
    return dict(
        Accuracy=accuracy_score(yte,pred),Precision=precision_score(yte,pred,zero_division=0),
        Recall=recall_score(yte,pred,zero_division=0),F1=f1_score(yte,pred,zero_division=0),
        MCC=matthews_corrcoef(yte,pred),PR_AUC=average_precision_score(yte,sc),
        TN=tn,FP=fp,FN=fn,TP=tp
    )

ton=pd.read_pickle(P/'ton_unique.pkl'); iot=pd.read_pickle(P/'iot_unique.pkl')
Xt,yt=ton[FEATURES],ton.target.astype(int); Xi,yi=iot[FEATURES],iot.target.astype(int)

def cross(SX,Sy,TX,Ty,direction):
    ps=pre(True); A=ps.fit_transform(SX); B=ps.transform(TX)
    pt=pre(False); C=pt.fit_transform(SX); D=pt.transform(TX)
    out=[]
    for n,m in mods().items():
        if n in ['Logistic Regression','Linear SVM']:
            out.append({'Direction':direction,'Model':n,**met(m,A,Sy,B,Ty)})
        else:
            out.append({'Direction':direction,'Model':n,**met(m,C,Sy,D,Ty)})
    return out

cdf=pd.DataFrame(cross(Xt,yt,Xi,yi,'ToN-IoT -> IoT-23')+cross(Xi,yi,Xt,yt,'IoT-23 -> ToN-IoT'))
cdf.to_csv(P/'cross_dataset_duplicate_aware.csv',index=False)
print(cdf.to_string(index=False))

u=pd.read_pickle(P/'iot_unique_detail.pkl'); Xu=u[FEATURES]; yu=u._y.astype(int); du=u._detail.astype(str)
counts=du.value_counts().rename_axis('Attack').reset_index(name='Unique_patterns')
counts.to_csv(P/'iot23_unique_attack_pattern_counts.csv',index=False)
print('\ncounts\n',counts.to_string(index=False))

ben=np.where(yu.to_numpy()==0)[0]; rows=[]
for held in ['C&C','DDoS','PartOfAHorizontalPortScan']:
    btr,bte=train_test_split(ben,test_size=.2,random_state=42)
    atk=np.where(du.to_numpy()==held)[0]
    oth=np.where((yu.to_numpy()==1)&(du.to_numpy()!=held))[0]
    tr=np.r_[btr,oth]; te=np.r_[bte,atk]
    ps=pre(True); A=ps.fit_transform(Xu.iloc[tr]); B=ps.transform(Xu.iloc[te])
    pt=pre(False); C=pt.fit_transform(Xu.iloc[tr]); D=pt.transform(Xu.iloc[te])
    for n,m in mods().items():
        mm=met(m,A,yu.iloc[tr],B,yu.iloc[te]) if n in ['Logistic Regression','Linear SVM'] else met(m,C,yu.iloc[tr],D,yu.iloc[te])
        rows.append({'Held-out attack':held,'Unique held-out patterns':len(atk),'Model':n,**mm})

udf=pd.DataFrame(rows)
udf.to_csv(P/'unseen_attack_duplicate_aware.csv',index=False)
print('\nunseen\n',udf.to_string(index=False))
