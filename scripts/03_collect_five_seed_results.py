from pathlib import Path
import pandas as pd

p = Path(__file__).resolve().parents[1] / 'data' / 'robustness_checks'
all_rows=[]
for d in ['ton','iot']:
    for s in [11,23,42,77,101]:
        all_rows.append(pd.read_csv(p/f'{d}_seed_{s}.csv'))

df=pd.concat(all_rows,ignore_index=True)
df.to_csv(p/'five_seed_duplicate_aware_results.csv',index=False)
metrics=['Accuracy','Precision','Recall','F1','MCC','PR_AUC']
sm=df.groupby(['Dataset','Model'])[metrics].agg(['mean','std']).reset_index()
sm.columns=['Dataset','Model']+[f'{a}_{b}' for a,b in sm.columns.tolist()[2:]]
sm.to_csv(p/'five_seed_duplicate_aware_summary.csv',index=False)
print(sm.to_string(index=False))
