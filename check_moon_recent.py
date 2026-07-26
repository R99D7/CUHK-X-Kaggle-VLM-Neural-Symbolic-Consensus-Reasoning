"""
Check if Moondream is in recent ensembles.
"""
import pandas as pd

recent_sub = pd.read_csv('submission_v263_FINAL.csv')
moon = pd.read_csv('submission_hybrid_moondream.csv')

sub_preds = dict(zip(recent_sub['qa_id'], recent_sub['prediction']))
moon_preds = dict(zip(moon['qa_id'], moon['prediction']))
te = pd.read_csv('test_qa.csv')
emo_qs = te[te['category'] == 'emotion']['qa_id'].tolist()

match = 0
for q in emo_qs:
    sp = str(sub_preds.get(q, '')).strip()
    mp = str(moon_preds.get(q, '')).strip()
    if sp == mp: match += 1

print(f"Recent submission matches Moondream: {match}/{len(emo_qs)}")
