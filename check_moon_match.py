"""
Check if submission has Moondream emotion predictions.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')
smart = pd.read_csv('submission_smart_v2.csv')
moon = pd.read_csv('submission_hybrid_moondream.csv')

emo_qs = te[te['category'] == 'emotion']['qa_id'].tolist()
sub_preds = dict(zip(sub['qa_id'], sub['prediction']))
smart_preds = dict(zip(smart['qa_id'], smart['prediction']))
moon_preds = dict(zip(moon['qa_id'], moon['prediction']))

match_sub_moon = 0
match_smart_moon = 0

for q in emo_qs:
    sp = str(sub_preds.get(q, '')).strip()
    smp = str(smart_preds.get(q, '')).strip()
    mp = str(moon_preds.get(q, '')).strip()
    
    if sp == mp: match_sub_moon += 1
    if smp == mp: match_smart_moon += 1

print(f"Submission matches Moondream: {match_sub_moon}/{len(emo_qs)}")
print(f"Smart_v2 matches Moondream: {match_smart_moon}/{len(emo_qs)}")
