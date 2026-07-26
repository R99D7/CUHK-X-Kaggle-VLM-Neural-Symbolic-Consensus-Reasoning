"""
Compare emotion predictions between our submission and Moondream.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv') # this is the 0.69590 one
moon = pd.read_csv('submission_hybrid_moondream.csv')

emo_qs = te[te['category'] == 'emotion']['qa_id'].tolist()

sub_preds = dict(zip(sub['qa_id'], sub['prediction']))
moon_preds = dict(zip(moon['qa_id'], moon['prediction']))

diff_count = 0
for q in emo_qs:
    sp = str(sub_preds.get(q, '')).strip()
    mp = str(moon_preds.get(q, '')).strip()
    if mp in ['A', 'B', 'C', 'D'] and sp != mp:
        diff_count += 1

print(f"Total emotion differences between submission and moondream: {diff_count} out of {len(emo_qs)}")
