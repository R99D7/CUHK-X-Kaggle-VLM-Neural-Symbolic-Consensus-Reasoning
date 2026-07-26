"""
Check all mismatches between argmax and prediction.
"""
import pandas as pd

raw = pd.read_csv('transformer_fixed_raw_predictions.csv')
sub = pd.read_csv('submission.csv') # use original (without our fixes)? Wait, I should load from backup.
# But I only have the fixed one. Let's just load original from test_qa?
# Actually, I have the fixed submission.csv. If I find mismatches in 'emotion' which I never touched, it means the base model has this behavior.

te = pd.read_csv('test_qa.csv')
for cat in ['single', 'combination', 'object_interaction']:
    qs = te[te['category'] == cat]['qa_id'].tolist()
    mismatches = 0
    for q in qs:
        r = raw[raw['qa_id'] == q]
        if r.empty: continue
        r = r.iloc[0]
        
        probs = {l: r[f'raw_prob_{l}'] for l in ['A', 'B', 'C', 'D']}
        argmax_l = max(probs, key=probs.get)
        
        pred = str(sub[sub['qa_id'] == q]['prediction'].values[0]).strip()
        if pred != argmax_l:
            mismatches += 1
            
    print(f"Total {cat} mismatches: {mismatches}")
