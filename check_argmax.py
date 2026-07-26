"""
Check if original submission matches argmax of raw probs.
"""
import pandas as pd

raw = pd.read_csv('transformer_fixed_raw_predictions.csv')
sub = pd.read_csv('submission.csv') # this has my fixes, let's load original
# Actually wait, we don't have the unmodified submission.csv anymore.
# We can just check which ones do NOT match the argmax.
# If they don't match, is it because of my fixes, or were they originally mismatched?
# I'll check emotion, which I NEVER touched!

te = pd.read_csv('test_qa.csv')
emo_q = te[te['category'] == 'emotion']['qa_id'].tolist()

mismatches = 0
for q in emo_q:
    r = raw[raw['qa_id'] == q]
    if r.empty: continue
    r = r.iloc[0]
    
    probs = {l: r[f'raw_prob_{l}'] for l in ['A', 'B', 'C', 'D']}
    argmax_l = max(probs, key=probs.get)
    
    pred = str(sub[sub['qa_id'] == q]['prediction'].values[0]).strip()
    if pred != argmax_l:
        print(f"{q}: pred is {pred}, but argmax is {argmax_l} ({probs[argmax_l]:.3f} vs {probs.get(pred, 0):.3f})")
        mismatches += 1
        
print(f"Total emotion mismatches: {mismatches}")
