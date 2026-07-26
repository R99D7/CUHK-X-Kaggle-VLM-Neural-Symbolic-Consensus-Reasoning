"""
Apply argmax to emotion and object_interaction.
"""
import pandas as pd

raw = pd.read_csv('transformer_fixed_raw_predictions.csv')
sub = pd.read_csv('submission.csv')
te = pd.read_csv('test_qa.csv')

changes = 0
for cat in ['emotion', 'object_interaction']:
    qs = te[te['category'] == cat]['qa_id'].tolist()
    for q in qs:
        r = raw[raw['qa_id'] == q]
        if r.empty: continue
        r = r.iloc[0]
        
        probs = {l: r[f'raw_prob_{l}'] for l in ['A', 'B', 'C', 'D']}
        argmax_l = max(probs, key=probs.get)
        
        old_pred = str(sub[sub['qa_id'] == q]['prediction'].values[0]).strip()
        if old_pred != argmax_l:
            print(f"{q} ({cat}): {old_pred} -> {argmax_l}")
            sub.loc[sub['qa_id'] == q, 'prediction'] = argmax_l
            changes += 1

print(f"\nTotal changes made: {changes}")
sub.to_csv('submission.csv', index=False)
