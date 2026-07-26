"""
Estimate accuracy by category using base model probs.
"""
import pandas as pd

sub = pd.read_csv('submission.csv')
te = pd.read_csv('test_qa.csv')
raw = pd.read_csv('transformer_fixed_raw_predictions.csv')

cat_probs = {}
for cat in ['single', 'multi', 'combination', 'sequence', 'emotion', 'object_interaction']:
    cat_qs = te[te['category'] == cat]['qa_id'].tolist()
    probs = []
    for q in cat_qs:
        pred = str(sub[sub['qa_id'] == q]['prediction'].values[0]).strip()
        
        # for single, combination, emotion, object_interaction, it's just a letter
        if cat in ['single', 'combination', 'emotion', 'object_interaction']:
            r = raw[raw['qa_id'] == q]
            if not r.empty:
                probs.append(r.iloc[0][f'raw_prob_{pred}'])
        
        # for sequence it's order, multi it's subset. We don't have exactly simple probs
        # We can just check how confident the model was on its original top choices.
    if probs:
        print(f"{cat}: mean prob = {sum(probs)/len(probs):.4f}")
