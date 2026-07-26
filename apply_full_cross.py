"""
Apply ALL remaining cross-leaks from current submission (v255):

1. MULTI->SINGLE: 100% accuracy on training - use predicted multi to fix single
2. Remaining COMB->SINGLE: using the 20 remaining questions

Start from submission_v255 and add both.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')  # v255

te_single = te[te['category'] == 'single']
te_comb = te[te['category'] == 'combination']
te_multi = te[te['category'] == 'multi']

# Build predicted MULTI acts per video
pred_multi_acts = {}
for idx, row in te_multi.iterrows():
    vid = row['path']
    pred_l = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    acts = set()
    for l in pred_l:
        if l in ['A', 'B', 'C', 'D']:
            acts.add(str(row[l]).strip().lower())
    if vid not in pred_multi_acts:
        pred_multi_acts[vid] = acts
    else:
        pred_multi_acts[vid] |= acts

# Build predicted COMBINATION acts per video (ALL predictions)
pred_comb_acts = {}
for idx, row in te_comb.iterrows():
    vid = row['path']
    pred_l = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    if len(pred_l) == 1:
        pred_text = str(row[pred_l]).strip().lower()
        acts = set(a.strip() for a in pred_text.split(','))
        if vid not in pred_comb_acts:
            pred_comb_acts[vid] = acts
        else:
            pred_comb_acts[vid] |= acts

changes = 0
for idx, row in te_single.iterrows():
    vid = row['path']
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()

    # Try MULTI->SINGLE first (higher confidence since multi is more definitive)
    if vid in pred_multi_acts:
        multi_acts = pred_multi_acts[vid]
        in_multi = [l for l, text in opts.items() if text in multi_acts]
        if len(in_multi) == 1 and pred != in_multi[0]:
            new_pred = in_multi[0]
            print(f"MULTI: {row['qa_id']}: {pred}({opts[pred]}) -> {new_pred}({opts[new_pred]}) acts={multi_acts}")
            sub.loc[sub['qa_id'] == row['qa_id'], 'prediction'] = new_pred
            pred = new_pred
            changes += 1
            continue

    # Then try COMB->SINGLE (for questions not already updated)
    if vid in pred_comb_acts:
        comb_acts = pred_comb_acts[vid]
        in_comb = [l for l, text in opts.items() if text in comb_acts]
        if len(in_comb) == 1 and pred != in_comb[0]:
            new_pred = in_comb[0]
            print(f"COMB:  {row['qa_id']}: {pred}({opts[pred]}) -> {new_pred}({opts[new_pred]}) acts={comb_acts}")
            sub.loc[sub['qa_id'] == row['qa_id'], 'prediction'] = new_pred
            changes += 1

print(f"\nTotal additional cross-leak single fixes: {changes}")
sub.to_csv('submission_v256_FULL_CROSS.csv', index=False)
sub.to_csv('submission.csv', index=False)
print("Saved to submission.csv")
