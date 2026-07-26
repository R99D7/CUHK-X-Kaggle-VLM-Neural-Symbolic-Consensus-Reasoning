"""
Now let's check ALL other cross-category leaks systematically:

1. Can MULTI predict COMBINATION? If multi answer includes both actions in a combination,
   that combination option is the answer.
2. Can SEQUENCE predict COMBINATION? (already done for exact subset)
3. Can MULTI predict EMOTION? (already showed 11% -> not useful)
4. The reverse: can COMBINATION predict MULTI?
   If the combination answer actions are a subset of multi options,
   those letters should be in the multi answer.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

# MULTI -> COMBINATION leak
# If multi answer acts include both actions of a combination option -> that's the answer
tr_multi_ans = {}
tr_comb_ans_l = {}

for idx, row in tr[tr['category'] == 'multi'].iterrows():
    vid = row['path']
    ans_l = str(row['answer']).strip()
    acts = set()
    for l in ans_l:
        if l in ['A', 'B', 'C', 'D']:
            acts.add(str(row[l]).strip().lower())
    tr_multi_ans[vid] = acts

for idx, row in tr[tr['category'] == 'combination'].iterrows():
    vid = row['path']
    ans_l = str(row['answer']).strip()
    if len(ans_l) == 1:
        ans_text = str(row[ans_l]).strip().lower()
        comb_acts = set(a.strip() for a in ans_text.split(','))
        tr_comb_ans_l[vid] = (ans_l, comb_acts)

# Validate MULTI->COMBINATION
correct = 0
total = 0
for vid in set(tr_multi_ans.keys()) & set(tr_comb_ans_l.keys()):
    multi_acts = tr_multi_ans[vid]
    true_ans_l, true_comb_acts = tr_comb_ans_l[vid]
    
    # Check which combination options are subsets of multi acts
    row_comb = tr[(tr['path'] == vid) & (tr['category'] == 'combination')].iloc[0]
    valid = []
    for l in ['A', 'B', 'C', 'D']:
        opt_text = str(row_comb[l]).strip().lower()
        opt_acts = set(a.strip() for a in opt_text.split(','))
        if opt_acts.issubset(multi_acts):
            valid.append(l)
    
    if len(valid) == 1:
        total += 1
        if valid[0] == true_ans_l:
            correct += 1

print(f"MULTI->COMBINATION (1 valid subset): {correct}/{total} = {correct/total:.1%}" if total else "No cases")

# Apply to TEST
te_comb = te[te['category'] == 'combination']
te_multi = te[te['category'] == 'multi']

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

changes = 0
for idx, row in te_comb.iterrows():
    vid = row['path']
    if vid not in pred_multi_acts:
        continue
    multi_acts = pred_multi_acts[vid]
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    valid = []
    for l in ['A', 'B', 'C', 'D']:
        opt_text = str(row[l]).strip().lower()
        opt_acts = set(a.strip() for a in opt_text.split(','))
        if opt_acts.issubset(multi_acts):
            valid.append(l)
    
    if len(valid) == 1 and pred != valid[0]:
        opt = str(row[valid[0]]).strip()
        print(f"{row['qa_id']}: pred={pred}, multi_cross={valid[0]}({opt}) multi_acts={multi_acts}")
        changes += 1

print(f"\nMULTI->COMBINATION changes: {changes}")
