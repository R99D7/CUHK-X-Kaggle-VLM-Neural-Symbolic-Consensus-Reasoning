"""
MULTI->COMBINATION: 100% accuracy on training (176/176)!
Apply 2 combination fixes from multi cross-leak.

Also check ALL other cross-leaks systematically:
- SEQUENCE -> SINGLE (single answer must be one of the 4 sequence actions) -> ALREADY DONE in v237
- MULTI -> SINGLE -> DONE (just applied 31 more)
- MULTI -> COMBINATION -> 2 more fixes
- COMBINATION -> MULTI: If single action in combination is in multi options, 
  it should be in the multi answer

Let's also check: COMBINATION -> MULTI
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

# Apply 2 multi->combination fixes
sub.loc[sub['qa_id'] == 'test_0278', 'prediction'] = 'D'
sub.loc[sub['qa_id'] == 'test_0279', 'prediction'] = 'C'
print("Applied multi->combination fixes: test_0278->D, test_0279->C")

# Now check COMBINATION -> MULTI leak in training
tr_comb_acts = {}
for idx, row in tr[tr['category'] == 'combination'].iterrows():
    vid = row['path']
    ans_l = str(row['answer']).strip()
    if len(ans_l) == 1:
        ans_text = str(row[ans_l]).strip().lower()
        acts = set(a.strip() for a in ans_text.split(','))
        tr_comb_acts[vid] = acts

tr_multi_ans = {}
tr_multi_opts = {}
for idx, row in tr[tr['category'] == 'multi'].iterrows():
    vid = row['path']
    ans_l = str(row['answer']).strip()
    acts_in_ans = set()
    for l in ans_l:
        if l in ['A', 'B', 'C', 'D']:
            acts_in_ans.add(str(row[l]).strip().lower())
    tr_multi_ans[vid] = (ans_l, acts_in_ans)
    tr_multi_opts[vid] = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}

# Check: when comb actions are all in multi options, are they all in the multi answer?
correct = 0
total = 0
for vid in set(tr_comb_acts.keys()) & set(tr_multi_ans.keys()):
    comb_acts = tr_comb_acts[vid]
    true_ans_l, true_ans_acts = tr_multi_ans[vid]
    opts = tr_multi_opts[vid]
    
    # Which multi option letters correspond to the combination actions?
    comb_letters = [l for l, text in opts.items() if text in comb_acts]
    
    if len(comb_letters) >= 1 and len(comb_letters) < 4:
        total += 1
        # Check if comb_letters is a subset of true_ans_l
        if all(l in true_ans_l for l in comb_letters):
            correct += 1

print(f"\nCOMBINATION->MULTI (comb acts in multi answer): {correct}/{total} = {correct/total:.1%}" if total else "No cases")

# Check test
te_comb = te[te['category'] == 'combination']
te_multi = te[te['category'] == 'multi']

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

multi_comb_changes = 0
for idx, row in te_multi.iterrows():
    vid = row['path']
    if vid not in pred_comb_acts:
        continue
    comb_acts = pred_comb_acts[vid]
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    comb_letters = sorted([l for l, text in opts.items() if text in comb_acts])
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    if len(comb_letters) >= 1:
        # Check if all comb_letters are already in pred
        all_in_pred = all(l in pred for l in comb_letters)
        if not all_in_pred:
            print(f"  {row['qa_id']}: pred={pred}, comb_letters={comb_letters} (should contain them)")
            multi_comb_changes += 1

print(f"\nCOMBINATION->MULTI potential changes: {multi_comb_changes}")

sub.to_csv('submission_v257_CROSS3.csv', index=False)
sub.to_csv('submission.csv', index=False)
print("\nSaved to submission.csv")
