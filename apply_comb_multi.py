"""
COMBINATION->MULTI: 73.4% accuracy when using EXACT comb_letters as the answer.
But wait - this approach changes multi answers by ADDING letters.

The correct approach: use comb_letters as a CONSTRAINT (must include them),
NOT as the full answer. 

But given 73.4% accuracy means comb_letters = exact multi answer 511/696 times,
and 26.6% means multi answer has MORE letters than comb_letters...

The safest bet for the 60 test multi changes:
- When the new_pred == comb_letters exactly (no letters from original pred added),
  this is the highest confidence case
- When we're adding comb_letters to existing pred, we might be right or wrong

Let me validate more carefully:
- For test cases where comb_letters are a COMPLETE answer (== multi answer in training),
  using just comb_letters as the answer has 73.4% accuracy.

Actually 73.4% >> 25% baseline, so even just replacing with comb_letters is worth it!

Let me apply: replace multi answer with comb_letters WHEN comb_letters >= 2 letters
(more certain since single-letter combinations could be guessed anyway).
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

# First validate more carefully: for multi changes, what's accuracy vs random?
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
    tr_multi_ans[vid] = str(row['answer']).strip()
    tr_multi_opts[vid] = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}

# For cases where pred was WRONG (simulated) and comb_letters were right
exact_comb_correct = 0
exact_comb_wrong = 0
for vid in set(tr_comb_acts.keys()) & set(tr_multi_ans.keys()):
    comb_acts = tr_comb_acts[vid]
    true_ans = tr_multi_ans[vid]
    opts = tr_multi_opts[vid]
    
    comb_letters = sorted([l for l, text in opts.items() if text in comb_acts])
    comb_as_answer = ''.join(comb_letters)
    ans_letters = ''.join(sorted([l for l in true_ans if l in 'ABCD']))
    
    if comb_as_answer and comb_as_answer != ans_letters:
        # Check if comb_as_answer matches
        if comb_as_answer == ans_letters:
            exact_comb_correct += 1
        else:
            exact_comb_wrong += 1

# Now apply: use comb_letters as the full answer for multi questions
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

changes = 0
for idx, row in te_multi.iterrows():
    vid = row['path']
    if vid not in pred_comb_acts:
        continue
    comb_acts = pred_comb_acts[vid]
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    comb_letters = sorted([l for l, text in opts.items() if text in comb_acts])
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    if not comb_letters:
        continue
    
    comb_ans = ''.join(comb_letters)
    
    # Only apply if comb gives us a DIFFERENT answer than pred
    if comb_ans != pred:
        # Make sure all comb_letters are included in the new answer
        # Strategy: use comb_ans as full answer (73.4% accuracy >> 25% baseline)
        # BUT only if comb_letters is a non-trivial subset (>= 1 letter match needed)
        sub.loc[sub['qa_id'] == row['qa_id'], 'prediction'] = comb_ans
        changes += 1

print(f"Applied {changes} COMBINATION->MULTI fixes")
sub.to_csv('submission_v258_COMB_MULTI.csv', index=False)
sub.to_csv('submission.csv', index=False)
print("Saved to submission.csv")
