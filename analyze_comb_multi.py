"""
COMBINATION->MULTI: 100% accuracy!
When combination answer actions match specific multi option letters,
those letters MUST be in the multi answer.

We need to be careful: the multi answer is a SET of letters like "ABC" or "CD".
The combination gives us some letters that MUST be included.
But we need to determine the FULL answer (which other letters to include).

Strategy: if combination tells us letters X must be in answer,
and current pred is missing some of X, we need to add them.
But adding letters to multi answer means changing from e.g. "CD" to "ACD".

The question is: which are the FULL correct letters?
The combination tells us lower bound (must include X).
The multi question asks which activities were in the video.

Actually: we need to be careful. The issue is:
- Current pred = "CD" but comb_letters = ['A'] means A should ALSO be in the answer
- But changing "CD" to "ACD" might be wrong if the correct answer is "AB" or just "A"

Let me check: in training, when comb_letters are a subset of multi answer,
how often is the multi answer EXACTLY comb_letters vs. a superset?
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

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
    tr_multi_ans[vid] = ans_l
    tr_multi_opts[vid] = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}

# Check: when comb_letters are forced into multi, is the multi answer ONLY comb_letters?
comb_exact = 0
comb_superset = 0
for vid in set(tr_comb_acts.keys()) & set(tr_multi_ans.keys()):
    comb_acts = tr_comb_acts[vid]
    true_ans_l = tr_multi_ans[vid]
    opts = tr_multi_opts[vid]
    
    comb_letters = sorted([l for l, text in opts.items() if text in comb_acts])
    ans_letters = sorted([l for l in true_ans_l if l in 'ABCD'])
    
    if len(comb_letters) >= 1 and set(comb_letters).issubset(set(ans_letters)):
        if comb_letters == ans_letters:
            comb_exact += 1
        else:
            comb_superset += 1

print(f"Comb letters == multi answer: {comb_exact}")
print(f"Comb letters subset of multi answer (more letters): {comb_superset}")
print(f"When we use comb letters as answer, % correct: {comb_exact}/{comb_exact+comb_superset} = {comb_exact/(comb_exact+comb_superset):.1%}")

# NOW: apply the fixes where we can be confident
# Strategy: if comb_letters gives us a COMPLETE answer (not just partial),
# i.e., comb_letters contain ALL actions that are in the video

# Actually the best strategy: 
# For each multi question, the answer must INCLUDE all comb_letters.
# If current pred is missing some comb_letters, we add them.
# If current pred has comb_letters already (subset check), keep as is.

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
    
    # Check if all comb_letters are in pred
    missing = [l for l in comb_letters if l not in pred]
    if missing:
        # Add missing letters to pred and sort
        new_pred_letters = sorted(set(list(pred.replace('nan', '')) + comb_letters))
        new_pred_letters = [l for l in new_pred_letters if l in 'ABCD']
        new_pred = ''.join(sorted(set(new_pred_letters)))
        
        if new_pred != pred and len(new_pred) <= 4:
            print(f"  {row['qa_id']}: pred={pred} + comb={comb_letters} -> {new_pred}")
            changes += 1
        # Don't apply yet - let's analyze first

print(f"\nTotal multi questions needing comb-letter additions: {changes}")
