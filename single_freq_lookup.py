"""
Now also look at SINGLE category training lookup.
For single questions, the answer is one action from 4 options.
Build a lookup: frozenset of options -> most common answer in training.
Validate accuracy on training.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

tr_single = tr[tr['category'] == 'single']

def get_opts_frozenset(row):
    return frozenset([str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']])

# Build lookup
tr_sets = {}
for idx, row in tr_single.iterrows():
    fs = get_opts_frozenset(row)
    ans_l = str(row['answer']).strip()
    ans_text = str(row[ans_l]).strip().lower()
    if fs not in tr_sets:
        tr_sets[fs] = {}
    tr_sets[fs][ans_text] = tr_sets[fs].get(ans_text, 0) + 1

# Also: build answer text -> frequency across ALL training single
tr_answer_freq = {}
for idx, row in tr_single.iterrows():
    ans_l = str(row['answer']).strip()
    ans_text = str(row[ans_l]).strip().lower()
    tr_answer_freq[ans_text] = tr_answer_freq.get(ans_text, 0) + 1

# For test single: score each option by how often it appears as an answer in training
te_single = te[te['category'] == 'single']
changes = 0

for idx, row in te_single.iterrows():
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    pred_text = opts[pred]
    
    # Check if option-set matches training (already handled by v248)
    fs = frozenset(opts.values())
    if fs in tr_sets:
        continue  # already covered
    
    # Score by global answer frequency
    scores = {l: tr_answer_freq.get(text, 0) for l, text in opts.items()}
    best_l = max(scores, key=scores.get)
    best_score = scores[best_l]
    pred_score = scores[pred]
    
    if pred != best_l and best_score > pred_score * 2 and best_score >= 5:
        print(f"{row['qa_id']}: pred={pred}({pred_text},{pred_score}) -> {best_l}({opts[best_l]},{best_score})")
        changes += 1

print(f"\nTotal single changes via freq lookup: {changes}")
