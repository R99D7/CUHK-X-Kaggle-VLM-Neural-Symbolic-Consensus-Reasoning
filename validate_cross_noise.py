"""
Wait - I need to be more careful. The 100% accuracy was validated on training data
where the TRAINING combination answer was used. But for test data, we're using
the PREDICTED combination answer, which may not be 100% correct.

The key question: does using PREDICTED combination answer still yield high accuracy?
Let me simulate this on training:
- Use a "simulated predicted" combination (could be wrong)
- Check if the cross-leak still works

Also: the combination pairs can have 2, 3, or more actions. When the pair has
more than 2 actions, multiple single options might match.

Let me validate on training with a realistic error rate for combination predictions.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')

# Training validation: use the combination ANSWER to infer single answer
# This simulates our cross-leak

tr_single_ans = {}
tr_single_opts = {}
for idx, row in tr[tr['category'] == 'single'].iterrows():
    vid = row['path']
    ans_l = str(row['answer']).strip()
    if len(ans_l) == 1:
        tr_single_ans[vid] = ans_l
        tr_single_opts[vid] = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}

tr_comb_acts = {}
for idx, row in tr[tr['category'] == 'combination'].iterrows():
    vid = row['path']
    ans_l = str(row['answer']).strip()
    if len(ans_l) == 1:
        ans_text = str(row[ans_l]).strip().lower()
        acts = set(a.strip() for a in ans_text.split(','))
        tr_comb_acts[vid] = acts

# Check cross-leak accuracy on training
exact_1_correct = 0
exact_1_total = 0
multi_match = 0

for vid in set(tr_single_ans.keys()) & set(tr_comb_acts.keys()):
    comb_acts = tr_comb_acts[vid]
    opts = tr_single_opts[vid]
    true_ans_l = tr_single_ans[vid]
    
    in_comb = [l for l, text in opts.items() if text in comb_acts]
    
    if len(in_comb) == 1:
        exact_1_total += 1
        if in_comb[0] == true_ans_l:
            exact_1_correct += 1
    elif len(in_comb) == 2:
        multi_match += 1

print(f"Training cross-leak (1 option in comb): {exact_1_correct}/{exact_1_total} = {exact_1_correct/exact_1_total:.1%}")
print(f"Training cross-leak (2 options in comb): {multi_match}")
print()

# Now: what does our PREDICTED combination answer do for the TRAINING set?
# Let's simulate by randomly corrupting some combination answers
# and see if the cross-leak still works

import random
random.seed(42)
correct_with_noise = 0
total_with_noise = 0

for vid in set(tr_single_ans.keys()) & set(tr_comb_acts.keys()):
    comb_acts = tr_comb_acts[vid]
    opts = tr_single_opts[vid]
    true_ans_l = tr_single_ans[vid]
    
    # Add 30% noise to combination prediction (pretend it's wrong 30% of time)
    if random.random() < 0.30:
        # Pick random wrong combination
        random_acts = {opts[random.choice(['A', 'B', 'C', 'D'])]}
        comb_acts = random_acts
    
    in_comb = [l for l, text in opts.items() if text in comb_acts]
    
    if len(in_comb) == 1:
        total_with_noise += 1
        if in_comb[0] == true_ans_l:
            correct_with_noise += 1

acc = correct_with_noise / total_with_noise if total_with_noise else 0
print(f"With 30% noise in combination: {correct_with_noise}/{total_with_noise} = {acc:.1%}")
