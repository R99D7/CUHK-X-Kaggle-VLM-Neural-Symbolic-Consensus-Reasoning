"""
Final comprehensive analysis: estimate how many more we can potentially get right.
Current score: 0.54678 -> ~373 correct out of 682.
Remaining: ~309 wrong.

Let's check accuracy by category for what we have vs train baseline.
Also: check if the sequence Markov ordering for questions WITHOUT training matches
could be improved using a different approach (e.g. only apply when score > threshold).
"""
import pandas as pd
from collections import defaultdict
from itertools import permutations

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')
v237 = pd.read_csv('submission_v237_SELF_LEAK.csv')

# Count changes from v237 by category
te_merged = te.merge(sub, on='qa_id').merge(v237, on='qa_id', suffixes=('_v248', '_v237'))
diffs = te_merged[te_merged['prediction_v248'] != te_merged['prediction_v237']]
print("Changes from v237 baseline by category:")
print(diffs['category'].value_counts())

print("\nSequence changes from v237:")
print(diffs[diffs['category'] == 'sequence'][['qa_id', 'prediction_v237', 'prediction_v248']].to_string())

# Build Full-Order Markov
transitions = defaultdict(int)
for idx, row in tr[tr['category'] == 'sequence'].iterrows():
    ans_letters = str(row['answer']).strip()
    ordered_actions = [str(row[l]).strip().lower() for l in ans_letters]
    for i in range(len(ordered_actions)):
        for j in range(i + 1, len(ordered_actions)):
            transitions[(ordered_actions[i], ordered_actions[j])] += 1

def score_seq_full(seq_acts):
    score = 0
    for i in range(len(seq_acts)):
        for j in range(i + 1, len(seq_acts)):
            score += transitions[(seq_acts[i], seq_acts[j])]
    return score

# Check current sequence predictions
print("\n\nAll sequence questions and predictions vs v237:")
for idx, row in te[te['category'] == 'sequence'].iterrows():
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    
    best_score = -1
    best_perm = None
    for perm in permutations(['A', 'B', 'C', 'D']):
        seq_acts = [opts[l] for l in perm]
        s = score_seq_full(seq_acts)
        if s > best_score:
            best_score = s
            best_perm = ''.join(perm)
    
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    v237_pred = str(v237[v237['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    if pred != v237_pred:
        print(f"  {row['qa_id']}: v237={v237_pred} -> v248={pred} (markov={best_perm}, score={best_score})")
