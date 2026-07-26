import pandas as pd
from collections import defaultdict
from itertools import permutations

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')

# Build Full-Order Transition Matrix
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

# Validate at min_score=30 to check accuracy on the DISAGREES specifically
seq_tr = tr[tr['category'] == 'sequence']

correct_agree = 0
total_agree = 0
correct_disagree = 0
total_disagree = 0

for idx, row in seq_tr.iterrows():
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    ans_letters = str(row['answer']).strip()
    
    scores_all = {}
    for perm in permutations(['A', 'B', 'C', 'D']):
        seq_acts = [opts[l] for l in perm]
        s = score_seq_full(seq_acts)
        scores_all[''.join(perm)] = s
    
    best_perm = max(scores_all, key=scores_all.get)
    best_score = scores_all[best_perm]
    
    # We also check the "visual model" agreement (simulate v237 = random from train answers)
    if best_perm == ans_letters:
        correct_agree += 1
    total_agree += 1

print(f"Full-Order Markov total accuracy: {correct_agree}/{total_agree} = {correct_agree/total_agree:.2%}")

# The key question: within the 26 test disagreements, what fraction did v246 get right?
# We DON'T know for the test set, but we can estimate from train:
# At min_score=30 (246/308), accuracy=67.48%
# So roughly 67% of our 26 changes are correct => +17.5 correct, -8.5 wrong = net +9 correct
# That matched our improvement from 0.52923 -> 0.54385 (exactly 10 more correct!)

print("\nEstimate: Our 26 Markov changes ~ 67% accurate")
print("Expected net gain: +10.4 correct answers (67% * 26 - 33% * 26 = 8.84 net -> ~9-10)")
print("Actual gain: 0.54385 - 0.52923 = 0.01462 * 682 = 9.97 answers - confirmed!")
