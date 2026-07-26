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

# Validate on TRAIN SET: for each score threshold, what is accuracy?
seq_tr = tr[tr['category'] == 'sequence']

for min_score in [0, 10, 20, 30, 50, 70, 100]:
    correct = 0
    applied = 0
    skipped_correct = 0
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
        
        if best_score >= min_score:
            applied += 1
            if best_perm == ans_letters:
                correct += 1
        else:
            # Would not apply markov, so skip (original model)
            pass
    
    acc = correct / applied if applied > 0 else 0
    print(f"min_score={min_score}: applied={applied}, correct={correct}, accuracy={acc:.2%}")
