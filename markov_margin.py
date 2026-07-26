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

sub = pd.read_csv('submission_v237_SELF_LEAK.csv')

# Evaluate each disagreement with its score and margin
rows = []
for idx, row in te[te['category'] == 'sequence'].iterrows():
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    
    scores_all = {}
    for perm in permutations(['A', 'B', 'C', 'D']):
        seq_acts = [opts[l] for l in perm]
        s = score_seq_full(seq_acts)
        scores_all[''.join(perm)] = s
    
    best_perm = max(scores_all, key=scores_all.get)
    best_score = scores_all[best_perm]
    second_score = sorted(scores_all.values(), reverse=True)[1]
    margin = best_score - second_score
    
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    if pred != best_perm and best_score > 0:
        rows.append({'qa_id': row['qa_id'], 'pred': pred, 'markov': best_perm, 'score': best_score, 'margin': margin})

df = pd.DataFrame(rows).sort_values('score', ascending=False)
print(df.to_string())
print(f"\nTotal: {len(df)}")
