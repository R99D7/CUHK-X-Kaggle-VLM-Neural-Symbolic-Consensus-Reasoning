import pandas as pd
from collections import defaultdict
from itertools import permutations

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
v237 = pd.read_csv('submission_v237_SELF_LEAK.csv')

transitions = defaultdict(int)
for idx, row in tr[tr['category'] == 'sequence'].iterrows():
    ans_letters = str(row['answer']).strip()
    ordered_actions = [str(row[l]).strip().lower() for l in ans_letters]
    for i in range(len(ordered_actions) - 1):
        transitions[(ordered_actions[i], ordered_actions[i+1])] += 1

def score_seq(seq_acts):
    score = 0
    for i in range(len(seq_acts) - 1):
        score += transitions[(seq_acts[i], seq_acts[i+1])]
    return score

diffs = 0
for idx, row in te[te['category'] == 'sequence'].iterrows():
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    
    best_score = -1
    best_perm = None
    
    for perm in permutations(['A', 'B', 'C', 'D']):
        seq_acts = [opts[l] for l in perm]
        s = score_seq(seq_acts)
        if s > best_score:
            best_score = s
            best_perm = ''.join(perm)
            
    pred = str(v237[v237['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    if pred != best_perm:
        print(f"{row['qa_id']}: v237={pred}, markov={best_perm} (score: {best_score})")
        diffs += 1

print(f'Markov chain disagrees with v237 on {diffs} sequence questions')
