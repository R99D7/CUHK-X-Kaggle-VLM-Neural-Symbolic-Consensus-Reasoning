import pandas as pd
from collections import defaultdict
from itertools import permutations

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission_v237_SELF_LEAK.csv')

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

changes = 0
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
            
    if best_score > 0:
        pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
        if pred != best_perm:
            print(f"{row['qa_id']}: v237={pred}, full_markov={best_perm} (score: {best_score})")
            sub.loc[sub['qa_id'] == row['qa_id'], 'prediction'] = best_perm
            changes += 1

print(f'Applied {changes} Full-Order Markov chain fixes.')

sub.to_csv('submission_v246_FULL_MARKOV.csv', index=False)
sub.to_csv('submission.csv', index=False)
print('Saved to submission.csv')
