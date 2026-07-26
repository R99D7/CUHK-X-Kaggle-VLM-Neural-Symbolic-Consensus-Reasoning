"""
For sequence questions that have multiple training matches, take a MAJORITY VOTE
on the chronological ordering (text-based). This is stronger than Markov alone.
"""
import pandas as pd
from collections import defaultdict, Counter
from itertools import permutations

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

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

# Build option-set -> text_orders from training
tr_cat = tr[tr['category'] == 'sequence']
tr_seq_sets = {}  # frozenset of 4 actions -> list of text orders
for idx, row in tr_cat.iterrows():
    fs = frozenset([str(row['A']).strip().lower(), str(row['B']).strip().lower(), 
                    str(row['C']).strip().lower(), str(row['D']).strip().lower()])
    ans_l = str(row['answer']).strip()
    ordered = tuple([str(row[l]).strip().lower() for l in ans_l])
    if fs not in tr_seq_sets:
        tr_seq_sets[fs] = []
    tr_seq_sets[fs].append(ordered)

changes = 0
for idx, row in te[te['category'] == 'sequence'].iterrows():
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    opts_rev = {v: k for k, v in opts.items()}  # text -> letter
    fs = frozenset(opts.values())
    
    # Get Markov best
    best_score = -1
    best_perm = None
    for perm in permutations(['A', 'B', 'C', 'D']):
        seq_acts = [opts[l] for l in perm]
        s = score_seq_full(seq_acts)
        if s > best_score:
            best_score = s
            best_perm = ''.join(perm)
    
    final_answer = best_perm if best_score > 0 else None
    
    # If we have training matches, do majority vote
    if fs in tr_seq_sets:
        orders = tr_seq_sets[fs]
        order_counts = Counter(orders)
        most_common_text_order = order_counts.most_common(1)[0][0]
        vote_count = order_counts.most_common(1)[0][1]
        
        # Map text order back to test letters
        te_answer = ''.join([opts_rev[t] for t in most_common_text_order])
        
        # Use training vote if it has majority (>50%) or more than 1 match
        if vote_count >= 2 or (len(orders) == 1 and vote_count == 1):
            total = len(orders)
            confidence = vote_count / total
            final_answer = te_answer
            print(f"{row['qa_id']}: Markov={best_perm}, TrainVote={te_answer} (conf={confidence:.0%}, {vote_count}/{total})")
    
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    if final_answer and pred != final_answer:
        sub.loc[sub['qa_id'] == row['qa_id'], 'prediction'] = final_answer
        changes += 1

print(f"\nApplied {changes} sequence fixes (Markov + Train Vote).")
sub.to_csv('submission_v248_MARKOV_VOTE.csv', index=False)
sub.to_csv('submission.csv', index=False)
print('Saved to submission.csv')
