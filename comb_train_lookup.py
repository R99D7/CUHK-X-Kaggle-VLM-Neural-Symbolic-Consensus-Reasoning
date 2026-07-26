"""
Fresh approach: Look at the combination questions more carefully.
In training, COMBINATION answers are ALWAYS a subset of the sequence actions (when they co-occur).
We know the sequence action set for 39 test videos.
For the COMBINATION questions on those same videos, we've already checked exact subset matching.

NEW ANGLE: For combination questions NOT on the same video as a sequence question,
can we use the QUESTION TEXT itself to infer the answer?

Also NEW: Check if any test COMBINATION options appear as a TRAINING answer
(i.e., the exact combination of 2 actions appears as a training answer).
This is a direct lookup that bypasses the visual model entirely.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission_v248_MARKOV_VOTE.csv')

# Build training combination answers as a set
tr_comb = tr[tr['category'] == 'combination']
tr_comb_answers = {}  # frozenset of actions -> vote counts for each pair

for idx, row in tr_comb.iterrows():
    ans_l = str(row['answer']).strip()
    ans_text = str(row[ans_l]).strip().lower()
    acts = frozenset(a.strip() for a in ans_text.split(','))
    tr_comb_answers[acts] = tr_comb_answers.get(acts, 0) + 1

print(f"Unique combination pairs in training: {len(tr_comb_answers)}")
print("Top 10 most common pairs:")
for pair, count in sorted(tr_comb_answers.items(), key=lambda x: -x[1])[:10]:
    print(f"  {pair}: {count} times")

# For each test combination question, check which options match training pairs
te_comb = te[te['category'] == 'combination']
print(f"\nTest combination questions: {len(te_comb)}")

changes = 0
for idx, row in te_comb.iterrows():
    opts = {}
    for l in ['A', 'B', 'C', 'D']:
        opt_text = str(row[l]).strip().lower()
        acts = frozenset(a.strip() for a in opt_text.split(','))
        opts[l] = (opt_text, acts)
    
    # Score each option by how many times its pair appeared in training
    scores = {l: tr_comb_answers.get(acts, 0) for l, (text, acts) in opts.items()}
    best_l = max(scores, key=scores.get)
    best_score = scores[best_l]
    
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    pred_score = scores[pred]
    
    if pred != best_l and best_score > pred_score and best_score > 0:
        print(f"{row['qa_id']}: pred={pred}({pred_score}), train_match={best_l}({best_score}) | {opts[best_l][0]}")
        changes += 1

print(f"\nTotal potential combination changes: {changes}")
