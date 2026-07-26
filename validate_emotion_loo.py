"""
Let's look at the emotion category more carefully.
We know emotion is NOT predictable from action set alone (11% match rate).
BUT: what if we look at the EXACT same option set in training for emotion?

From earlier: 18 test emotion questions have exact option-set matches in training.
The consistency within those was only 21% - meaning they ALSO disagree.
The question is: for the SPECIFIC emotion options we see in test,
what does training tell us?

Let me look at this from a different angle:
For training emotion questions with the SAME 4-action set as a test sequence question,
what emotion answer is most common?

Also: check if there's a structural leak in the sequence ordering for test videos
where the Markov chain and the training vote DISAGREE.
Specifically: test_0642 (Markov=DACB, TrainVote=DABC) - we used DABC.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

# Look at training emotion accuracy per option-set
tr_emo = tr[tr['category'] == 'emotion']
te_emo = te[te['category'] == 'emotion']

def get_opts_frozenset(row):
    return frozenset([str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']])

tr_emo_sets = {}
for idx, row in tr_emo.iterrows():
    fs = get_opts_frozenset(row)
    ans_l = str(row['answer']).strip()
    ans_text = str(row[ans_l]).strip().lower()
    if fs not in tr_emo_sets:
        tr_emo_sets[fs] = {}
    tr_emo_sets[fs][ans_text] = tr_emo_sets[fs].get(ans_text, 0) + 1

# Validate on training: LOO accuracy for emotion
correct = 0
total = 0
applied = 0
for idx, row in tr_emo.iterrows():
    fs = get_opts_frozenset(row)
    ans_l = str(row['answer']).strip()
    ans_text = str(row[ans_l]).strip().lower()
    
    if fs in tr_emo_sets and sum(tr_emo_sets[fs].values()) > 1:
        votes = dict(tr_emo_sets[fs])
        votes[ans_text] -= 1  # LOO
        if votes[ans_text] == 0:
            del votes[ans_text]
        
        if votes:
            best_text = max(votes, key=votes.get)
            best_votes = votes[best_text]
            total_votes = sum(votes.values())
            applied += 1
            if best_text == ans_text:
                correct += 1

print(f"Emotion LOO accuracy (when option-set matches): {correct}/{applied} ({correct/applied:.1%} if applied > 0)")

# Check test emotion vs training
changes = 0
for idx, row in te_emo.iterrows():
    fs = get_opts_frozenset(row)
    if fs in tr_emo_sets:
        votes = tr_emo_sets[fs]
        best_text = max(votes, key=votes.get)
        best_votes = votes[best_text]
        total = sum(votes.values())
        
        opts_rev = {str(row[l]).strip().lower(): l for l in ['A', 'B', 'C', 'D']}
        best_l = opts_rev.get(best_text, '?')
        pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
        
        if pred != best_l and best_votes >= 2 and best_votes / total >= 0.70:
            print(f"{row['qa_id']}: pred={pred}, match={best_l}({best_text}) {best_votes}/{total}")
            changes += 1

print(f"\nEmotion changes with >=70% conf: {changes}")
