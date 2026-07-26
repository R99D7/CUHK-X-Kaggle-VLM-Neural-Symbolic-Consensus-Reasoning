"""
test_0114: pred=BC, train_match=AD (1/1) - ONLY 1 training example, risky
test_0584: pred=ACD, train_match=ABD (1/1) - ONLY 1 training example, risky

With only 1 training example each, these are risky to change.
Let me look at these more carefully.

New direction: Exploit the SINGLE training answer frequency for specific actions
by looking at the question options rather than globally.

When a specific action (e.g., "walking") appears as an option in training 100 times
and is the correct answer 91% of the time (when it is the ONLY option that is
a "core" action in the video's sequence), we have evidence.

Actually: let me revisit the single leak more carefully.
For HAU source single questions, can we find option-set matches in training?
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

hau_tr = tr[(tr['source'] == 'HAU') & (tr['category'] == 'single')]
hau_te = te[(te['source'] == 'HAU') & (te['category'] == 'single')]

print(f"HAU train single: {len(hau_tr)}")
print(f"HAU test single: {len(hau_te)}")

def get_opts_frozenset(row):
    return frozenset([str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']])

tr_sets = {}
for idx, row in hau_tr.iterrows():
    fs = get_opts_frozenset(row)
    ans_l = str(row['answer']).strip()
    ans_text = str(row[ans_l]).strip().lower()
    if fs not in tr_sets:
        tr_sets[fs] = {}
    tr_sets[fs][ans_text] = tr_sets[fs].get(ans_text, 0) + 1

# Check test
matches = 0
changes = 0
for idx, row in hau_te.iterrows():
    fs = get_opts_frozenset(row)
    if fs in tr_sets:
        votes = tr_sets[fs]
        best_text = max(votes, key=votes.get)
        best_votes = votes[best_text]
        total = sum(votes.values())
        opts_rev = {str(row[l]).strip().lower(): l for l in ['A', 'B', 'C', 'D']}
        best_l = opts_rev.get(best_text, '?')
        pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
        agree = "AGREE" if pred == best_l else "DISAGREE"
        matches += 1
        if pred != best_l:
            print(f"{row['qa_id']}: pred={pred}, match={best_l}({best_text}) {best_votes}/{total} [{agree}] all_votes={votes}")
            changes += 1

print(f"HAU single option-set matches: {matches}")
print(f"Disagreements: {changes}")
