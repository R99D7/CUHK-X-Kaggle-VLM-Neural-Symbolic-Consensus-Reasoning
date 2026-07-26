"""
The exhaustive scan reveals that all remaining candidates where our prediction
DISAGREES with a high-conf pair are cases where our CURRENT prediction ALSO has
a high-conf pair with equal or higher accuracy! These are conflicts.

Let me check each conflict case to decide which to trust more:
- test_0226: pred=B (2/2=100%) vs A (2/2=100%) - TIE -> keep pred
- test_0249: pred=C (10/12=83%) vs B (4/5=80%) - pred slightly better -> keep
- test_0278: pred=B (7/7=100%) vs D (7/9=78%) - pred better -> keep
- test_0324: pred=B (12/12=100%) vs C (4/5=80%) - pred better -> keep
- test_0325: pred=D (9/9=100%) vs A (3/4=75%) - pred better -> keep
- test_0627: pred=B (10/10=100%) vs A (4/5=80%) - pred better -> keep
- test_0636: pred=B (7/7=100%) vs A (7/9=78%) - pred better -> keep

CONCLUSION: All combination candidates are exhausted! Our predictions are already
using the best-performing pairs for all the cases we can confirm.

Now let's look at the SINGLE category using a different approach:
For single questions on the same video as sequence (known 4 actions),
the answer should be ONE of those 4 actions. But we already apply this.

New idea: Look at the MULTI category.
For multi questions NOT on the same video as sequence,
use the training multi answer frequency by action text.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

# For multi questions: what is the distribution of how many options are selected?
tr_multi = tr[tr['category'] == 'multi']
te_multi = te[te['category'] == 'multi']
sub_multi = sub[sub['qa_id'].isin(te_multi['qa_id'])]

print("Current test multi predictions:")
print(sub_multi.merge(te_multi[['qa_id', 'category']], on='qa_id')['prediction'].value_counts())

# Check: for training multi questions, is there a pattern for specific option sets?
tr_multi_sets = {}
for idx, row in tr_multi.iterrows():
    fs = frozenset([str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']])
    ans = str(row['answer']).strip()
    if fs not in tr_multi_sets:
        tr_multi_sets[fs] = {}
    tr_multi_sets[fs][ans] = tr_multi_sets[fs].get(ans, 0) + 1

# Find test multi questions that match training option sets exactly
changes = 0
for idx, row in te_multi.iterrows():
    fs = frozenset([str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']])
    if fs in tr_multi_sets:
        votes = tr_multi_sets[fs]
        best_ans = max(votes, key=votes.get)
        best_votes = votes[best_ans]
        total_votes = sum(votes.values())
        
        pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
        agree = "AGREE" if pred == best_ans else "DISAGREE"
        
        if pred != best_ans:
            print(f"{row['qa_id']}: pred={pred}, train_match={best_ans} ({best_votes}/{total_votes}) [{agree}] all={votes}")
            changes += 1

print(f"\nTotal multi option-set matches with disagreement: {changes}")
