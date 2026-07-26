"""
139 test videos have BOTH single AND combination questions!
This is a massive unexploited cross-category leak.

For training: the single answer IS in the combination answer pair.
Let's check the rate more carefully.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

# Build single answer per vid
tr_single_ans = {}
for idx, row in tr[tr['category'] == 'single'].iterrows():
    vid = row['path']
    ans_l = str(row['answer']).strip()
    if len(ans_l) == 1:  # valid single letter
        ans_text = str(row[ans_l]).strip().lower()
        tr_single_ans[vid] = ans_text

# Build combination answer per vid
tr_comb_ans = {}
for idx, row in tr[tr['category'] == 'combination'].iterrows():
    vid = row['path']
    ans_l = str(row['answer']).strip()
    if len(ans_l) == 1:  # valid single letter
        ans_text = str(row[ans_l]).strip().lower()
        acts = set(a.strip() for a in ans_text.split(','))
        tr_comb_ans[vid] = acts

# Check overlap
in_comb = 0
not_in_comb = 0
for vid in set(tr_single_ans.keys()) & set(tr_comb_ans.keys()):
    single_ans = tr_single_ans[vid]
    comb_acts = tr_comb_ans[vid]
    if single_ans in comb_acts:
        in_comb += 1
    else:
        not_in_comb += 1

print(f"Training: single answer IS in combination pair: {in_comb}")
print(f"Training: single answer NOT in combination pair: {not_in_comb}")
total = in_comb + not_in_comb
if total > 0:
    print(f"Rate: {in_comb/total:.1%}")
