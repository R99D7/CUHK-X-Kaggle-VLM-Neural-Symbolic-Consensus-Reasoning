"""
All main categories are now largely exhausted with the strategies we have.

Let me try one more angle: for SINGLE questions that are on a video with
a known COMBINATION answer (from test_qa.csv), we can infer which 2 actions
are in the video. If a single question on the same video has one of those
2 known actions as an option, that's the answer.

Wait, actually let me check: do test videos have both COMBINATION and SINGLE questions?
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

# Check which test videos have multiple categories
cats_per_vid = te.groupby('path')['category'].apply(list)

# Find videos with both single and combination
both = cats_per_vid[cats_per_vid.apply(lambda x: 'single' in x and 'combination' in x)]
print(f"Test videos with both single and combination: {len(both)}")

# For HAU test: find combinations where the correct pair can tell us the single action
# In training: do single and combination answers correlate for same video?
tr_cats = tr.groupby('path')['category'].apply(list)
both_tr = tr_cats[tr_cats.apply(lambda x: 'single' in x and 'combination' in x)]
print(f"Train videos with both single and combination: {len(both_tr)}")

# For training: when video has both single and combination,
# is the single answer one of the actions in the combination answer?
tr_single_ans = {}
tr_comb_ans = {}
for idx, row in tr.iterrows():
    vid = row['path']
    ans_l = str(row['answer']).strip()
    ans_text = str(row[ans_l]).strip().lower()
    if row['category'] == 'single':
        tr_single_ans[vid] = ans_text
    elif row['category'] == 'combination':
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

print(f"\nTraining: single answer IS in combination answer: {in_comb}")
print(f"Training: single answer is NOT in combination answer: {not_in_comb}")
if in_comb + not_in_comb > 0:
    print(f"Rate: {in_comb/(in_comb+not_in_comb):.1%}")
