"""
80.1% of the time, the SINGLE answer is one of the TWO actions in the COMBINATION answer!

This is a very strong cross-category leak. For test videos with BOTH single and combination:
- We know the combination answer (from our current prediction)
- The single answer should be one of the 2 combination actions 80.1% of the time
- So we can check which single options are IN the combination pair -> that's likely the answer

Validate this on training first: when 1 single option matches combination pair, is it right?
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

# Build training data
tr_single = {}
for idx, row in tr[tr['category'] == 'single'].iterrows():
    vid = row['path']
    ans_l = str(row['answer']).strip()
    if len(ans_l) == 1:
        tr_single[vid] = {'ans': str(row[ans_l]).strip().lower(),
                          'opts': {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}}

tr_comb = {}
for idx, row in tr[tr['category'] == 'combination'].iterrows():
    vid = row['path']
    ans_l = str(row['answer']).strip()
    if len(ans_l) == 1:
        ans_text = str(row[ans_l]).strip().lower()
        tr_comb[vid] = set(a.strip() for a in ans_text.split(','))

# Validate: for training videos with both, when exactly 1 single option is in comb pair
correct_1 = 0
total_1 = 0
correct_2 = 0
total_2 = 0
correct_0 = 0
total_0 = 0

for vid in set(tr_single.keys()) & set(tr_comb.keys()):
    single_info = tr_single[vid]
    comb_acts = tr_comb[vid]
    
    in_comb = [l for l, text in single_info['opts'].items() if text in comb_acts]
    
    if len(in_comb) == 1:
        total_1 += 1
        if in_comb[0] == tr_single[vid]['ans'] or single_info['opts'][in_comb[0]] == single_info['ans']:
            correct_1 += 1
    elif len(in_comb) == 2:
        total_2 += 1
        if any(single_info['opts'][l] == single_info['ans'] for l in in_comb):
            correct_2 += 1
    elif len(in_comb) == 0:
        total_0 += 1

print(f"When exactly 1 single option matches comb pair: {correct_1}/{total_1} ({correct_1/total_1:.1%} if total > 0)")
print(f"When exactly 2 single options match comb pair: {correct_2}/{total_2} ({correct_2/total_2:.1%} if total > 0)")
print(f"When 0 single options match comb pair: {total_0}")
