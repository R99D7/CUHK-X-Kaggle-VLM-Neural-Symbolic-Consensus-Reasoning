"""
Validate option-set leak on TRAINING DATA:
If we use the most-voted option from train (leave-one-out), how accurate is it?
"""
import pandas as pd
from sklearn.model_selection import LeaveOneOut

tr = pd.read_csv('training_qa.csv')

def get_opts_frozenset(row):
    return frozenset([str(row['A']).strip().lower(), str(row['B']).strip().lower(), 
                      str(row['C']).strip().lower(), str(row['D']).strip().lower()])

for cat in ['emotion', 'object_interaction']:
    tr_cat = tr[tr['category'] == cat].reset_index(drop=True)
    
    # Build sets
    all_sets = {}
    for i, row in tr_cat.iterrows():
        fs = get_opts_frozenset(row)
        try:
            ans_l = str(row['answer']).strip()
            ans_text = str(row[ans_l]).strip().lower()
            if fs not in all_sets:
                all_sets[fs] = []
            all_sets[fs].append(ans_text)
        except:
            pass
    
    # Count how often option sets repeat
    multi = {k: v for k, v in all_sets.items() if len(v) > 1}
    print(f"\n{cat}: {len(multi)} option-sets appear multiple times")
    
    # Check consistency within repeated option sets
    consistent = 0
    total = 0
    for fs, answers in multi.items():
        unique_ans = set(answers)
        if len(unique_ans) == 1:
            consistent += len(answers)
        total += len(answers)
    
    print(f"  Answers within repeated sets - consistent: {consistent}/{total} ({consistent/total:.1%})")
