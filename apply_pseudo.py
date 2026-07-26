"""
Apply pseudo labels to test set.
"""
import pandas as pd
import numpy as np

pseudo = pd.read_csv('pseudo_test_labels.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

ps_dict = dict(zip(pseudo['qa_id'], pseudo['answer']))
sub_dict = dict(zip(sub['qa_id'], sub['prediction']))

changes = 0
for idx, row in te.iterrows():
    q = row['qa_id']
    if q in ps_dict and not pd.isna(ps_dict[q]):
        p_ans = str(ps_dict[q]).strip()
        s_ans = str(sub_dict.get(q, '')).strip()
        
        # for multi and comb, sort
        if row['category'] in ['multi', 'combination']:
            p_ans = "".join(sorted(p_ans))
            s_ans = "".join(sorted(s_ans))
            
        if p_ans and p_ans != s_ans:
            print(f"{q} ({row['category']}): {s_ans} -> {p_ans} (from pseudo labels)")
            sub.loc[sub['qa_id'] == q, 'prediction'] = p_ans
            changes += 1

print(f"Total changes from pseudo labels: {changes}")
sub.to_csv('submission_with_pseudo.csv', index=False)
