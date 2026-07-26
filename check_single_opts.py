"""
The object_interaction category has 88.9% consistency within option-set matches!
This is a VERY strong leak. Let's validate more carefully on test data.
Also explore single-category option-set matches which had 13 matches.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

def get_opts_frozenset(row):
    return frozenset([str(row['A']).strip().lower(), str(row['B']).strip().lower(), 
                      str(row['C']).strip().lower(), str(row['D']).strip().lower()])

# Check single category option-set matches and see if consistent
cat = 'single'
tr_cat = tr[tr['category'] == cat]
te_cat = te[te['category'] == cat]

tr_sets = {}
for idx, row in tr_cat.iterrows():
    fs = get_opts_frozenset(row)
    try:
        ans_l = str(row['answer']).strip()
        ans_text = str(row[ans_l]).strip().lower()
        if fs not in tr_sets:
            tr_sets[fs] = {}
        tr_sets[fs][ans_text] = tr_sets[fs].get(ans_text, 0) + 1
    except:
        pass

print("Single category option-set matches with training:")
for idx, row in te_cat.iterrows():
    fs = get_opts_frozenset(row)
    if fs in tr_sets:
        best_ans_text = max(tr_sets[fs], key=tr_sets[fs].get)
        total_votes = sum(tr_sets[fs].values())
        best_votes = tr_sets[fs][best_ans_text]
        confidence = best_votes / total_votes
        opts_map = {str(row[l]).strip().lower(): l for l in ['A', 'B', 'C', 'D']}
        best_letter = opts_map.get(best_ans_text, '?')
        pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
        agree = "AGREE" if pred == best_letter else "DISAGREE"
        print(f"  {row['qa_id']}: pred={pred}, option-match={best_letter} ({best_ans_text}) conf={confidence:.0%} [{agree}] votes={tr_sets[fs]}")
