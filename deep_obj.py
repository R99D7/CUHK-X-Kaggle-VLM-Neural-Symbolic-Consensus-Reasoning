"""
Focus: object_interaction has 88.9% consistency when option sets match in training.
Let's check ALL object_interaction test questions against training option sets
with a broader matching strategy.
Also: look at what the 21 object_interaction questions are.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

te_obj = te[te['category'] == 'object_interaction']
tr_obj = tr[tr['category'] == 'object_interaction']

print(f"Test object_interaction: {len(te_obj)}")
print(f"Train object_interaction: {len(tr_obj)}")

def get_opts_frozenset(row):
    opts = set()
    for l in ['A', 'B', 'C', 'D']:
        v = str(row[l]).strip().lower()
        if v and v != 'nan':
            opts.add(v)
    return frozenset(opts)

# Build training lookup
tr_sets = {}
for idx, row in tr_obj.iterrows():
    fs = get_opts_frozenset(row)
    ans_l = str(row['answer']).strip()
    ans_text = str(row[ans_l]).strip().lower()
    if fs not in tr_sets:
        tr_sets[fs] = {}
    tr_sets[fs][ans_text] = tr_sets[fs].get(ans_text, 0) + 1

print("\nAll test object_interaction questions:")
for idx, row in te_obj.iterrows():
    fs = get_opts_frozenset(row)
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    pred_text = str(row[pred]).strip().lower() if pred in ['A', 'B', 'C', 'D'] else '?'
    
    if fs in tr_sets:
        best_ans_text = max(tr_sets[fs], key=tr_sets[fs].get)
        opts_map = {str(row[l]).strip().lower(): l for l in ['A', 'B', 'C', 'D'] if str(row[l]).strip().lower() != 'nan'}
        best_letter = opts_map.get(best_ans_text, '?')
        agree = "AGREE" if pred == best_letter else "DISAGREE"
        print(f"  {row['qa_id']}: pred={pred} ({pred_text}), train_match={best_letter} ({best_ans_text}) [{agree}] votes={tr_sets[fs]}")
    else:
        print(f"  {row['qa_id']}: pred={pred} ({pred_text}) | A={row['A']}, B={row['B']}, C={row['C']}, D={row['D']} | NO MATCH")
