"""
Deeper exploration: For the 'sequence' category test questions where the options
exactly match a training sequence, we can get a perfect answer!
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

cat = 'sequence'
tr_cat = tr[tr['category'] == cat]
te_cat = te[te['category'] == cat]

# For sequence, the option-set is the same (same 4 actions), 
# but we're looking for an exact match of OPTIONS (A,B,C,D labels map to same actions)
# Actually for sequence, a match means the same 4 actions appear in a different order!
# The answer tells us the order.

tr_sets = {}  # frozenset of 4 actions -> list of correct answers (letter sequences)
for idx, row in tr_cat.iterrows():
    fs = frozenset([str(row['A']).strip().lower(), str(row['B']).strip().lower(), 
                    str(row['C']).strip().lower(), str(row['D']).strip().lower()])
    ans_l = str(row['answer']).strip()
    # Map to actual chronological texts
    ordered = tuple([str(row[l]).strip().lower() for l in ans_l])
    if fs not in tr_sets:
        tr_sets[fs] = []
    tr_sets[fs].append({'letter_order': ans_l, 'text_order': ordered, 'A': str(row['A']).strip().lower(), 'B': str(row['B']).strip().lower(), 'C': str(row['C']).strip().lower(), 'D': str(row['D']).strip().lower()})

matches = 0
for idx, row in te_cat.iterrows():
    fs = frozenset([str(row['A']).strip().lower(), str(row['B']).strip().lower(), 
                    str(row['C']).strip().lower(), str(row['D']).strip().lower()])
    if fs in tr_sets:
        matches += 1
        # Map the training answer to test labels
        for tr_entry in tr_sets[fs]:
            # Map text order to test letters
            te_opts_rev = {str(row[l]).strip().lower(): l for l in ['A', 'B', 'C', 'D']}
            te_answer = ''.join([te_opts_rev[t] for t in tr_entry['text_order']])
            pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
            agree = "AGREE" if pred == te_answer else "DISAGREE"
            print(f"  {row['qa_id']}: pred={pred}, train_inferred={te_answer} [{agree}] (from train order={tr_entry['letter_order']} text={tr_entry['text_order']})")

print(f"Total sequence option-set matches: {matches}")
