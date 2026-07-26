"""
HUGE DISCOVERY: HARn test single questions have NaN as option D!
This means for most HARn test single questions, there are only 3 valid options (A,B,C).
The answer is one of A, B, C.

Also: HARn training paths embed the action category.
The test paths are like 'LM_test_0001' which are NOT in training.
But: if we can match the question structure (3 options, one of which is the answer),
we can use option-set matching across train.

Let's check HARn test single option-set matches against HARn training.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

harn_tr = tr[tr['source'] == 'HARn']
harn_te = te[te['source'] == 'HARn']
harn_tr_single = harn_tr[harn_tr['category'] == 'single']
harn_te_single = harn_te[harn_te['category'] == 'single']

# For HARn single: the action IS in the path label. 
# e.g. 'HARn/16_Fold_clothes/user1/...' -> answer is "folding clothes"

# Build mapping from training: frozenset of options -> answer letter
def get_opts_frozenset_3(row):
    opts = set()
    for l in ['A', 'B', 'C', 'D']:
        v = str(row[l]).strip().lower()
        if v and v != 'nan':
            opts.add(v)
    return frozenset(opts)

tr_sets = {}
for idx, row in harn_tr_single.iterrows():
    fs = get_opts_frozenset_3(row)
    ans_l = str(row['answer']).strip()
    ans_text = str(row[ans_l]).strip().lower()
    if fs not in tr_sets:
        tr_sets[fs] = {}
    tr_sets[fs][ans_text] = tr_sets[fs].get(ans_text, 0) + 1

# Check test HARn single against training
print("HARn test single option-set matches:")
matches = 0
for idx, row in harn_te_single.iterrows():
    fs = get_opts_frozenset_3(row)
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    if fs in tr_sets:
        best_ans_text = max(tr_sets[fs], key=tr_sets[fs].get)
        total_votes = sum(tr_sets[fs].values())
        best_votes = tr_sets[fs][best_ans_text]
        
        opts_map = {}
        for l in ['A', 'B', 'C', 'D']:
            v = str(row[l]).strip().lower()
            if v and v != 'nan':
                opts_map[v] = l
        
        if best_ans_text in opts_map:
            best_letter = opts_map[best_ans_text]
            agree = "AGREE" if pred == best_letter else "DISAGREE"
            print(f"  {row['qa_id']}: pred={pred}, match={best_letter} ({best_ans_text}) {best_votes}/{total_votes} [{agree}] votes={tr_sets[fs]}")
            matches += 1

print(f"\nTotal HARn single option-set matches: {matches}")
print(f"Total HARn single test: {len(harn_te_single)}")

# Also: for HARn single with D=nan, print all
print("\nAll HARn test single questions:")
for idx, row in harn_te_single.iterrows():
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    d_val = str(row['D']).strip()
    print(f"  {row['qa_id']}: A={row['A']}, B={row['B']}, C={row['C']}, D={d_val} -> pred={pred}")
