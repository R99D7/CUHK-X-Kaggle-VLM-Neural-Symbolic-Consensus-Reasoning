"""
The cross-leak worked perfectly (0.55263 -> 0.57602, +16 correct).
Now find MORE cross-category leaks:

1. MULTI -> SINGLE: When exactly 1 single option matches the multi answer actions?
2. Apply cross-leak with ALL combination predictions (not just high-conf)
3. Can we reverse the leak? If single=X, which combination options contain X?
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')

# Check: MULTI -> SINGLE leak in training
tr_single_ans = {}
tr_single_opts = {}
for idx, row in tr[tr['category'] == 'single'].iterrows():
    vid = row['path']
    ans_l = str(row['answer']).strip()
    if len(ans_l) == 1:
        tr_single_ans[vid] = str(row[ans_l]).strip().lower()
        tr_single_opts[vid] = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}

tr_multi_acts = {}
for idx, row in tr[tr['category'] == 'multi'].iterrows():
    vid = row['path']
    ans_l = str(row['answer']).strip()
    acts = set()
    for l in ans_l:
        if l in ['A', 'B', 'C', 'D']:
            acts.add(str(row[l]).strip().lower())
    tr_multi_acts[vid] = acts

correct_1 = 0
total_1 = 0
for vid in set(tr_single_ans.keys()) & set(tr_multi_acts.keys()):
    multi_acts = tr_multi_acts[vid]
    opts = tr_single_opts[vid]
    true_ans = tr_single_ans[vid]
    in_multi = [l for l, text in opts.items() if text in multi_acts]
    if len(in_multi) == 1:
        total_1 += 1
        if opts[in_multi[0]] == true_ans:
            correct_1 += 1

print(f"MULTI->SINGLE: when 1 single option in multi answer: {correct_1}/{total_1} = {correct_1/total_1:.1%}")

# Also check: COMBINATION with ALL predictions (not just high conf)
# How many more single questions can we fix?
sub = pd.read_csv('submission.csv')
te_comb = te[te['category'] == 'combination']
te_single = te[te['category'] == 'single']

# Build all predicted combination acts (including lower confidence)
all_pred_comb_acts = {}
for idx, row in te_comb.iterrows():
    vid = row['path']
    pred_l = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    if len(pred_l) == 1:
        pred_text = str(row[pred_l]).strip().lower()
        acts = set(a.strip() for a in pred_text.split(','))
        if vid not in all_pred_comb_acts:
            all_pred_comb_acts[vid] = acts
        else:
            all_pred_comb_acts[vid] |= acts

additional_changes = 0
for idx, row in te_single.iterrows():
    vid = row['path']
    if vid not in all_pred_comb_acts:
        continue
    comb_acts = all_pred_comb_acts[vid]
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    in_comb = [l for l, text in opts.items() if text in comb_acts]
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    if len(in_comb) == 1 and pred not in in_comb:
        print(f"  {row['qa_id']}: pred={pred}({opts[pred]}), comb_cross={in_comb[0]}({opts[in_comb[0]]})")
        additional_changes += 1

print(f"\nAdditional single fixes using ALL comb predictions: {additional_changes}")
