"""
Now exhaustively find ALL remaining combination pairs where:
- The pair appears as the correct answer in training with >= 70% accuracy
- AND at least 2 occurrences as correct answer
- AND the current prediction doesn't already predict that option

This gives us a complete picture of what's left.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

tr_comb = tr[tr['category'] == 'combination']

# Per-pair accuracy
pair_as_ans = {}
pair_as_opt = {}
for idx, row in tr_comb.iterrows():
    ans_l = str(row['answer']).strip()
    for l in ['A', 'B', 'C', 'D']:
        opt_text = str(row[l]).strip().lower()
        acts = frozenset(a.strip() for a in opt_text.split(','))
        pair_as_opt[acts] = pair_as_opt.get(acts, 0) + 1
        if l == ans_l:
            pair_as_ans[acts] = pair_as_ans.get(acts, 0) + 1

# High-confidence pairs
high_conf_pairs = {}
for acts, ans_count in pair_as_ans.items():
    opt_count = pair_as_opt.get(acts, 0)
    acc = ans_count / opt_count
    if ans_count >= 2 and acc >= 0.70:
        high_conf_pairs[acts] = (ans_count, opt_count, acc)

print(f"High-confidence pairs (>=2 correct, >=70% acc): {len(high_conf_pairs)}")
for acts, (ans_c, opt_c, acc) in sorted(high_conf_pairs.items(), key=lambda x: -x[1][2]):
    print(f"  {set(acts)}: {ans_c}/{opt_c} ({acc:.0%})")

# Now scan test combination questions
te_comb = te[te['category'] == 'combination']
te_video_actions = {}
for idx, row in te[te['category'] == 'sequence'].iterrows():
    acts = frozenset([str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']])
    te_video_actions[row['path']] = acts

print("\n\nTest combinations where a high-conf pair != current prediction:")
for idx, row in te_comb.iterrows():
    vid = row['path']
    if vid in te_video_actions:
        known_acts = te_video_actions[vid]
        valid_opts = [l for l in ['A', 'B', 'C', 'D']
                      if set(x.strip() for x in str(row[l]).strip().lower().split(',')).issubset(known_acts)]
        if len(valid_opts) == 1:
            continue

    opts = {}
    for l in ['A', 'B', 'C', 'D']:
        opt_text = str(row[l]).strip().lower()
        acts = frozenset(a.strip() for a in opt_text.split(','))
        opts[l] = (opt_text, acts)

    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    pred_acts = opts[pred][1]
    pred_info = high_conf_pairs.get(pred_acts, None)

    for l, (opt_text, acts) in opts.items():
        if acts in high_conf_pairs and l != pred:
            ans_c, opt_c, acc = high_conf_pairs[acts]
            pred_ans_c = pair_as_ans.get(pred_acts, 0)
            pred_opt_c = pair_as_opt.get(pred_acts, 0)
            pred_acc = pred_ans_c/pred_opt_c if pred_opt_c else 0
            print(f"  {row['qa_id']}: pred={pred}({pred_ans_c}/{pred_opt_c}={pred_acc:.0%}), candidate={l}({ans_c}/{opt_c}={acc:.0%}) | {opt_text[:50]}")
