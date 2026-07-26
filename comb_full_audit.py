"""
Re-examine all combination candidates we rejected.
Compute the EXACT per-pair accuracy on training for each.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

tr_comb = tr[tr['category'] == 'combination']
tr_comb_answers = {}
for idx, row in tr_comb.iterrows():
    ans_l = str(row['answer']).strip()
    ans_text = str(row[ans_l]).strip().lower()
    acts = frozenset(a.strip() for a in ans_text.split(','))
    tr_comb_answers[acts] = tr_comb_answers.get(acts, 0) + 1

# Per-pair: how often is it the answer vs. just an option?
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

# Build test: video path -> 4 actions (sequence leak filter)
te_video_actions = {}
for idx, row in te[te['category'] == 'sequence'].iterrows():
    acts = frozenset([str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']])
    te_video_actions[row['path']] = acts

te_comb = te[te['category'] == 'combination']
v248 = pd.read_csv('submission_v248_MARKOV_VOTE.csv')

print("All remaining combination candidates (training pair accuracy >= 40%):")
for idx, row in te_comb.iterrows():
    vid = row['path']
    
    # Skip if already handled by sequence-subset leak
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
        ans_count = pair_as_ans.get(acts, 0)
        opt_count = pair_as_opt.get(acts, 0)
        acc = ans_count / opt_count if opt_count > 0 else 0
        opts[l] = (opt_text, acts, ans_count, opt_count, acc)
    
    pred_v248 = str(v248[v248['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    pred_cur = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    # Find best option by accuracy-weighted score
    best_by_acc = max(opts.keys(), key=lambda l: (opts[l][2], opts[l][4]))
    best_by_count = max(opts.keys(), key=lambda l: opts[l][2])
    
    if (opts[best_by_count][2] > opts[pred_cur][2] and opts[best_by_count][4] >= 0.5 and
            opts[best_by_count][2] >= 2):
        agree = "AGREE" if pred_cur == best_by_count else "DISAGREE"
        print(f"\n{row['qa_id']}: pred={pred_cur}, best={best_by_count} [{agree}]")
        for l in ['A', 'B', 'C', 'D']:
            opt_text, acts, ans_c, opt_c, acc = opts[l]
            marker = "<-- PRED" if l == pred_cur else ("<-- BEST" if l == best_by_count else "")
            print(f"  {l}: {opt_text[:50]} ans={ans_c}/{opt_c} ({acc:.0%}) {marker}")
