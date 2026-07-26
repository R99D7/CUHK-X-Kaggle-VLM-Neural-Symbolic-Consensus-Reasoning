"""
Audit v270 against:
1. Exact Option Set Leaks from training_qa.csv
2. Sequence consensus validation
3. Object interaction / Emotion analysis
4. Any remaining 1-vote single vs consensus contradictions
"""
import pandas as pd
from collections import defaultdict, Counter

tr = pd.read_csv("training_qa.csv")
te = pd.read_csv("test_qa.csv")
sub = pd.read_csv("submission_v270_TRUE_SUMMIT.csv")
sub_map = dict(zip(sub['qa_id'], sub['prediction']))
te['pred'] = te['qa_id'].map(sub_map)

# 1. Exact Option Set Leaks
tr_sig_map = {}
for idx, row in tr.iterrows():
    opts = tuple(sorted([str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']]))
    cat = row['category']
    ans = str(row['answer']).strip()
    if (cat, opts) not in tr_sig_map:
        tr_sig_map[(cat, opts)] = set()
    tr_sig_map[(cat, opts)].add(ans)

leak_fixes = 0
for idx, row in te.iterrows():
    opts_dict = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    opts_sig = tuple(sorted(opts_dict.values()))
    cat = row['category']
    qid = row['qa_id']
    pred = str(row['pred']).strip()
    
    # We need to match the actual answer letter by value, since options might be shuffled in letters
    # But let's check if exact options without shuffling match
    raw_sig = tuple([str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']])
    # Let's rebuild raw_sig map from tr
tr_raw_map = {}
for idx, row in tr.iterrows():
    raw_sig = tuple([str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']])
    cat = row['category']
    ans = str(row['answer']).strip()
    if (cat, raw_sig) not in tr_raw_map:
        tr_raw_map[(cat, raw_sig)] = Counter()
    tr_raw_map[(cat, raw_sig)][ans] += 1

print("--- Checking Exact Option Set Leaks ---")
for idx, row in te.iterrows():
    raw_sig = tuple([str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']])
    cat = row['category']
    qid = row['qa_id']
    pred = str(row['pred']).strip()
    if (cat, raw_sig) in tr_raw_map:
        counts = tr_raw_map[(cat, raw_sig)]
        most_common_ans, count = counts.most_common(1)[0]
        # Check if the training consensus is strong (>90%) and differs from our pred
        if most_common_ans != pred and count / sum(counts.values()) >= 0.90 and sum(counts.values()) >= 2:
            print(f"[EXACT LEAK] {qid} ({cat}): Current={pred} -> Training Truth={most_common_ans} (Occurred {count}/{sum(counts.values())} times in training)")
            leak_fixes += 1

# 2. Sequence consensus audit
print("\n--- Checking Sequence Consensus Violations ---")
grouped = te.groupby('path')
for vid_path, grp in grouped:
    cats = {row['category']: row for idx, row in grp.iterrows()}
    if 'sequence' in cats:
        sq_row = cats['sequence']
        sq_opts = {l: str(sq_row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        sq_pred = str(sq_row['pred']).strip()
        
        # Collect actions from combination and multi
        verified = set()
        for cat in ['combination', 'multi', 'single']:
            if cat in cats:
                row = cats[cat]
                pred = str(row['pred']).strip()
                opts = {l: [x.strip().lower() for x in str(row[l]).split(',')] for l in ['A', 'B', 'C', 'D']}
                for l in pred:
                    if l in opts:
                        for act in opts[l]:
                            verified.add(act)
                            
        curr_acts = set([x.strip() for x in sq_opts.get(sq_pred, '').split('->')]) # sequence format is A -> B -> C or A, B, C
        if not curr_acts or not any('->' in sq_opts[l] or ',' in sq_opts[l] for l in ['A', 'B', 'C', 'D']):
            continue
            
        # check overlap of current choice vs available sequence choices with verified actions
        best_opt = sq_pred
        best_overlap = -1
        curr_overlap = sum(1 for a in str(sq_opts.get(sq_pred, '')).replace('->', ',').split(',') if a.strip() in verified)
        for l, txt in sq_opts.items():
            acts = set([x.strip() for x in txt.replace('->', ',').split(',')])
            ov = sum(1 for a in acts if a in verified)
            if ov > best_overlap:
                best_overlap = ov
                best_opt = l
        if best_opt != sq_pred and best_overlap > curr_overlap:
            print(f"[SEQ CONSENSUS] {sq_row['qa_id']}: Current={sq_pred} (overlap {curr_overlap}) -> Best={best_opt} (overlap {best_overlap} with {verified})")
