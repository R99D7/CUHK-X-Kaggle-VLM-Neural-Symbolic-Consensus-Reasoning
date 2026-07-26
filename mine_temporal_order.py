"""
Mine empirical temporal ordering rules (Action X always comes before Action Y) from sequence questions in training_qa.csv.
Then check if our v270 submission violates any of these strict temporal laws!
"""
import pandas as pd
from collections import defaultdict

tr = pd.read_csv("training_qa.csv")
order_counts = defaultdict(int) # (before_action, after_action) -> count

for idx, r in tr[tr['category'] == 'sequence'].iterrows():
    opts = {l: str(r[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    ans = str(r['answer']).strip().replace(' ', '').replace(',', '').replace('->', '')
    
    # Get ordered sequence of action words
    seq_words = []
    for char in ans:
        if char in opts:
            seq_words.append(opts[char])
            
    # Record all pairwise orderings (before, after) in this ground truth sequence
    for i in range(len(seq_words)):
        for j in range(i + 1, len(seq_words)):
            order_counts[(seq_words[i], seq_words[j])] += 1

# Analyze pairs to find one-way directed physical rules (where (A -> B) occurs >= 5 times and (B -> A) occurs 0 times)
strict_order_rules = {}
print("--- 100% STRICT TEMPORAL ORDERING LAWS IN TRAINING DATA (Occurred >= 5 times, 0 violations) ---")
all_actions = set([k[0] for k in order_counts.keys()] + [k[1] for k in order_counts.keys()])
for a in all_actions:
    for b in all_actions:
        if a == b: continue
        ab = order_counts.get((a, b), 0)
        ba = order_counts.get((b, a), 0)
        if ab >= 5 and ba == 0:
            strict_order_rules[(a, b)] = ab
            print(f"[{ab:2d} : {ba}] '{a}' ALWAYS before '{b}'")

# Now check our v270 submission for sequence ordering violations
print("\n--- Checking v270 Test Predictions Against Strict Temporal Laws ---")
sub = pd.read_csv("submission_v270_TRUE_SUMMIT.csv")
te = pd.read_csv("test_qa.csv")
sub_map = dict(zip(sub['qa_id'], sub['prediction']))

violations = []
for idx, r in te[te['category'] == 'sequence'].iterrows():
    qid = r['qa_id']
    opts = {l: str(r[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    pred = str(sub_map[qid]).strip().replace(' ', '').replace(',', '').replace('->', '')
    
    seq_words = [opts[char] for char in pred if char in opts]
    for i in range(len(seq_words)):
        for j in range(i + 1, len(seq_words)):
            act_before = seq_words[i]
            act_after = seq_words[j]
            # If the strict rule says act_after MUST come before act_before (i.e. we reversed it)
            if (act_after, act_before) in strict_order_rules:
                violations.append((qid, pred, act_after, act_before, strict_order_rules[(act_after, act_before)]))
                print(f"[TEMPORAL VIOLATION] {qid}: Predicted '{act_before}' before '{act_after}', but in training '{act_after}' is ALWAYS before '{act_before}' (ratio {strict_order_rules[(act_after, act_before)]}:0)!")

print(f"Total temporal order violations in v270: {len(violations)}")
