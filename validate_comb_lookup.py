"""
Validate the combination training lookup on the TRAINING set itself.
When option A has higher training pair frequency than the actual answer,
is the prediction correct?
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')

tr_comb = tr[tr['category'] == 'combination']
tr_comb_answers = {}

for idx, row in tr_comb.iterrows():
    ans_l = str(row['answer']).strip()
    ans_text = str(row[ans_l]).strip().lower()
    acts = frozenset(a.strip() for a in ans_text.split(','))
    tr_comb_answers[acts] = tr_comb_answers.get(acts, 0) + 1

# Leave-one-out validation: for each train comb question,
# remove it, then score options by freq, check if top-scored matches answer
correct_with_lookup = 0
total_lookup_applied = 0
correct_without_lookup = 0
total = 0

for idx, row in tr_comb.iterrows():
    ans_l = str(row['answer']).strip()
    ans_text = str(row[ans_l]).strip().lower()
    ans_acts = frozenset(a.strip() for a in ans_text.split(','))
    
    opts = {}
    for l in ['A', 'B', 'C', 'D']:
        opt_text = str(row[l]).strip().lower()
        acts = frozenset(a.strip() for a in opt_text.split(','))
        # Remove current answer from count (leave-one-out)
        count = tr_comb_answers.get(acts, 0)
        if acts == ans_acts:
            count -= 1  # leave-one-out
        opts[l] = (opt_text, acts, count)
    
    scores = {l: count for l, (text, acts, count) in opts.items()}
    best_l = max(scores, key=scores.get)
    best_score = scores[best_l]
    
    total += 1
    if best_score > 0:
        total_lookup_applied += 1
        if best_l == ans_l:
            correct_with_lookup += 1

acc = correct_with_lookup / total_lookup_applied if total_lookup_applied > 0 else 0
print(f"Train combo lookup applied: {total_lookup_applied}/{total}")
print(f"Correct when lookup applied: {correct_with_lookup} ({acc:.1%})")
print(f"Random baseline: 25%")
