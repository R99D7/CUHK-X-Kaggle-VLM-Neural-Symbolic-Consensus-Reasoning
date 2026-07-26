"""
Validate the global frequency approach: if we always pick the most common action
from training as our answer for single questions, what would be the train accuracy?
This tells us if "walking" (113 times) being the most frequent answer is actually correct.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')

tr_single = tr[tr['category'] == 'single']

# Build global answer frequency
tr_answer_freq = {}
for idx, row in tr_single.iterrows():
    ans_l = str(row['answer']).strip()
    ans_text = str(row[ans_l]).strip().lower()
    tr_answer_freq[ans_text] = tr_answer_freq.get(ans_text, 0) + 1

# LOO validation: pick highest freq option
correct = 0
total = 0
correct_2x = 0
total_2x = 0

for idx, row in tr_single.iterrows():
    ans_l = str(row['answer']).strip()
    ans_text = str(row[ans_l]).strip().lower()
    
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    scores = {l: tr_answer_freq.get(text, 0) for l, text in opts.items()}
    
    # LOO: subtract 1 from correct answer
    scores[ans_l] -= 1
    
    best_l = max(scores, key=scores.get)
    best_score = scores[best_l]
    pred_score = scores[ans_l] + 1  # restore
    
    total += 1
    if best_l == ans_l:
        correct += 1
    
    # Only when best has 2x advantage
    if best_score >= 5 and best_score > (scores.get(ans_l, 0) + 1) * 2:
        total_2x += 1
        if best_l == ans_l:
            correct_2x += 1

print(f"Global freq lookup accuracy on train (all): {correct}/{total} ({correct/total:.1%})")
print(f"Global freq lookup when 2x advantage: {correct_2x}/{total_2x} ({correct_2x/total_2x:.1%} if total > 0)")
print(f"Baseline (random): 25%")
