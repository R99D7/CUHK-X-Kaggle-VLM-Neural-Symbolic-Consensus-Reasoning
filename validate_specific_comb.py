"""
Validate the 7 combination changes we applied on the training set.
For each specific pair we applied, check training consistency.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')

# The 7 changes applied:
# test_0244: D = "wiping hands, massaging oneself"  (7 times in train)
# test_0271: D = "checking body temperature, getting dressed" (3 times)
# test_0296: A = "mopping, walking" (4 times)
# test_0303: A = "taking a selfie, walking" (10 times)
# test_0609: C = "combing hair, checking the time" (3 times)
# test_0618: A = "walking, squats" (5 times) [NOT in our original train lookup!]
# test_0622: A = "taking a selfie, walking" (10 times)

pairs_to_check = [
    "wiping hands, massaging oneself",
    "checking body temperature, getting dressed",
    "mopping, walking",
    "taking a selfie, walking",
    "combing hair, checking the time",
    "walking, squats",
]

tr_comb = tr[tr['category'] == 'combination']

for pair_text in pairs_to_check:
    acts = frozenset(a.strip() for a in pair_text.split(','))
    count = 0
    total_as_opt = 0
    for idx, row in tr_comb.iterrows():
        for l in ['A', 'B', 'C', 'D']:
            opt_text = str(row[l]).strip().lower()
            opt_acts = frozenset(a.strip() for a in opt_text.split(','))
            if opt_acts == acts:
                total_as_opt += 1
                ans_l = str(row['answer']).strip()
                if l == ans_l:
                    count += 1
    acc = count / total_as_opt if total_as_opt > 0 else 0
    print(f"'{pair_text}': correct {count}/{total_as_opt} ({acc:.0%})")
