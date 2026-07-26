"""
For the 6 sequence questions NOT in training matches (not covered by TrainVote),
check if there's a Markov score threshold above which we should apply vs not.

The 20 sequence changes in current submission vs v237:
- Some are from TrainVote (high confidence)
- Some are pure Markov with varying scores

Let's separately identify which ones came from TrainVote vs pure Markov.
The TrainVote questions were: test_0330, test_0334, test_0336, test_0337, test_0341,
test_0342, test_0344, test_0345, test_0346, test_0347, test_0350, test_0351, test_0359,
test_0642, test_0644, test_0646, test_0648, test_0649

But v248 changes from v237 are: test_0332, test_0333, test_0335, test_0338, test_0339,
test_0340, test_0343, test_0348, test_0349, test_0352, test_0353, test_0354, test_0355,
test_0356, test_0357, test_0641, test_0642, test_0643, test_0645, test_0647

So the questions that changed are PURE MARKOV (not training vote).
The training vote agreed with v237 for those questions.

Now: the Markov chain has ~67% accuracy. Of the 20 changes:
- Expected ~13 correct, ~7 wrong 
- Net gain: +13 - 7 = +6 correct

Our actual gain from 0.52923 to 0.54678 = 0.01755 * 682 = 11.97 ≈ 12 extra correct.
With 23 total changes (20 seq + 2 obj + 1 single), if accuracy is 52% on seq
and ~100% on obj_interaction and single = 52% * 20 + 100% * 3 = 10.4 + 3 = 13.4 net...
That roughly matches!

Question: Can we improve the Markov quality?
The 6 low-score ones (score < 20): test_0354 (11), test_0355 (11), test_0643 (28), test_0645 (15)
These might be hurting us. Let's revert only the very low-confidence ones.
"""
import pandas as pd
from collections import defaultdict
from itertools import permutations

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')

# Build Full-Order Markov
transitions = defaultdict(int)
for idx, row in tr[tr['category'] == 'sequence'].iterrows():
    ans_letters = str(row['answer']).strip()
    ordered_actions = [str(row[l]).strip().lower() for l in ans_letters]
    for i in range(len(ordered_actions)):
        for j in range(i + 1, len(ordered_actions)):
            transitions[(ordered_actions[i], ordered_actions[j])] += 1

def score_seq_full(seq_acts):
    score = 0
    for i in range(len(seq_acts)):
        for j in range(i + 1, len(seq_acts)):
            score += transitions[(seq_acts[i], seq_acts[j])]
    return score

# Validate at different min_score thresholds for the DISAGREEMENT cases
# We need to know: among cases where Markov disagrees with v237,
# what is the accuracy of Markov vs v237 on the training set?
seq_tr = tr[tr['category'] == 'sequence']

# For each train seq question, check if Markov best != "visual model best" (simulated)
# We can't simulate v237 perfectly, but we can check Markov accuracy overall
for min_score in [0, 15, 20, 25, 30, 40, 50]:
    applied_correct = 0
    applied_total = 0
    for idx, row in seq_tr.iterrows():
        opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        ans_letters = str(row['answer']).strip()
        
        best_score = -1
        best_perm = None
        all_scores = {}
        for perm in permutations(['A', 'B', 'C', 'D']):
            seq_acts = [opts[l] for l in perm]
            s = score_seq_full(seq_acts)
            if s > best_score:
                best_score = s
                best_perm = ''.join(perm)
            all_scores[''.join(perm)] = s
        
        second_best = sorted(all_scores.values(), reverse=True)[1]
        margin = best_score - second_best
        
        if best_score >= min_score:
            applied_total += 1
            if best_perm == ans_letters:
                applied_correct += 1
    
    acc = applied_correct / applied_total if applied_total > 0 else 0
    print(f"min_score={min_score}: applied={applied_total}/{len(seq_tr)}, accuracy={acc:.2%}")
