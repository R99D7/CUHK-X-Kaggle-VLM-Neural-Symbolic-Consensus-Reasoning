import pandas as pd
from itertools import combinations
from collections import defaultdict

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')

video_all_actions = {}
for idx, row in tr.iterrows():
    vid = row['path']
    ans_letters = str(row['answer']).strip()
    try:
        if row['category'] == 'sequence': ans_texts = tuple([str(row[l]).strip().lower() for l in ans_letters])
        elif row['category'] == 'multi': ans_texts = frozenset([str(row[l]).strip().lower() for l in ans_letters])
        else: ans_texts = str(row[ans_letters]).strip().lower()
        actions = set()
        if isinstance(ans_texts, tuple) or isinstance(ans_texts, frozenset):
            for t in ans_texts: actions.add(t)
        else:
            for t in ans_texts.split(','): actions.add(t.strip())
        if vid not in video_all_actions: video_all_actions[vid] = set()
        video_all_actions[vid].update(actions)
    except: pass

cooccur = defaultdict(int)
for acts in video_all_actions.values():
    for a1, a2 in combinations(sorted(list(acts)), 2):
        cooccur[(a1, a2)] += 1

sub = pd.read_csv('submission.csv') # v243 which scored 0.54093

# 1. Fix combination questions
comb_changes = 0
for idx, row in te[te['category'] == 'combination'].iterrows():
    opts = {str(row[l]).strip().lower(): l for l in ['A', 'B', 'C', 'D']}
    
    scores = {}
    for opt_text, letter in opts.items():
        acts = [x.strip() for x in opt_text.split(',')]
        if len(acts) == 2:
            pair = tuple(sorted(acts))
            scores[letter] = cooccur[pair]
        else:
            scores[letter] = 0
            
    best_letter = max(scores, key=scores.get)
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    if pred != best_letter and scores[best_letter] > scores[pred]:
        sub.loc[sub['qa_id'] == row['qa_id'], 'prediction'] = best_letter
        comb_changes += 1

# 2. Fix multi questions with 0 co-occurrence
probs = pd.read_csv('transformer_fixed_raw_predictions.csv')
multi_changes = 0
for idx, row in te[te['category'] == 'multi'].iterrows():
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    pred_letters = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    if len(pred_letters) >= 2:
        pred_acts = [opts[l] for l in pred_letters]
        pred_score = 0
        for a1, a2 in combinations(sorted(pred_acts), 2):
            pred_score += cooccur[(a1, a2)]
            
        if pred_score == 0:
            # Overwrite with highest probability single letter
            p_row = probs[probs['qa_id'] == row['qa_id']].iloc[0]
            letter_probs = {'A': p_row['raw_prob_A'], 'B': p_row['raw_prob_B'], 'C': p_row['raw_prob_C'], 'D': p_row['raw_prob_D']}
            best_l = max(letter_probs, key=letter_probs.get)
            sub.loc[sub['qa_id'] == row['qa_id'], 'prediction'] = best_l
            multi_changes += 1

print(f"Applied {comb_changes} combination fixes and {multi_changes} multi fixes.")
sub.to_csv('submission_v244_FINAL_COOCCUR.csv', index=False)
sub.to_csv('submission.csv', index=False)
print("Saved to submission.csv")
