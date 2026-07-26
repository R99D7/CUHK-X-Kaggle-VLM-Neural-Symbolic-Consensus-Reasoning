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

sub = pd.read_csv('submission.csv') # this is v243 which scored 0.54093

changes = 0
for idx, row in te[te['category'] == 'multi'].iterrows():
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    
    pred_letters = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    if len(pred_letters) >= 2:
        pred_acts = [opts[l] for l in pred_letters]
        pred_score = 0
        for a1, a2 in combinations(sorted(pred_acts), 2):
            pred_score += cooccur[(a1, a2)]
            
        if pred_score == 0:
            print(f"{row['qa_id']}: predicted={pred_letters} (acts={pred_acts}) has 0 co-occurrence!")
            changes += 1

print(f"Total multi predictions with 0 co-occurrence: {changes}")
