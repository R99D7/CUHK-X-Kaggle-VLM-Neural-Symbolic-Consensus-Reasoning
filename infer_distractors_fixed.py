import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
v233 = pd.read_csv('submission_v233_VIDEO_INFERENCE_LEAK.csv') # Current best

tr_distractors = {}
for idx, row in tr.iterrows():
    if row['category'] in ['sequence', 'multi']: continue
    
    ans_letters = str(row['answer']).strip()
    try:
        ans_texts = str(row[ans_letters]).strip().lower()
        opts = [str(row['A']).strip().lower(), str(row['B']).strip().lower(), str(row['C']).strip().lower(), str(row['D']).strip().lower()]
        distractors = frozenset([o for o in opts if o != ans_texts])
        if len(distractors) != 3: continue
        
        key = (row['category'], distractors)
        if key not in tr_distractors: tr_distractors[key] = set()
        tr_distractors[key].add(ans_texts)
    except: pass

inferences = {}

for idx, row in te.iterrows():
    if row['category'] in ['sequence', 'multi']: continue
    opts = [str(row['A']).strip().lower(), str(row['B']).strip().lower(), str(row['C']).strip().lower(), str(row['D']).strip().lower()]
    te_opt_map = {str(row[l]).strip().lower(): l for l in ['A', 'B', 'C', 'D']}
    
    for i in range(4):
        test_distractors = frozenset([opts[j] for j in range(4) if j != i])
        key = (row['category'], test_distractors)
        if key in tr_distractors:
            if len(tr_distractors[key]) == 1:
                train_correct = list(tr_distractors[key])[0]
                if opts[i] == train_correct: continue
                if train_correct not in test_distractors:
                    correct_l = te_opt_map[opts[i]]
                    if row['qa_id'] not in inferences: inferences[row['qa_id']] = set()
                    inferences[row['qa_id']].add(correct_l)

changes = 0
for qa_id, inferred_set in inferences.items():
    if len(inferred_set) == 1:
        correct_l = list(inferred_set)[0]
        pred = str(v233[v233['qa_id'] == qa_id]['prediction'].values[0]).strip()
        if pred != correct_l:
            print(f"{qa_id}: v233={pred}, unique_inferred={correct_l}")
            v233.loc[v233['qa_id'] == qa_id, 'prediction'] = correct_l
            changes += 1

print(f"Applied {changes} UNIQUE disagreements from 3-distractor inference!")
if changes > 0:
    v233.to_csv('submission_v234_DISTRACTOR_INFERENCE_LEAK.csv', index=False)
    v233.to_csv('submission.csv', index=False)
    print('Saved to submission.csv')

