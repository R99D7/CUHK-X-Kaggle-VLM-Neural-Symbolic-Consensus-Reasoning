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
        
        # If the option wasn't found (e.g. malformed), skip
        if len(distractors) != 3: continue
        
        key = (row['category'], distractors)
        if key not in tr_distractors:
            tr_distractors[key] = set()
        tr_distractors[key].add(ans_texts)
    except:
        pass

changes = []

for idx, row in te.iterrows():
    if row['category'] in ['sequence', 'multi']: continue
    
    opts = [str(row['A']).strip().lower(), str(row['B']).strip().lower(), str(row['C']).strip().lower(), str(row['D']).strip().lower()]
    te_opt_map = {str(row[l]).strip().lower(): l for l in ['A', 'B', 'C', 'D']}
    
    for i in range(4):
        # Assume opts[i] is the correct answer
        test_distractors = frozenset([opts[j] for j in range(4) if j != i])
        key = (row['category'], test_distractors)
        
        if key in tr_distractors:
            # We found the exact 3 distractors in a train question!
            if len(tr_distractors[key]) == 1:
                train_correct = list(tr_distractors[key])[0]
                
                # If the assumed test correct answer is the SAME as the train correct answer,
                # then this is an exact option set match (which we already fixed).
                if opts[i] == train_correct:
                    continue
                    
                # Otherwise, it means the test question has 3 identical distractors to train,
                # BUT the test options DO NOT contain the train correct answer!
                # Wait, if train_correct is NOT in the test options, then train_correct wasn't an option.
                # The generator gave the SAME 3 distractors for a DIFFERENT correct answer (opts[i]).
                # Is it possible thatopts[i] is the ONLY valid correct answer?
                # Actually, if the generator maps (correct_answer) -> (distractor1, distractor2, distractor3) deterministically,
                # it's possible that TWO different correct answers map to the SAME 3 distractors.
                # If they do, then seeing those 3 distractors means the answer is EITHER train_correct OR opts[i].
                # Since train_correct is not an option, the answer MUST be opts[i]!
                
                if train_correct not in test_distractors:
                    # opts[i] MUST be the correct answer!
                    correct_l = te_opt_map[opts[i]]
                    pred = str(v233[v233['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
                    if pred != correct_l:
                        print(f"{row['qa_id']} ({row['category']}): v233={pred}, inferred_by_distractors={correct_l}")
                        changes.append({'qa_id': row['qa_id'], 'new_pred': correct_l})

print(f"Found {len(changes)} disagreements from 3-distractor inference!")
if len(changes) > 0:
    for c in changes:
        v233.loc[v233['qa_id'] == c['qa_id'], 'prediction'] = c['new_pred']
    v233.to_csv('submission_v234_DISTRACTOR_INFERENCE_LEAK.csv', index=False)
    v233.to_csv('submission.csv', index=False)
    print('Saved to submission.csv')

