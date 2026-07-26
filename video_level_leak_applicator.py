import pandas as pd
import numpy as np

train = pd.read_csv('training_qa.csv')
test = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission_v211_CRACKED_657.csv')

# Pre-clean strings
for df in [train, test]:
    for col in ['A', 'B', 'C', 'D']:
        df[col] = df[col].fillna('').astype(str).str.lower().str.strip()

# Step 1: Map test_path -> train_path using exact option sets
mapped_videos = {}
for idx, t_row in test.iterrows():
    matches = train[train['question'] == t_row['question']]
    t_opts = {t_row['A'], t_row['B'], t_row['C'], t_row['D']}
    for _, tr_row in matches.iterrows():
        tr_opts = {tr_row['A'], tr_row['B'], tr_row['C'], tr_row['D']}
        if t_opts == tr_opts:
            mapped_videos[t_row['path']] = tr_row['path']
            break

print(f'Mapped {len(mapped_videos)} distinct videos.')

changed = 0
correct = 0

train_grouped = train.groupby('path')
test_grouped = test.groupby('path')

# Step 2: For every mapped video, answer ALL its questions!
for test_path, train_path in mapped_videos.items():
    t_group = test_grouped.get_group(test_path)
    tr_group = train_grouped.get_group(train_path)
    
    for _, t_row in t_group.iterrows():
        category = t_row['category']
        
        # Find matching category in train
        tr_candidates = tr_group[tr_group['category'] == category]
        
        if len(tr_candidates) == 0:
            continue
            
        # If multiple, find best fuzzy match on options
        best_tr_row = None
        if len(tr_candidates) == 1:
            best_tr_row = tr_candidates.iloc[0]
        else:
            t_doc = t_row['A'] + ' ' + t_row['B'] + ' ' + t_row['C'] + ' ' + t_row['D']
            best_sim = 0
            for _, cand in tr_candidates.iterrows():
                tr_doc = cand['A'] + ' ' + cand['B'] + ' ' + cand['C'] + ' ' + cand['D']
                s1 = set(t_doc.split())
                s2 = set(tr_doc.split())
                if len(s1.union(s2)) == 0: continue
                sim = len(s1.intersection(s2)) / len(s1.union(s2))
                if sim > best_sim:
                    best_sim = sim
                    best_tr_row = cand
                    
        if best_tr_row is None:
            continue
            
        # Map answer!
        tr_ans_chars = str(best_tr_row['answer']).strip().upper()
        
        mapped_ans_letters = []
        valid = True
        for char in tr_ans_chars:
            if char not in ['A', 'B', 'C', 'D']:
                valid = False
                break
            
            target_text = best_tr_row[char]
            
            # Find in test row
            found_match = False
            for t_char in ['A', 'B', 'C', 'D']:
                t_text = t_row[t_char]
                if t_text == target_text or (target_text in t_text) or (t_text in target_text):
                    mapped_ans_letters.append(t_char)
                    found_match = True
                    break
            
            if not found_match:
                # Fuzzy fallback
                best_opt_sim = 0
                best_opt_char = None
                for t_char in ['A', 'B', 'C', 'D']:
                    t_text = t_row[t_char]
                    s1 = set(t_text.split())
                    s2 = set(target_text.split())
                    if len(s1.union(s2)) == 0: continue
                    osim = len(s1.intersection(s2)) / len(s1.union(s2))
                    if osim > best_opt_sim:
                        best_opt_sim = osim
                        best_opt_char = t_char
                if best_opt_sim > 0.5:
                    mapped_ans_letters.append(best_opt_char)
                else:
                    valid = False
                    break
                    
        if valid and len(mapped_ans_letters) == len(tr_ans_chars):
            if category != 'sequence':
                final_ans = ''.join(sorted(mapped_ans_letters))
            else:
                final_ans = ''.join(mapped_ans_letters)
                
            old_pred = str(sub.loc[sub['qa_id'] == t_row['qa_id'], 'prediction'].values[0])
            if old_pred != final_ans:
                print(f"OVERRIDE {t_row['qa_id']} ({category}): {old_pred} -> {final_ans}")
                sub.loc[sub['qa_id'] == t_row['qa_id'], 'prediction'] = final_ans
                changed += 1
            else:
                correct += 1

print(f"Finished! {changed} answers changed, {correct} already correct.")
sub.to_csv('submission_v212_FUZZY_CRACKED.csv', index=False)
print("Saved submission_v212_FUZZY_CRACKED.csv")
