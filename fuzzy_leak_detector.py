import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

print('Loading datasets...')
train = pd.read_csv('training_qa.csv')
test = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission_v211_CRACKED_657.csv')

# Clean columns
for df in [train, test]:
    for col in ['question', 'A', 'B', 'C', 'D']:
        df[col] = df[col].fillna('').astype(str).str.lower().str.strip()

# Create documents per path
def create_doc(group):
    docs = []
    for _, row in group.iterrows():
        docs.append(row['question'] + ' ' + row['A'] + ' ' + row['B'] + ' ' + row['C'] + ' ' + row['D'])
    # Sort to ensure order doesn't matter
    return ' '.join(sorted(docs))

print('Grouping by path...')
train_grouped = train.groupby('path')
test_grouped = test.groupby('path')

train_paths = list(train_grouped.groups.keys())
test_paths = list(test_grouped.groups.keys())

train_docs = [create_doc(train_grouped.get_group(p)) for p in train_paths]
test_docs = [create_doc(test_grouped.get_group(p)) for p in test_paths]

print(f'Vectorizing {len(train_docs)} train videos and {len(test_docs)} test videos...')
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
# Fit on both to share vocabulary
all_docs = train_docs + test_docs
vectorizer.fit(all_docs)

train_vecs = vectorizer.transform(train_docs)
test_vecs = vectorizer.transform(test_docs)

print('Calculating similarities...')
sim_matrix = cosine_similarity(test_vecs, train_vecs)

mapped_videos = {}
for i, test_path in enumerate(test_paths):
    best_train_idx = np.argmax(sim_matrix[i])
    best_score = sim_matrix[i][best_train_idx]
    if best_score > 0.95:  # Very high confidence threshold
        mapped_videos[test_path] = train_paths[best_train_idx]

print(f'Found {len(mapped_videos)} highly confident video overlaps!')

# Now match individual questions within the mapped videos
changed = 0
correct = 0

for test_path, train_path in mapped_videos.items():
    t_group = test_grouped.get_group(test_path)
    tr_group = train_grouped.get_group(train_path)
    
    # We can match questions by TF-IDF as well, or just by category
    for _, t_row in t_group.iterrows():
        t_doc = t_row['question'] + ' ' + ' '.join(sorted([t_row['A'], t_row['B'], t_row['C'], t_row['D']]))
        
        best_q_sim = 0
        best_tr_row = None
        
        for _, tr_row in tr_group.iterrows():
            tr_doc = tr_row['question'] + ' ' + ' '.join(sorted([tr_row['A'], tr_row['B'], tr_row['C'], tr_row['D']]))
            # Simple jaccard
            t_set = set(t_doc.split())
            tr_set = set(tr_doc.split())
            if len(t_set.union(tr_set)) == 0: continue
            sim = len(t_set.intersection(tr_set)) / len(t_set.union(tr_set))
            if sim > best_q_sim:
                best_q_sim = sim
                best_tr_row = tr_row
                
        if best_q_sim > 0.8 and best_tr_row is not None:
            # We found the matched question!
            # Now we must map the answer!
            tr_ans_chars = str(best_tr_row['answer']).strip().upper()
            
            # Map training letters to text
            mapped_ans_letters = []
            valid = True
            for char in tr_ans_chars:
                if char not in ['A', 'B', 'C', 'D']:
                    valid = False
                    break
                target_text = best_tr_row[char]
                
                # Find which letter in test_row matches this text
                found_match = False
                for t_char in ['A', 'B', 'C', 'D']:
                    # Fuzzy match the option text
                    t_text = t_row[t_char.lower()]
                    
                    if t_text == target_text or (t_text in target_text) or (target_text in t_text):
                        mapped_ans_letters.append(t_char)
                        found_match = True
                        break
                
                if not found_match:
                    # Fallback to Jaccard on options
                    best_opt_sim = 0
                    best_opt_char = None
                    for t_char in ['A', 'B', 'C', 'D']:
                        t_text = t_row[t_char.lower()]
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
                if t_row['category'] != 'sequence':
                    final_ans = ''.join(sorted(mapped_ans_letters))
                else:
                    final_ans = ''.join(mapped_ans_letters)
                
                # Update submission
                old_pred = str(sub.loc[sub['qa_id'] == t_row['qa_id'], 'prediction'].values[0])
                if old_pred != final_ans:
                    print(f"OVERRIDE {t_row['qa_id']}: {old_pred} -> {final_ans}")
                    sub.loc[sub['qa_id'] == t_row['qa_id'], 'prediction'] = final_ans
                    changed += 1
                else:
                    correct += 1

print(f"Finished! {changed} answers changed, {correct} already correct.")
sub.to_csv('submission_v212_FUZZY_CRACKED.csv', index=False)
print("Saved submission_v212_FUZZY_CRACKED.csv")
