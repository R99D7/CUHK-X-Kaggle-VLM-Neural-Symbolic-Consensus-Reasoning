"""
Meta-Learner Architecture for 0.88+ Leaderboard Target.
Trains a Gradient Boosted Decision Engine on training_qa.csv (4,351 labeled sequences)
using Vision Transformer probabilities, Option Structural Features, Co-occurrence Scores, and Category Distributions.
Evaluates 5-Fold Stratified CV accuracy and generates submission_v281_META_LEARNER_088.csv.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from collections import Counter

# 1. Load Data
train = pd.read_csv("training_qa.csv")
test = pd.read_csv("test_qa.csv")
raw = pd.read_csv("transformer_fixed_raw_predictions.csv")
raw_map = dict(zip(raw['qa_id'], raw.to_dict('records')))

print(f"Loaded {len(train)} training questions and {len(test)} test questions.")

# 2. Build Action Co-Occurrence Matrix from Train Ground Truth
cooccur = Counter()
for idx, r in train.iterrows():
    ans = str(r['answer']).strip()
    if ans in ['A', 'B', 'C', 'D']:
        ans_text = str(r[ans]).strip().lower()
        acts = [x.strip() for x in ans_text.replace('->', ',').split(',')]
        for i in range(len(acts)):
            for j in range(i+1, len(acts)):
                cooccur[(acts[i], acts[j])] += 1
                cooccur[(acts[j], acts[i])] += 1

def get_cooccur_score(option_text):
    acts = [x.strip().lower() for x in str(option_text).replace('->', ',').split(',')]
    if len(acts) < 2: return 0.0
    score = 0.0
    for i in range(len(acts)):
        for j in range(i+1, len(acts)):
            score += cooccur.get((acts[i], acts[j]), 0)
    return score

# 3. Feature Extraction Function for Question-Level Choice Prediction
def extract_qa_features(df, is_train=True):
    X = []
    y = []
    
    cat_map = {'single': 0, 'multi': 1, 'sequence': 2, 'combination': 3, 'emotion': 4}
    
    for idx, r in df.iterrows():
        qid = r['qa_id']
        cat_id = cat_map.get(str(r['category']).strip().lower(), 0)
        
        # Raw probabilities for A, B, C, D
        r_dict = raw_map.get(qid, {})
        probs = [r_dict.get(f'raw_prob_{l}', 0.25) for l in ['A', 'B', 'C', 'D']]
        max_p = max(probs)
        sum_p = sum(probs) if sum(probs) > 0 else 1.0
        norm_probs = [p / sum_p for p in probs]
        
        # Features for each option
        feats = [cat_id]
        
        for i, l in enumerate(['A', 'B', 'C', 'D']):
            opt_str = str(r[l]) if pd.notna(r[l]) else ""
            opt_len = len(opt_str)
            num_acts = len(opt_str.replace('->', ',').split(',')) if opt_str else 0
            co_sc = get_cooccur_score(opt_str)
            p = probs[i]
            norm_p = norm_probs[i]
            diff_to_max = max_p - p
            is_max = 1.0 if p == max_p else 0.0
            
            feats.extend([p, norm_p, diff_to_max, is_max, opt_len, num_acts, co_sc])
            
        X.append(feats)
        
        if is_train:
            ans = str(r['answer']).strip()
            ans_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            y.append(ans_map.get(ans, 0))
            
    return np.array(X), np.array(y) if is_train else None

print("\nExtracting features for Training and Test sets...")
X_train, y_train = extract_qa_features(train, is_train=True)
X_test, _ = extract_qa_features(test, is_train=False)

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# 4. Train Meta-Learner with 5-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
test_preds_list = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]
    
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42)
    clf.fit(X_tr, y_tr)
    
    val_acc = clf.score(X_va, y_va)
    cv_scores.append(val_acc)
    print(f"Fold {fold+1} Accuracy: {round(val_acc, 4)}")
    
    test_probs = clf.predict_proba(X_test)
    test_preds_list.append(test_probs)

mean_cv = np.mean(cv_scores)
print(f"\n=== 5-FOLD STRATIFIED CV ACCURACY: {round(mean_cv, 4)} ===")

# 5. Generate Test Predictions from Ensemble of Folds
avg_test_probs = np.mean(test_preds_list, axis=0)
final_preds_idx = np.argmax(avg_test_probs, axis=1)

ans_lookup = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
test_preds = [ans_lookup[idx] for idx in final_preds_idx]

# 6. Create Submission
sub_meta = pd.DataFrame({
    'qa_id': test['qa_id'],
    'prediction': test_preds
})

# For multi-choice questions, blend meta-learner choice with verified multi strings
v276 = pd.read_csv("submission_v276_APEX_SUMMIT.csv")
v276_map = dict(zip(v276['qa_id'], v276['prediction']))

blended_preds = []
for idx, r in test.iterrows():
    qid = r['qa_id']
    cat = str(r['category']).strip().lower()
    meta_p = sub_meta.loc[sub_meta['qa_id'] == qid, 'prediction'].iloc[0]
    v276_p = v276_map.get(qid, meta_p)
    
    if cat == 'multi':
        # For multi, retain multi-letter string if length > 1, else use meta-learner
        blended_preds.append(v276_p if len(v276_p) > 1 else meta_p)
    else:
        # For single, combination, sequence, emotion: use meta-learner prediction
        blended_preds.append(meta_p)

sub_meta['prediction'] = blended_preds
sub_meta.to_csv("submission_v281_META_LEARNER_088.csv", index=False)

print(f"\nGenerated submission_v281_META_LEARNER_088.csv with {len(sub_meta)} rows!")
