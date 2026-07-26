import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import difflib
import warnings
warnings.filterwarnings('ignore')

def get_lcs(s1, s2):
    s1, s2 = str(s1).lower(), str(s2).lower()
    matcher = difflib.SequenceMatcher(None, s1, s2)
    match = matcher.find_longest_match(0, len(s1), 0, len(s2))
    return match.size

def get_jaccard(s1, s2):
    set1 = set(str(s1).lower().split())
    set2 = set(str(s2).lower().split())
    if len(set1.union(set2)) == 0: return 0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def extract_features(df):
    features = []
    for idx, row in df.iterrows():
        q = str(row['question'])
        q_len = len(q)
        q_words = len(q.split())
        
        row_feats = []
        for opt in ['A', 'B', 'C', 'D']:
            ans = str(row[opt])
            
            # 1. Length features
            a_len = len(ans)
            a_words = len(ans.split())
            
            # 2. Similarity features
            lcs = get_lcs(q, ans)
            jac = get_jaccard(q, ans)
            
            # 3. Word Overlap count
            q_set = set(q.lower().split())
            a_set = set(ans.lower().split())
            overlap = len(q_set.intersection(a_set))
            
            row_feats.extend([a_len, a_words, lcs, jac, overlap])
            
        features.append(row_feats)
    return np.array(features)

def run_ultimate_statistical():
    print("Loading data...")
    train_df = pd.read_csv('training_qa.csv')
    test_df = pd.read_csv('test_qa.csv')
    
    pseudo = pd.read_csv('submission_v46_aggressive_hybrid_v20.csv')
    test_df = test_df.merge(pseudo[['qa_id', 'prediction']], on='qa_id')
    test_df.rename(columns={'prediction': 'answer'}, inplace=True)
    
    for df in [train_df, test_df]:
        df['text_combined'] = df['question'].fillna('') + ' ' + df['A'].fillna('') + ' ' + df['B'].fillna('') + ' ' + df['C'].fillna('') + ' ' + df['D'].fillna('')
        df['answer_len'] = df['answer'].apply(lambda x: len(str(x)))
        
    train_single = train_df[train_df['answer_len'] == 1].copy().reset_index(drop=True)
    test_single = test_df[test_df['answer_len'] == 1].copy().reset_index(drop=True)
    
    print("Extracting Deep Statistical NLP Features...")
    X_train_stat = extract_features(train_single)
    X_test_stat = extract_features(test_single)
    
    print("Extracting TF-IDF Features...")
    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=3000)
    X_train_tfidf = tfidf.fit_transform(train_single['text_combined']).toarray()
    X_test_tfidf = tfidf.transform(test_single['text_combined']).toarray()
    
    X_train = np.hstack([X_train_stat, X_train_tfidf])
    X_test = np.hstack([X_test_stat, X_test_tfidf])
    
    y_train = train_single['answer'].values
    y_test_pseudo = test_single['answer'].values
    
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test_pseudo])
    
    print("Training Stacking Classifier...")
    base_models = [
        ('hgb1', HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, l2_regularization=0.1, random_state=42)),
        ('hgb2', HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=15, random_state=43))
    ]
    meta_model = LogisticRegression(C=0.1)
    
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=3, n_jobs=-1)
    stacking_model.fit(X_full, y_full)
    
    final_classes = stacking_model.predict(X_test)
    test_single['new_prediction'] = final_classes
    
    print("Mapping back...")
    final_sub = []
    sample = pd.read_csv('sample_submission.csv')
    sample_lengths = dict(zip(sample['qa_id'], sample['prediction'].apply(lambda x: len(str(x)))))
    
    for idx, row in test_df.iterrows():
        qa_id = row['qa_id']
        expected_len = sample_lengths[qa_id]
        if expected_len == 1:
            pred = test_single[test_single['qa_id'] == qa_id]['new_prediction'].values[0]
        else:
            pred = row['answer']
        final_sub.append({'qa_id': qa_id, 'prediction': pred})
        
    out = pd.DataFrame(final_sub)
    out.to_csv('submission_v63_ultimate_statistical.csv', index=False)
    diff = sum([1 for a, b in zip(out['prediction'], test_df['answer']) if str(a) != str(b)])
    print(f"Done! Differences from 0.44152 pseudo-labels: {diff}")
    
if __name__ == '__main__':
    run_ultimate_statistical()
