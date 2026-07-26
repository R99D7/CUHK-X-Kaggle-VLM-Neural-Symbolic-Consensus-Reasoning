import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

def run_mega_forest():
    print("Loading data...")
    train_df = pd.read_csv('training_qa.csv')
    test_df = pd.read_csv('test_qa.csv')
    
    # 0.44152 Base
    pseudo = pd.read_csv('submission_v46_aggressive_hybrid_v20.csv')
    test_df = test_df.merge(pseudo[['qa_id', 'prediction']], on='qa_id')
    test_df.rename(columns={'prediction': 'answer'}, inplace=True)
    
    # Text features
    for df in [train_df, test_df]:
        df['text_combined'] = df['question'].fillna('') + ' ' + df['A'].fillna('') + ' ' + df['B'].fillna('') + ' ' + df['C'].fillna('') + ' ' + df['D'].fillna('')
        df['answer_len'] = df['answer'].apply(lambda x: len(str(x)))
        df['q_len'] = df['question'].apply(lambda x: len(str(x)))
        
    train_single = train_df[train_df['answer_len'] == 1].copy().reset_index(drop=True)
    test_single = test_df[test_df['answer_len'] == 1].copy().reset_index(drop=True)
    
    print("Generating Features...")
    tfidf_word = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), max_features=5000, stop_words='english')
    X_train_word = tfidf_word.fit_transform(train_single['text_combined']).toarray()
    X_test_word = tfidf_word.transform(test_single['text_combined']).toarray()
    
    tfidf_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=5000)
    X_train_char = tfidf_char.fit_transform(train_single['text_combined']).toarray()
    X_test_char = tfidf_char.transform(test_single['text_combined']).toarray()
    
    X_train = np.hstack([X_train_word, X_train_char])
    X_test = np.hstack([X_test_word, X_test_char])
    
    y_train = train_single['answer'].values
    y_test_pseudo = test_single['answer'].values
    
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test_pseudo])
    
    print("Training Mega Pipeline Forest (100 independent models)...")
    all_preds = []
    
    # We will train 20 models with different random seeds and feature subsets
    for seed in range(20):
        print(f"Training tree {seed+1}/20...")
        model = HistGradientBoostingClassifier(
            max_iter=150, 
            learning_rate=0.1, 
            max_features=0.6, # Random Subspace Method (crucial for diversity!)
            max_leaf_nodes=31,
            random_state=seed,
            l2_regularization=0.1
        )
        model.fit(X_full, y_full)
        preds = model.predict(X_test)
        all_preds.append(preds)
        
    # Majority vote
    all_preds = np.array(all_preds)
    final_classes = []
    for i in range(all_preds.shape[1]):
        vals, counts = np.unique(all_preds[:, i], return_counts=True)
        final_classes.append(vals[np.argmax(counts)])
        
    test_single['new_prediction'] = final_classes
    
    print("Mapping back...")
    final_sub = []
    sample = pd.read_csv('sample_submission.csv')
    sample_lengths = dict(zip(sample['qa_id'], sample['prediction'].apply(lambda x: len(str(x)))))
    
    for idx, row in test_df.iterrows():
        qa_id = row['qa_id']
        expected_len = sample_lengths[qa_id]
        orig_pred = row['answer']
        
        if expected_len == 1:
            pred = test_single[test_single['qa_id'] == qa_id]['new_prediction'].values[0]
        else:
            pred = orig_pred
            
        final_sub.append({'qa_id': qa_id, 'prediction': pred})
        
    out = pd.DataFrame(final_sub)
    out.to_csv('submission_v60_mega_forest.csv', index=False)
    
    diff = sum([1 for a, b in zip(out['prediction'], test_df['answer']) if str(a) != str(b)])
    print(f"Done! Differences from 0.44152 pseudo-labels: {diff}")
    
if __name__ == '__main__':
    run_mega_forest()
