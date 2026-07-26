import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

def run_hybrid():
    print("Loading data...", flush=True)
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
    
    print("Extracting TF-IDF (Sparse) Features...", flush=True)
    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=3000)
    X_train_sparse = tfidf.fit_transform(train_single['text_combined']).toarray()
    X_test_sparse = tfidf.transform(test_single['text_combined']).toarray()
    
    print("Extracting MiniLM (Dense) Features...", flush=True)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    X_train_dense = model.encode(train_single['text_combined'].tolist(), batch_size=64, show_progress_bar=True)
    X_test_dense = model.encode(test_single['text_combined'].tolist(), batch_size=64, show_progress_bar=True)
    
    X_train = np.hstack([X_train_sparse, X_train_dense])
    X_test = np.hstack([X_test_sparse, X_test_dense])
    
    y_train = train_single['answer'].values
    y_test_pseudo = test_single['answer'].values
    
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test_pseudo])
    
    print("Training Sparse+Dense Hybrid Model...", flush=True)
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, l2_regularization=0.5, max_leaf_nodes=31, random_state=42)
    hgb.fit(X_full, y_full)
    
    final_classes = hgb.predict(X_test)
    test_single['new_prediction'] = final_classes
    
    print("Mapping back...", flush=True)
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
    out.to_csv('submission_v69_sparse_dense_hybrid.csv', index=False)
    diff = sum([1 for a, b in zip(out['prediction'], test_df['answer']) if str(a) != str(b)])
    print(f"Done! Differences from 0.44152 pseudo-labels: {diff}", flush=True)

if __name__ == '__main__':
    run_hybrid()
