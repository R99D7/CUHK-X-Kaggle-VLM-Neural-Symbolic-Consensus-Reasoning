import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import catboost as cb
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

def run_ultra_pipeline():
    print("Loading data...")
    train_df = pd.read_csv('training_qa.csv')
    test_df = pd.read_csv('test_qa.csv')
    
    # 0.44152 Base
    pseudo = pd.read_csv('submission_v46_aggressive_hybrid_v20.csv')
    test_df = test_df.merge(pseudo[['qa_id', 'prediction']], on='qa_id')
    test_df.rename(columns={'prediction': 'answer'}, inplace=True)
    
    # Prepare text
    for df in [train_df, test_df]:
        df['text_combined'] = df['question'].fillna('') + ' ' + df['A'].fillna('') + ' ' + df['B'].fillna('') + ' ' + df['C'].fillna('') + ' ' + df['D'].fillna('')
        df['answer_len'] = df['answer'].apply(lambda x: len(str(x)))
        
    train_single = train_df[train_df['answer_len'] == 1].copy().reset_index(drop=True)
    test_single = test_df[test_df['answer_len'] == 1].copy().reset_index(drop=True)
    
    print("Generating BGE-Large Neural Embeddings...")
    st_model = SentenceTransformer('BAAI/bge-large-en-v1.5', device='cpu')
    X_train_emb = st_model.encode(train_single['text_combined'].tolist(), show_progress_bar=True)
    X_test_emb = st_model.encode(test_single['text_combined'].tolist(), show_progress_bar=True)
    
    print("Generating TF-IDF Char N-Grams (for semantic overlap)...")
    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=5000)
    X_train_tfidf = tfidf.fit_transform(train_single['text_combined']).toarray()
    X_test_tfidf = tfidf.transform(test_single['text_combined']).toarray()
    
    X_train = np.hstack([X_train_emb, X_train_tfidf])
    X_test = np.hstack([X_test_emb, X_test_tfidf])
    
    y_train = train_single['answer'].values
    y_test_pseudo = test_single['answer'].values
    
    print("Stacking for Pseudo-Labeling...")
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test_pseudo])
    
    print("Training Ensemble...")
    models = [
        ('hgb', HistGradientBoostingClassifier(max_iter=200, random_state=42, l2_regularization=0.1)),
        ('lr', LogisticRegression(max_iter=2000, random_state=42, C=0.5)),
        ('cb', cb.CatBoostClassifier(iterations=300, depth=6, verbose=0, random_seed=42))
    ]
    
    meta_preds = []
    
    for name, model in models:
        print(f"Training {name}...")
        model.fit(X_full, y_full)
        preds = model.predict_proba(X_test)
        meta_preds.append(preds)
        
    avg_preds = np.mean(meta_preds, axis=0)
    final_classes = models[0][1].classes_[np.argmax(avg_preds, axis=1)]
    
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
    out.to_csv('submission_v56_ultra_bge.csv', index=False)
    
    diff = sum([1 for a, b in zip(out['prediction'], test_df['answer']) if str(a) != str(b)])
    print(f"Done! Differences from 0.44152 pseudo-labels: {diff}")
    
if __name__ == '__main__':
    run_ultra_pipeline()
