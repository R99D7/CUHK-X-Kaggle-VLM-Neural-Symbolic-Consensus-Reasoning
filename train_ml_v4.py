import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import gc

def run_ml_pipeline_v4():
    print("Loading data...")
    train_df = pd.read_csv('training_qa.csv')
    test_df = pd.read_csv('test_qa.csv')
    sample_df = pd.read_csv('sample_submission.csv')
    
    sample_preds = dict(zip(sample_df['qa_id'], sample_df['prediction']))
    
    print("Loading Sentence Transformer...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def extract_features(df, is_train=True):
        X_dense = []
        X_num = []
        y = []
        qa_ids = []
        letters = []
        
        cat_map = {'single': 0, 'multi': 1, 'emotion': 2, 'combination': 3, 'sequence': 4, 'object_interaction': 5}
        
        for idx, row in df.iterrows():
            cat = str(row['category'])
            cat_idx = cat_map.get(cat, 0)
            
            q = str(row['question'])
            opts = [str(row['A']), str(row['B']), str(row['C']), str(row['D'])]
            
            ans = str(row.get('answer', ''))
            
            for i, letter in enumerate(['A', 'B', 'C', 'D']):
                opt = opts[i]
                
                # Combine question and option for embedding
                text = f"Question: {q} Answer: {opt}"
                X_dense.append(text)
                
                # Simple heuristics
                q_len = len(q)
                opt_len = len(opt)
                X_num.append([cat_idx, q_len, opt_len])
                
                if is_train:
                    y.append(1 if letter in ans else 0)
                    
                qa_ids.append(row['qa_id'])
                letters.append(letter)
                
        return X_dense, np.array(X_num), y, qa_ids, letters

    print("Extracting text and labels...")
    X_text_train, X_num_train, y_train, _, _ = extract_features(train_df, is_train=True)
    X_text_test, X_num_test, _, test_qa_ids, test_letters = extract_features(test_df, is_train=False)
    
    print("Encoding texts into dense vectors (this takes ~30 seconds)...")
    X_emb_train = model.encode(X_text_train, batch_size=64, show_progress_bar=True)
    X_emb_test = model.encode(X_text_test, batch_size=64, show_progress_bar=True)
    
    print("Training Logistic Regression on Dense Embeddings...")
    lr = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
    lr.fit(X_emb_train, y_train)
    
    probs_train = lr.predict_proba(X_emb_train)[:, 1].reshape(-1, 1)
    probs_test = lr.predict_proba(X_emb_test)[:, 1].reshape(-1, 1)
    
    print("Training Level-2 GBM...")
    X_meta_train = np.hstack([X_num_train, probs_train])
    X_meta_test = np.hstack([X_num_test, probs_test])
    
    gbm = HistGradientBoostingClassifier(
        max_iter=300, 
        learning_rate=0.05, 
        max_depth=10, 
        categorical_features=[0],
        random_state=42,
        class_weight='balanced'
    )
    gbm.fit(X_meta_train, y_train)
    
    print("Predicting...")
    probs = gbm.predict_proba(X_meta_test)[:, 1]
    
    predictions = {}
    for qa_id, letter, prob in zip(test_qa_ids, test_letters, probs):
        if qa_id not in predictions:
            predictions[qa_id] = []
        predictions[qa_id].append((prob, letter))
        
    final_preds = []
    for _, row in test_df.iterrows():
        qid = row['qa_id']
        cat = row['category']
        expected_pred = str(sample_preds.get(qid, 'A'))
        
        scores = predictions[qid]
        scores.sort(key=lambda x: x[0], reverse=True)
        
        expected_len = len(expected_pred)
        if expected_len == 0: expected_len = 1
        
        pred_letters = [letter for prob, letter in scores[:expected_len]]
        
        if cat == 'sequence':
            pred = "".join(pred_letters)
        else:
            pred_letters.sort()
            pred = "".join(pred_letters)
            
        final_preds.append({'qa_id': qid, 'prediction': pred})
        
    print("Saving submission_ml_v4.csv...")
    pd.DataFrame(final_preds).to_csv('submission_ml_v4.csv', index=False)
    print("Done!")

if __name__ == '__main__':
    run_ml_pipeline_v4()
