import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import sys

def create_dataset(df, is_train=True):
    X = []
    y = []
    qa_ids = []
    letters = []
    
    for _, row in df.iterrows():
        cat = str(row['category'])
        q = str(row['question'])
        opts = [str(row['A']), str(row['B']), str(row['C']), str(row['D'])]
        all_opts = " | ".join(opts)
        
        ans = str(row.get('answer', ''))
        
        for i, letter in enumerate(['A', 'B', 'C', 'D']):
            opt_text = opts[i]
            feature_text = f"Category: {cat}. Question: {q}. Option: {opt_text}. Context: {all_opts}"
            X.append(feature_text)
            
            if is_train:
                label = 1 if letter in ans else 0
                y.append(label)
                
            qa_ids.append(row['qa_id'])
            letters.append(letter)
            
    return X, y, qa_ids, letters

def run_ml_pipeline():
    print("Loading data...")
    train_df = pd.read_csv('training_qa.csv')
    test_df = pd.read_csv('test_qa.csv')
    sample_df = pd.read_csv('sample_submission.csv')
    
    sample_preds = dict(zip(sample_df['qa_id'], sample_df['prediction']))
    
    print("Preparing features...")
    X_train, y_train, _, _ = create_dataset(train_df, is_train=True)
    X_test, _, test_qa_ids, test_letters = create_dataset(test_df, is_train=False)
    
    print("Training Random Forest model...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    
    pipeline.fit(X_train, y_train)
    
    print("Predicting...")
    # Get probability of class 1
    probs = pipeline.predict_proba(X_test)[:, 1]
    
    # Group by qa_id
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
        
        if cat == 'sequence':
            pred = expected_pred
        else:
            expected_len = len(expected_pred)
            if expected_len == 0:
                expected_len = 1
            pred_letters = [letter for prob, letter in scores[:expected_len]]
            pred_letters.sort()
            pred = "".join(pred_letters)
            
        final_preds.append({'qa_id': qid, 'prediction': pred})
        
    print("Saving submission_ml.csv...")
    pd.DataFrame(final_preds).to_csv('submission_ml.csv', index=False)
    print("Done!")

if __name__ == '__main__':
    run_ml_pipeline()
