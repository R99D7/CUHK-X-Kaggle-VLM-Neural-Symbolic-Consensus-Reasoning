import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import gc

def run_ml_pipeline_v6():
    print("Loading data...")
    train_df = pd.read_csv('training_qa.csv')
    test_df = pd.read_csv('test_qa.csv')
    sample_df = pd.read_csv('sample_submission.csv')
    
    sample_preds = dict(zip(sample_df['qa_id'], sample_df['prediction']))
    
    def extract_features(df, is_train=True):
        X_text = []
        X_num = []
        y = []
        qa_ids = []
        letters = []
        
        cat_map = {
            'single': 0, 'multi': 1, 'emotion': 2, 
            'combination': 3, 'sequence': 4, 'object_interaction': 5
        }
        
        letter_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        
        for _, row in df.iterrows():
            cat = str(row['category'])
            cat_idx = cat_map.get(cat, 0)
            
            q = str(row['question'])
            q_words = set(q.lower().split())
            
            opts = [str(row['A']), str(row['B']), str(row['C']), str(row['D'])]
            opt_lengths = [len(o) for o in opts]
            mean_opt_len = np.mean(opt_lengths)
            
            ans = str(row.get('answer', ''))
            
            for i, letter in enumerate(['A', 'B', 'C', 'D']):
                opt = opts[i]
                
                # Text for TF-IDF (Adding letter to see if that helps TF-IDF indirectly)
                text = f"{cat} {letter} {q} {opt}"
                X_text.append(text)
                
                # Numeric / NLP Heuristic Features
                q_len = len(q)
                opt_len = len(opt)
                len_diff = opt_len - mean_opt_len
                len_ratio = opt_len / (q_len + 1e-5)
                
                opt_words = set(opt.lower().split())
                word_overlap = len(q_words.intersection(opt_words))
                jaccard = word_overlap / (len(q_words.union(opt_words)) + 1e-5)
                
                # Categorical letter index to learn base-rate biases!
                l_idx = letter_map[letter]
                
                X_num.append([cat_idx, l_idx, q_len, opt_len, len_diff, len_ratio, word_overlap, jaccard])
                
                if is_train:
                    label = 1 if letter in ans else 0
                    y.append(label)
                    
                qa_ids.append(row['qa_id'])
                letters.append(letter)
                
        return X_text, np.array(X_num), y, qa_ids, letters

    print("Extracting features from train set...")
    X_text_train, X_num_train, y_train, _, _ = extract_features(train_df, is_train=True)
    
    print("Extracting features from test set...")
    X_text_test, X_num_test, _, test_qa_ids, test_letters = extract_features(test_df, is_train=False)
    
    print("Vectorizing text (this might take a moment)...")
    # Using larger ngram range and max features
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=30000, sublinear_tf=True)
    X_tfidf_train = vectorizer.fit_transform(X_text_train)
    X_tfidf_test = vectorizer.transform(X_text_test)
    
    print("Training Level-1 Text Model (Logistic Regression on TF-IDF)...")
    text_clf = LogisticRegression(C=2.0, max_iter=1000, class_weight='balanced', solver='liblinear')
    text_clf.fit(X_tfidf_train, y_train)
    
    print("Generating text probabilities...")
    text_probs_train = text_clf.predict_proba(X_tfidf_train)[:, 1].reshape(-1, 1)
    text_probs_test = text_clf.predict_proba(X_tfidf_test)[:, 1].reshape(-1, 1)
    
    del X_tfidf_train
    del X_tfidf_test
    gc.collect()
    
    print("Training Level-2 Final Model (HistGradientBoosting)...")
    X_meta_train = np.hstack([X_num_train, text_probs_train])
    X_meta_test = np.hstack([X_num_test, text_probs_test])
    
    # Categorical features: category_idx (0) and letter_idx (1)
    gbm = HistGradientBoostingClassifier(
        max_iter=500, 
        learning_rate=0.03, 
        max_depth=12, 
        categorical_features=[0, 1],
        random_state=42,
        class_weight='balanced',
        l2_regularization=0.1
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
        
    print("Saving submission_ml_v6.csv...")
    pd.DataFrame(final_preds).to_csv('submission_ml_v6.csv', index=False)
    print("Done!")

if __name__ == '__main__':
    run_ml_pipeline_v6()
