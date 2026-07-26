import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
import gc

def run_ml_pipeline_v7():
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
        
        cat_map = {'single': 0, 'multi': 1, 'emotion': 2, 'combination': 3, 'sequence': 4, 'object_interaction': 5}
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
                
                # Text for TF-IDF
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
                
                l_idx = letter_map[letter]
                
                X_num.append([cat_idx, l_idx, q_len, opt_len, len_diff, len_ratio, word_overlap, jaccard])
                
                if is_train:
                    label = 1 if letter in ans else 0
                    y.append(label)
                    
                qa_ids.append(row['qa_id'])
                letters.append(letter)
                
        return X_text, np.array(X_num), y, qa_ids, letters

    print("Extracting features from train set...")
    X_text_train, X_num_train, y_train, train_qa_ids, train_letters = extract_features(train_df, is_train=True)
    
    print("Extracting features from test set...")
    X_text_test, X_num_test, _, test_qa_ids, test_letters = extract_features(test_df, is_train=False)
    
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=40000, sublinear_tf=True)
    X_tfidf_train = vectorizer.fit_transform(X_text_train)
    X_tfidf_test = vectorizer.transform(X_text_test)
    
    y_train_arr = np.array(y_train)
    
    print("Generating Out-Of-Fold (OOF) text predictions for Stacking...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_lr = np.zeros(len(y_train_arr))
    oof_nb = np.zeros(len(y_train_arr))
    oof_svm = np.zeros(len(y_train_arr))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_tfidf_train, y_train_arr)):
        print(f"  Training Fold {fold+1}...")
        X_tr, y_tr = X_tfidf_train[train_idx], y_train_arr[train_idx]
        X_va = X_tfidf_train[val_idx]
        
        lr = LogisticRegression(C=2.0, max_iter=500, class_weight='balanced', solver='liblinear')
        lr.fit(X_tr, y_tr)
        oof_lr[val_idx] = lr.predict_proba(X_va)[:, 1]
        
        nb = MultinomialNB()
        nb.fit(X_tr, y_tr)
        oof_nb[val_idx] = nb.predict_proba(X_va)[:, 1]
        
        svm = SGDClassifier(loss='log_loss', max_iter=1000, class_weight='balanced', random_state=42)
        svm.fit(X_tr, y_tr)
        oof_svm[val_idx] = svm.predict_proba(X_va)[:, 1]

    print("Training Full Text Models for Test Set...")
    # Train on full train data for test set predictions
    lr_full = LogisticRegression(C=2.0, max_iter=500, class_weight='balanced', solver='liblinear')
    lr_full.fit(X_tfidf_train, y_train_arr)
    test_lr = lr_full.predict_proba(X_tfidf_test)[:, 1]
    
    nb_full = MultinomialNB()
    nb_full.fit(X_tfidf_train, y_train_arr)
    test_nb = nb_full.predict_proba(X_tfidf_test)[:, 1]
    
    svm_full = SGDClassifier(loss='log_loss', max_iter=1000, class_weight='balanced', random_state=42)
    svm_full.fit(X_tfidf_train, y_train_arr)
    test_svm = svm_full.predict_proba(X_tfidf_test)[:, 1]
    
    del X_tfidf_train
    del X_tfidf_test
    gc.collect()
    
    print("Training Level-2 Final Model (HistGradientBoosting)...")
    # Stack the 3 text probabilities with the numeric features
    X_meta_train = np.column_stack([X_num_train, oof_lr, oof_nb, oof_svm])
    X_meta_test = np.column_stack([X_num_test, test_lr, test_nb, test_svm])
    
    gbm = HistGradientBoostingClassifier(
        max_iter=500, 
        learning_rate=0.03, 
        max_depth=12, 
        categorical_features=[0, 1], # category index and letter index
        random_state=42,
        class_weight='balanced',
        l2_regularization=0.1
    )
    gbm.fit(X_meta_train, y_train_arr)
    
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
        
    print("Saving submission_ml_v7.csv...")
    pd.DataFrame(final_preds).to_csv('submission_ml_v7.csv', index=False)
    print("Done!")

if __name__ == '__main__':
    run_ml_pipeline_v7()
