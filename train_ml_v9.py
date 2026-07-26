import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import catboost as cb
import gc

def run_ml_pipeline_v9():
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
                
                text = f"{cat} {letter} {q} {opt}"
                X_text.append(text)
                
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

    print("Extracting features...")
    X_text_train, X_num_train, y_train, train_qa_ids, train_letters = extract_features(train_df, is_train=True)
    X_text_test, X_num_test, _, test_qa_ids, test_letters = extract_features(test_df, is_train=False)
    
    print("Vectorizing text with TF-IDF (Word & Char n-grams)...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 4), analyzer='word', max_features=50000, sublinear_tf=True)
    X_tfidf_train = vectorizer.fit_transform(X_text_train)
    X_tfidf_test = vectorizer.transform(X_text_test)
    
    y_train_arr = np.array(y_train)
    
    print("Generating Level-1 OOF text predictions via 10-Fold CV...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    oof_lr = np.zeros(len(y_train_arr))
    oof_nb = np.zeros(len(y_train_arr))
    oof_svm = np.zeros(len(y_train_arr))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_tfidf_train, y_train_arr)):
        X_tr, y_tr = X_tfidf_train[train_idx], y_train_arr[train_idx]
        X_va = X_tfidf_train[val_idx]
        
        lr = LogisticRegression(C=2.5, max_iter=1000, class_weight='balanced', solver='liblinear')
        lr.fit(X_tr, y_tr)
        oof_lr[val_idx] = lr.predict_proba(X_va)[:, 1]
        
        nb = MultinomialNB(alpha=0.5)
        nb.fit(X_tr, y_tr)
        oof_nb[val_idx] = nb.predict_proba(X_va)[:, 1]
        
        svm = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, max_iter=2000, class_weight='balanced', random_state=42)
        svm.fit(X_tr, y_tr)
        oof_svm[val_idx] = svm.predict_proba(X_va)[:, 1]

    print("Training Full Level-1 Text Models for Test Set...")
    lr_full = LogisticRegression(C=2.5, max_iter=1000, class_weight='balanced', solver='liblinear')
    lr_full.fit(X_tfidf_train, y_train_arr)
    test_lr = lr_full.predict_proba(X_tfidf_test)[:, 1]
    
    nb_full = MultinomialNB(alpha=0.5)
    nb_full.fit(X_tfidf_train, y_train_arr)
    test_nb = nb_full.predict_proba(X_tfidf_test)[:, 1]
    
    svm_full = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, max_iter=2000, class_weight='balanced', random_state=42)
    svm_full.fit(X_tfidf_train, y_train_arr)
    test_svm = svm_full.predict_proba(X_tfidf_test)[:, 1]
    
    del X_tfidf_train, X_tfidf_test
    gc.collect()
    
    print("Training Level-2 Meta Models (XGBoost, CatBoost, HistGBM)...")
    
    # Create pandas DataFrames to properly pass categorical types to CatBoost
    feature_names = ['cat_idx', 'letter_idx', 'q_len', 'opt_len', 'len_diff', 'len_ratio', 'word_overlap', 'jaccard', 'oof_lr', 'oof_nb', 'oof_svm']
    
    df_meta_train = pd.DataFrame(np.column_stack([X_num_train, oof_lr, oof_nb, oof_svm]), columns=feature_names)
    df_meta_test = pd.DataFrame(np.column_stack([X_num_test, test_lr, test_nb, test_svm]), columns=feature_names)
    
    # Convert categoricals to int for CatBoost and HistGBM
    df_meta_train['cat_idx'] = df_meta_train['cat_idx'].astype(int)
    df_meta_train['letter_idx'] = df_meta_train['letter_idx'].astype(int)
    df_meta_test['cat_idx'] = df_meta_test['cat_idx'].astype(int)
    df_meta_test['letter_idx'] = df_meta_test['letter_idx'].astype(int)

    # 1. HistGradientBoosting
    gbm = HistGradientBoostingClassifier(
        max_iter=700, learning_rate=0.02, max_depth=12, 
        categorical_features=['cat_idx', 'letter_idx'], random_state=42, class_weight='balanced', l2_regularization=0.2
    )
    gbm.fit(df_meta_train, y_train_arr)
    prob_gbm = gbm.predict_proba(df_meta_test)[:, 1]
    
    # 2. XGBoost
    xgb_clf = xgb.XGBClassifier(
        n_estimators=700, learning_rate=0.02, max_depth=6,
        scale_pos_weight=(len(y_train_arr)-sum(y_train_arr))/sum(y_train_arr),
        random_state=42, tree_method='hist'
    )
    xgb_clf.fit(df_meta_train, y_train_arr)
    prob_xgb = xgb_clf.predict_proba(df_meta_test)[:, 1]
    
    # 3. CatBoost
    cat_clf = cb.CatBoostClassifier(
        iterations=700, learning_rate=0.02, depth=6,
        cat_features=['cat_idx', 'letter_idx'], auto_class_weights='Balanced',
        random_state=42, verbose=False
    )
    cat_clf.fit(df_meta_train, y_train_arr)
    prob_cat = cat_clf.predict_proba(df_meta_test)[:, 1]
    
    # Ensemble the meta models
    print("Ensembling Meta Models...")
    probs = (prob_gbm + prob_xgb + prob_cat) / 3.0
    
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
        
    print("Saving submission_ml_v9.csv...")
    pd.DataFrame(final_preds).to_csv('submission_ml_v9.csv', index=False)
    print("Done!")

if __name__ == '__main__':
    run_ml_pipeline_v9()
