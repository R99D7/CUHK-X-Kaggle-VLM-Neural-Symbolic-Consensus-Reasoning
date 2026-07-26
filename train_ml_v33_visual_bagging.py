import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import catboost as cb
import gc

def run_ml_pipeline_v11():
    print("Loading data...")
    train_df = pd.read_csv('training_qa.csv')
    test_df = pd.read_csv('test_qa.csv')
    sample_df = pd.read_csv('sample_submission.csv')
    
    sample_preds = dict(zip(sample_df['qa_id'], sample_df['prediction']))
    
    print("Loading SentenceTransformer model (this might take a moment)...")
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def extract_features(df, is_train=True):
        X_text = []
        X_num = []
        y = []
        qa_ids = []
        letters = []
        raw_sentences = []
        
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
            
            # --- LOAD VISUAL FEATURES ---
            import os
            import torch
            qa_id = row['qa_id']
            vid_id = qa_id.split('_')[0] if '_' in qa_id and not qa_id.startswith('test') and not qa_id.startswith('train') else qa_id
            if len(vid_id) > 12 and qa_id.startswith('LM_test'):
                vid_id = '_'.join(qa_id.split('_')[:3])
                
            pt_path = f"video_features_resnet/{vid_id}_Depth_Color.pt"
            if not os.path.exists(pt_path):
                pt_path = f"video_features_resnet/{vid_id}.pt"
                
            if os.path.exists(pt_path):
                try:
                    vid_tensor = torch.load(pt_path) # [16, 2048]
                    vid_feat = vid_tensor.mean(dim=0).tolist() # [2048]
                except:
                    vid_feat = [0.0] * 2048
            else:
                vid_feat = [0.0] * 2048
            # ---------------------------
            
            for i, letter in enumerate(['A', 'B', 'C', 'D']):
                opt = opts[i]
                
                text = f"{cat} {letter} {q} {opt}"
                X_text.append(text)
                
                # Semantic sentence for embedding
                semantic_text = f"Category: {cat}. Question: {q}. Answer: {opt}"
                raw_sentences.append(semantic_text)
                
                q_len = len(q)
                opt_len = len(opt)
                len_diff = opt_len - mean_opt_len
                len_ratio = opt_len / (q_len + 1e-5)
                
                opt_words = set(opt.lower().split())
                word_overlap = len(q_words.intersection(opt_words))
                jaccard = word_overlap / (len(q_words.union(opt_words)) + 1e-5)
                
                l_idx = letter_map[letter]
                
                base_num = [cat_idx, l_idx, q_len, opt_len, len_diff, len_ratio, word_overlap, jaccard]
                X_num.append(base_num + vid_feat)
                
                if is_train:
                    label = 1 if letter in ans else 0
                    y.append(label)
                    
                qa_ids.append(row['qa_id'])
                letters.append(letter)
                
        return X_text, np.array(X_num), y, qa_ids, letters, raw_sentences

    print("Extracting numeric and text features...")
    X_text_train, X_num_train, y_train, train_qa_ids, train_letters, train_sentences = extract_features(train_df, is_train=True)
    X_text_test, X_num_test, _, test_qa_ids, test_letters, test_sentences = extract_features(test_df, is_train=False)
    
    print("Generating Neural Embeddings using SentenceTransformers...")
    embeddings_train = st_model.encode(train_sentences, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    embeddings_test = st_model.encode(test_sentences, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    
    print("Vectorizing text with TF-IDF (Word & Char n-grams)...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 4), analyzer='word', max_features=50000, sublinear_tf=True)
    X_tfidf_train = vectorizer.fit_transform(X_text_train)
    X_tfidf_test = vectorizer.transform(X_text_test)
    
    y_train_arr = np.array(y_train)
    
    print("Generating Level-1 OOF text predictions via 10-Fold CV...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    oof_lr = np.zeros(len(y_train_arr))
    oof_nb = np.zeros(len(y_train_arr))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_tfidf_train, y_train_arr)):
        X_tr, y_tr = X_tfidf_train[train_idx], y_train_arr[train_idx]
        X_va = X_tfidf_train[val_idx]
        
        lr = LogisticRegression(C=2.5, max_iter=1000, class_weight='balanced', solver='liblinear')
        lr.fit(X_tr, y_tr)
        oof_lr[val_idx] = lr.predict_proba(X_va)[:, 1]
        
        nb = MultinomialNB(alpha=0.5)
        nb.fit(X_tr, y_tr)
        oof_nb[val_idx] = nb.predict_proba(X_va)[:, 1]

    print("Training Full Level-1 Text Models for Test Set...")
    lr_full = LogisticRegression(C=2.5, max_iter=1000, class_weight='balanced', solver='liblinear')
    lr_full.fit(X_tfidf_train, y_train_arr)
    test_lr = lr_full.predict_proba(X_tfidf_test)[:, 1]
    
    nb_full = MultinomialNB(alpha=0.5)
    nb_full.fit(X_tfidf_train, y_train_arr)
    test_nb = nb_full.predict_proba(X_tfidf_test)[:, 1]
    
    del X_tfidf_train, X_tfidf_test
    gc.collect()
    
    print("Training Level-2 Meta Models (XGBoost, CatBoost, HistGBM)...")
    
    # Merge Numeric + OOF Probs + 384D Embeddings
    train_meta_np = np.column_stack([X_num_train, oof_lr, oof_nb, embeddings_train])
    test_meta_np = np.column_stack([X_num_test, test_lr, test_nb, embeddings_test])
    
    # Create column names
    num_cols = ['cat_idx', 'letter_idx', 'q_len', 'opt_len', 'len_diff', 'len_ratio', 'word_overlap', 'jaccard'] + [f'vid_{i}' for i in range(2048)]
    oof_cols = ['oof_lr', 'oof_nb']
    emb_cols = [f'emb_{i}' for i in range(embeddings_train.shape[1])]
    feature_names = num_cols + oof_cols + emb_cols
    
    df_meta_train = pd.DataFrame(train_meta_np, columns=feature_names)
    df_meta_test = pd.DataFrame(test_meta_np, columns=feature_names)
    
    # Convert categoricals to int
    df_meta_train['cat_idx'] = df_meta_train['cat_idx'].astype(int)
    df_meta_train['letter_idx'] = df_meta_train['letter_idx'].astype(int)
    df_meta_test['cat_idx'] = df_meta_test['cat_idx'].astype(int)
    df_meta_test['letter_idx'] = df_meta_test['letter_idx'].astype(int)

    # 1. HistGradientBoosting, XGBoost, CatBoost Bagging
    print('Training Bagged Meta Models (XGBoost, CatBoost, HistGBM)...')
    prob_gbm_total = np.zeros(len(df_meta_test))
    prob_xgb_total = np.zeros(len(df_meta_test))
    prob_cat_total = np.zeros(len(df_meta_test))
    seeds = [42, 123, 456, 789, 1024]
    
    for seed in seeds:
        print(f'  -> Seed {seed}...')
        gbm = HistGradientBoostingClassifier(
            max_iter=600, learning_rate=0.03, max_depth=12, 
            categorical_features=['cat_idx', 'letter_idx'], random_state=seed, class_weight='balanced', l2_regularization=0.1
        )
        gbm.fit(df_meta_train, y_train_arr)
        prob_gbm_total += gbm.predict_proba(df_meta_test)[:, 1]
        
        xgb_clf = xgb.XGBClassifier(
            n_estimators=600, learning_rate=0.03, max_depth=6,
            scale_pos_weight=(len(y_train_arr)-sum(y_train_arr))/sum(y_train_arr),
            random_state=seed, tree_method='hist'
        )
        xgb_clf.fit(df_meta_train, y_train_arr)
        prob_xgb_total += xgb_clf.predict_proba(df_meta_test)[:, 1]
        
        cat_clf = cb.CatBoostClassifier(
            iterations=600, learning_rate=0.03, depth=6,
            cat_features=['cat_idx', 'letter_idx'], auto_class_weights='Balanced',
            random_state=seed, verbose=False
        )
        cat_clf.fit(df_meta_train, y_train_arr)
        prob_cat_total += cat_clf.predict_proba(df_meta_test)[:, 1]
        
    prob_gbm = prob_gbm_total / len(seeds)
    prob_xgb = prob_xgb_total / len(seeds)
    prob_cat = prob_cat_total / len(seeds)
    
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
        
    print("Saving submission_ml_v33_visual_bagging.csv...")
    pd.DataFrame(final_preds).to_csv('submission_ml_v33_visual_bagging.csv', index=False)
    print("Done!")

if __name__ == '__main__':
    run_ml_pipeline_v11()
