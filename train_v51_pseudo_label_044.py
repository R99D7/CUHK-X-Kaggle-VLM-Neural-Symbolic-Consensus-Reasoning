import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sentence_transformers import SentenceTransformer
import catboost as cb
import gc
import torch
import os

def run_ml_pipeline():
    print("Loading data...")
    train_df = pd.read_csv('training_qa.csv')
    test_df = pd.read_csv('test_qa.csv')
    
    # Load the 0.44152 submission!
    pseudo = pd.read_csv('submission_v46_aggressive_hybrid_v20.csv')
    test_df = test_df.merge(pseudo[['qa_id', 'prediction']], on='qa_id')
    test_df.rename(columns={'prediction': 'answer'}, inplace=True)
    
    print("Extracting features...")
    train_df['text_combined'] = train_df['question'].fillna('') + ' ' + train_df['A'].fillna('') + ' ' + train_df['B'].fillna('') + ' ' + train_df['C'].fillna('') + ' ' + train_df['D'].fillna('')
    test_df['text_combined'] = test_df['question'].fillna('') + ' ' + test_df['A'].fillna('') + ' ' + test_df['B'].fillna('') + ' ' + test_df['C'].fillna('') + ' ' + test_df['D'].fillna('')
    
    train_df['answer_len'] = train_df['answer'].apply(lambda x: len(str(x)))
    test_df['answer_len'] = test_df['answer'].apply(lambda x: len(str(x)))
    
    train_single = train_df[train_df['answer_len'] == 1].copy()
    test_single = test_df[test_df['answer_len'] == 1].copy()
    
    print("Generating Neural Embeddings...")
    st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    X_train_text = st_model.encode(train_single['text_combined'].tolist())
    X_test_text = st_model.encode(test_single['text_combined'].tolist())
    
    print("Loading ResNet Video Features...")
    def get_resnet_features(qa_id):
        vid_id = qa_id.split('_')[0] if '_' in qa_id and not qa_id.startswith('test') and not qa_id.startswith('train') else qa_id
        if qa_id.startswith('LM_test'):
            vid_id = '_'.join(qa_id.split('_')[:3])
        pt_path = f"video_features_resnet/{vid_id}_Depth_Color.pt"
        if not os.path.exists(pt_path):
            pt_path = f"video_features_resnet/{vid_id}.pt"
        if os.path.exists(pt_path):
            vid_tensor = torch.load(pt_path, map_location='cpu') # [16, 2048]
            return vid_tensor.mean(dim=0).numpy()
        return np.zeros(2048)
        
    X_train_img = np.array([get_resnet_features(vid) for vid in train_single['qa_id']])
    X_test_img = np.array([get_resnet_features(vid) for vid in test_single['qa_id']])
    
    X_train = np.hstack([X_train_text, X_train_img])
    X_test = np.hstack([X_test_text, X_test_img])
    
    y_train = train_single['answer'].values
    y_test_pseudo = test_single['answer'].values
    
    # Combine train and pseudo-labeled test set
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test_pseudo])
    
    print("Training Models...")
    models = [
        ('hgb', HistGradientBoostingClassifier(max_iter=100, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('cb', cb.CatBoostClassifier(iterations=100, verbose=0, random_seed=42))
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
    
    # Merge back
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
    out.to_csv('submission_v51_pseudo_label.csv', index=False)
    
    diff = sum([1 for a, b in zip(out['prediction'], test_df['answer']) if str(a) != str(b)])
    print(f"Done! Differences from 0.44152 pseudo-labels: {diff}")

if __name__ == '__main__':
    run_ml_pipeline()
