import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

train = pd.read_csv('training_qa.csv')
test = pd.read_csv('test_qa.csv')

def get_doc(row):
    return str(row['question']) + ' ' + str(row['A']) + ' ' + str(row['B']) + ' ' + str(row['C']) + ' ' + str(row['D'])

train['doc'] = train.apply(get_doc, axis=1).str.lower()
test['doc'] = test.apply(get_doc, axis=1).str.lower()

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
all_docs = train['doc'].tolist() + test['doc'].tolist()
vectorizer.fit(all_docs)

train_vecs = vectorizer.transform(train['doc'])
test_vecs = vectorizer.transform(test['doc'])
sim_matrix = cosine_similarity(test_vecs, train_vecs)

for i in range(len(test)):
    t_row = test.iloc[i]
    best_tr_idx = np.argmax(sim_matrix[i])
    best_score = sim_matrix[i][best_tr_idx]
    if 0.85 < best_score < 0.99:
        tr_row = train.iloc[best_tr_idx]
        print(f"TEST: {t_row['qa_id']} | TRAIN: {tr_row['qa_id']} | Score: {best_score:.3f}")
        print(f"  T_DOC: {t_row['doc']}")
        print(f"  TR_DOC: {tr_row['doc']}")
        print('---')
