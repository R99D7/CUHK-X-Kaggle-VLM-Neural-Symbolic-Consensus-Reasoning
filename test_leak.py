import pandas as pd
train = pd.read_csv('training_qa.csv')
test = pd.read_csv('test_qa.csv')
train['q_options'] = train['question'].fillna('') + train['A'].fillna('') + train['B'].fillna('') + train['C'].fillna('') + train['D'].fillna('')
test['q_options'] = test['question'].fillna('') + test['A'].fillna('') + test['B'].fillna('') + test['C'].fillna('') + test['D'].fillna('')
overlap = set(train['q_options']).intersection(set(test['q_options']))
for o in overlap:
    t_row = train[train['q_options'] == o].iloc[0]
    te_row = test[test['q_options'] == o].iloc[0]
    print(f"Train QA: {t_row['qa_id']} -> Answer: {t_row['answer']}")
    print(f"Test QA: {te_row['qa_id']}")
