import pandas as pd
train = pd.read_csv('training_qa.csv')
for vid in ['HAU/user20/4-2-2', 'HAU/user20/4-2-3']:
    q = train[(train['path'] == vid) & (train['category'] == 'emotion')]
    for _, row in q.iterrows():
        print(f"{vid} -> Answer: {row['answer']} ({row[row['answer']]})")
