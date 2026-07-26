import pandas as pd
train = pd.read_csv('training_qa.csv')
for q_id in ['training_2654', 'training_2655', 'training_2656']:
    row = train[train['qa_id'] == q_id].iloc[0]
    print(f"{q_id} -> Ans: {row['answer']} (A:{row['A']} B:{row['B']} C:{row['C']} D:{row['D']})")
