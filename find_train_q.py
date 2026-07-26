import pandas as pd
train = pd.read_csv('training_qa.csv')
vid_path = train[train['qa_id'] == 'training_2654']['path'].values[0]
print(f'Training video path: {vid_path}')
matches = train[train['path'] == vid_path]
for idx, row in matches.iterrows():
    print(f"{row['qa_id']} - {row['question']} -> Answer: {row['answer']}")
    print(f"  A:{row['A']} B:{row['B']} C:{row['C']} D:{row['D']}")
