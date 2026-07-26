import pandas as pd
import numpy as np

train = pd.read_csv('training_qa.csv')
train['answer'] = train['answer'].fillna('A').astype(str)

def get_longest(row):
    opts = [str(row['A']), str(row['B']), str(row['C']), str(row['D'])]
    idx = np.argmax([len(o) for o in opts])
    return chr(65 + idx)

def get_overlap(row):
    q = set(str(row['question']).lower().split())
    opts = [set(str(row[x]).lower().split()) for x in ['A','B','C','D']]
    idx = np.argmax([len(q.intersection(o)) for o in opts])
    return chr(65 + idx)

train['long'] = train.apply(get_longest, axis=1)
train['overlap'] = train.apply(get_overlap, axis=1)

print('Longest accuracy:', (train['long'] == train['answer']).mean())
print('Overlap accuracy:', (train['overlap'] == train['answer']).mean())

# Check if there is an exact string match leak across the entire training set!
ans_counts = train['answer'].value_counts()
print(ans_counts)
