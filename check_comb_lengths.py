"""
Check the length distribution of combination answers in training.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
comb = tr[tr['category'] == 'combination']

lengths = comb['answer'].apply(lambda x: len(str(x).strip()))
print(lengths.value_counts())
