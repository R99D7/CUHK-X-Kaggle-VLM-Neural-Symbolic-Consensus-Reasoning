"""
Check the length distribution of multi answers in training.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
multi = tr[tr['category'] == 'multi']

lengths = multi['answer'].apply(lambda x: len(str(x).strip()))
print(lengths.value_counts())
