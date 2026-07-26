"""
Check for exact test set matches in training set.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')

def get_sig(row):
    # Sort options to match regardless of order, OR check exact match
    # Let's check exact match of all options
    opts = frozenset([str(row['A']).strip().lower(), str(row['B']).strip().lower(), 
                      str(row['C']).strip().lower(), str(row['D']).strip().lower()])
    return (row['category'], opts)

tr_sigs = {}
for idx, row in tr.iterrows():
    sig = get_sig(row)
    if sig not in tr_sigs:
        tr_sigs[sig] = []
    tr_sigs[sig].append(row)

matches = 0
for idx, row in te.iterrows():
    sig = get_sig(row)
    if sig in tr_sigs:
        print(f"Test {row['qa_id']} has exact option match in training!")
        matches += 1

print(f"Total test questions with exact option matches in training: {matches}")
