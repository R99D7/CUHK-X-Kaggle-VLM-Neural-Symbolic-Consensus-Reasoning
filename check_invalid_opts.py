"""
Check if our 0.69590 submission predicts any option that is NaN, empty, or obviously invalid.
"""
import pandas as pd
import numpy as np

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')
sub_map = dict(zip(sub['qa_id'], sub['prediction']))
te['prediction'] = te['qa_id'].map(sub_map)

for idx, row in te.iterrows():
    pred = str(row['prediction']).strip()
    for char in pred:
        if char in ['A', 'B', 'C', 'D']:
            opt_val = row[char]
            if pd.isna(opt_val) or str(opt_val).strip() == '' or str(opt_val).strip().lower() == 'nan':
                print(f"ALARM: {row['qa_id']} predicts {pred}, but option {char} is invalid ({opt_val})!")
                
print("Check completed.")
