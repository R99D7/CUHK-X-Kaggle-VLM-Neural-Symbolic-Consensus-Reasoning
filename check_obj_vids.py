"""
Check what questions exist in test_qa.csv for the 21 videos with object_interaction.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
te_obj = te[te['category'] == 'object_interaction']
vids = set(te_obj['path'])

te_for_vids = te[te['path'].isin(vids)]
print("Categories present for the 21 videos with object_interaction:")
print(te_for_vids.groupby('path')['category'].apply(list))
