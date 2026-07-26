"""
Check if object_interaction videos have combination questions.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
obj_vids = set(te[te['category'] == 'object_interaction']['path'])
comb_vids = set(te[te['category'] == 'combination']['path'])

overlap = obj_vids & comb_vids
print(f"Overlap between object_interaction and combination: {len(overlap)}")
