"""
Check if object_interaction videos have multi questions.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
obj_vids = set(te[te['category'] == 'object_interaction']['path'])
multi_vids = set(te[te['category'] == 'multi']['path'])

overlap = obj_vids & multi_vids
print(f"Overlap between object_interaction and multi: {len(overlap)}")
