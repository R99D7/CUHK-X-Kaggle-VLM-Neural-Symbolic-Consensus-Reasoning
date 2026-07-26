"""
Check pseudo sources.
"""
import pandas as pd
ps = pd.read_csv('pseudo_test_labels.csv')
print(ps['source'].value_counts())
