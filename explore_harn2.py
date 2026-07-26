"""
Look at HARn source more carefully - what categories do they have?
And check if there's a specific structural leak unique to HARn.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

harn_te = te[te['source'] == 'HARn']
print("HARn test categories:")
print(harn_te['category'].value_counts())

harn_tr = tr[tr['source'] == 'HARn']
print("\nHARn train categories:")
print(harn_tr['category'].value_counts())

print("\nHARn test paths (first 5):")
print(harn_te['path'].unique()[:5])

print("\nHARn train paths (first 5):")
print(harn_tr['path'].unique()[:5])

# Check if any HARn test videos match HARn train videos
harn_tr_vids = set(harn_tr['path'].unique())
harn_te_vids = set(harn_te['path'].unique())
overlap = harn_tr_vids.intersection(harn_te_vids)
print(f"\nHARn test-train video overlap: {len(overlap)}")

# Check if test paths contain the test video ID
print("\nSample test paths:")
print(harn_te['path'].head(10).values)
