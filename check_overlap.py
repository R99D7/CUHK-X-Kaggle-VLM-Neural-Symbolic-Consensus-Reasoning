"""
Check overlap of videos across all categories.
"""
import pandas as pd
from collections import defaultdict

te = pd.read_csv('test_qa.csv')
cat_vids = defaultdict(set)

for idx, row in te.iterrows():
    cat = row['category']
    vid = row['path']
    cat_vids[cat].add(vid)

categories = list(cat_vids.keys())
print("Video counts per category:")
for cat in categories:
    print(f"  {cat}: {len(cat_vids[cat])}")

print("\nOverlaps:")
for i in range(len(categories)):
    for j in range(i+1, len(categories)):
        c1, c2 = categories[i], categories[j]
        overlap = cat_vids[c1] & cat_vids[c2]
        print(f"  {c1} & {c2}: {len(overlap)}")
