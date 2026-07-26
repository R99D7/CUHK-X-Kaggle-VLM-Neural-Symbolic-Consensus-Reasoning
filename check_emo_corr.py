"""
Check emotion-action correlation in training.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

# Build mapping of video path -> emotion answer text and single answer text
emo_map = {}
single_map = {}

for _, row in tr.iterrows():
    ans = str(row['answer'])
    if len(ans) > 1: continue
    
    ans_text = str(row[ans]).strip().lower()
    
    if row['category'] == 'emotion':
        emo_map[row['path']] = ans_text
    elif row['category'] == 'single':
        single_map[row['path']] = ans_text

# Count pairs
pairs = {}
for path, emo in emo_map.items():
    if path in single_map:
        act = single_map[path]
        pair = (act, emo)
        pairs[pair] = pairs.get(pair, 0) + 1

# Print top pairs
print("Top action-emotion pairs in training:")
sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
for pair, count in sorted_pairs[:20]:
    print(f"{pair[0]} -> {pair[1]}: {count}")

