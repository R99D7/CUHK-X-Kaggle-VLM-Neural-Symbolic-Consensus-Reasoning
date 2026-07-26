"""
For the 39 test videos that have BOTH sequence AND emotion/multi/combination,
we know the exact 4 actions in the video (from sequence options).

Strategy: Find training videos with EXACTLY the same 4 actions and take
majority vote on emotion answers. Also validate this approach on train data.
"""
import pandas as pd
from collections import Counter, defaultdict

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

# Build training: video path -> set of 4 actions
tr_video_actions = {}
for idx, row in tr[tr['category'] == 'sequence'].iterrows():
    acts = frozenset([str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']])
    tr_video_actions[row['path']] = acts

# Build training emotion: video path -> emotion answer text
tr_video_emotion = {}
for idx, row in tr[tr['category'] == 'emotion'].iterrows():
    ans_l = str(row['answer']).strip()
    tr_video_emotion[row['path']] = str(row[ans_l]).strip().lower()

# Validate on training: when two train videos share EXACT same 4 actions, is emotion the same?
exact_same_count = 0
exact_same_emo = 0
for vid1, acts1 in tr_video_actions.items():
    for vid2, acts2 in tr_video_actions.items():
        if vid1 < vid2 and acts1 == acts2:
            exact_same_count += 1
            if vid1 in tr_video_emotion and vid2 in tr_video_emotion:
                if tr_video_emotion[vid1] == tr_video_emotion[vid2]:
                    exact_same_emo += 1

print(f'Train pairs with EXACT same 4 actions: {exact_same_count}')
print(f'Of those, same emotion: {exact_same_emo} ({exact_same_emo/exact_same_count:.1%} if count > 0)')
