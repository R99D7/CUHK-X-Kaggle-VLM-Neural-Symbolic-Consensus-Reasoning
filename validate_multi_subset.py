"""
Emotion is NOT predictable from action set (11% match rate = essentially random).
Let's look at the multi questions differently.

For test videos with sequence, we know the 4 actions.
For the multi question on the SAME video, options should be a subset of those 4 actions.
Let's check how often THIS is true in training.
"""
import pandas as pd
from collections import Counter

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

# In TRAINING: videos with both sequence and multi
tr_video_actions = {}
for idx, row in tr[tr['category'] == 'sequence'].iterrows():
    acts = frozenset([str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']])
    tr_video_actions[row['path']] = acts

multi_checks = 0
multi_in_sequence = 0
multi_wrong = []

for idx, row in tr[tr['category'] == 'multi'].iterrows():
    vid = row['path']
    if vid in tr_video_actions:
        known_acts = tr_video_actions[vid]
        ans_letters = str(row['answer']).strip()
        ans_acts = set([str(row[l]).strip().lower() for l in ans_letters])
        
        multi_checks += 1
        if ans_acts.issubset(known_acts):
            multi_in_sequence += 1
        else:
            wrong = ans_acts - known_acts
            multi_wrong.append({'vid': vid, 'ans': ans_acts, 'known': known_acts, 'extra': wrong})

print(f'Train multi questions on same video as sequence: {multi_checks}')
print(f'Where multi answer IS subset of sequence actions: {multi_in_sequence} ({multi_in_sequence/multi_checks:.1%})')
print(f'Where multi answer is NOT a subset: {len(multi_wrong)}')
if multi_wrong:
    print('\nFirst 5 non-subset cases:')
    for w in multi_wrong[:5]:
        print(f"  vid={w['vid']}, answer={w['ans']}, seq_acts={w['known']}, extra={w['extra']}")
