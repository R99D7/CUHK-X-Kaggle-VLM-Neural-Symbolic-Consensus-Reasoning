import pandas as pd
tr = pd.read_csv('training_qa.csv')

video_all_actions = {}
for idx, row in tr.iterrows():
    vid = row['path']
    ans_letters = str(row['answer']).strip()
    try:
        if row['category'] == 'sequence': ans_texts = tuple([str(row[l]).strip().lower() for l in ans_letters])
        elif row['category'] == 'multi': ans_texts = frozenset([str(row[l]).strip().lower() for l in ans_letters])
        else: ans_texts = str(row[ans_letters]).strip().lower()
        actions = set()
        if isinstance(ans_texts, tuple) or isinstance(ans_texts, frozenset):
            for t in ans_texts: actions.add(t)
        else:
            for t in ans_texts.split(','): actions.add(t.strip())
        if vid not in video_all_actions: video_all_actions[vid] = set()
        video_all_actions[vid].update(actions)
    except: pass

# Extract partial actions ONLY from sequence
partial_actions = {}
for idx, row in tr[tr['category'] == 'sequence'].iterrows():
    vid = row['path']
    opts = set([str(row['A']).strip().lower(), str(row['B']).strip().lower(), str(row['C']).strip().lower(), str(row['D']).strip().lower()])
    if vid not in partial_actions: partial_actions[vid] = set()
    partial_actions[vid].update(opts)

# Check if multi questions have options that occurred in the video but ARE NOT in the sequence options
bad = 0
for idx, row in tr[tr['category'] == 'multi'].iterrows():
    vid = row['path']
    if vid in partial_actions:
        true_all = video_all_actions[vid]
        known = partial_actions[vid]
        
        opts = [str(row['A']).strip().lower(), str(row['B']).strip().lower(), str(row['C']).strip().lower(), str(row['D']).strip().lower()]
        for o in opts:
            if o in true_all and o not in known:
                bad += 1
                break

print(f'Multi questions where an option happened but was not in sequence: {bad}')
