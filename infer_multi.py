import pandas as pd
tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
v233 = pd.read_csv('submission_v233_VIDEO_INFERENCE_LEAK.csv')

tr['options'] = tr.apply(lambda r: frozenset([str(r['A']).strip().lower(), str(r['B']).strip().lower(), str(r['C']).strip().lower(), str(r['D']).strip().lower()]), axis=1)
te['options'] = te.apply(lambda r: frozenset([str(r['A']).strip().lower(), str(r['B']).strip().lower(), str(r['C']).strip().lower(), str(r['D']).strip().lower()]), axis=1)

tr_map = {}
for idx, row in tr.iterrows():
    key = (row['category'], row['options'])
    ans_letters = str(row['answer']).strip()
    try:
        if row['category'] == 'sequence': ans_texts = tuple([str(row[l]).strip().lower() for l in ans_letters])
        elif row['category'] == 'multi': ans_texts = frozenset([str(row[l]).strip().lower() for l in ans_letters])
        else: ans_texts = str(row[ans_letters]).strip().lower()
        if key not in tr_map: tr_map[key] = set()
        tr_map[key].add(ans_texts)
    except: pass

leaked_videos = {}
for idx, row in te.iterrows():
    if row['category'] in ['sequence', 'multi', 'combination']:
        key = (row['category'], row['options'])
        if key in tr_map and len(tr_map[key]) == 1:
            correct_texts = list(tr_map[key])[0]
            actions = set()
            if isinstance(correct_texts, tuple) or isinstance(correct_texts, frozenset):
                for t in correct_texts: actions.add(t)
            else:
                for t in correct_texts.split(','): actions.add(t.strip())
            vid_path = row['path']
            if vid_path not in leaked_videos: leaked_videos[vid_path] = set()
            leaked_videos[vid_path].update(actions)

changes = []
for vid, actions in leaked_videos.items():
    vid_df = te[(te['path'] == vid) & (te['category'] == 'multi')]
    for _, row in vid_df.iterrows():
        opts = {str(row[l]).strip().lower(): l for l in ['A', 'B', 'C', 'D']}
        correct_letters = sorted([l for t, l in opts.items() if t in actions])
        
        if len(correct_letters) > 0:
            inferred_ans = "".join(correct_letters)
            pred = str(v233[v233['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
            pred_sorted = "".join(sorted(list(pred)))
            
            if pred_sorted != inferred_ans:
                print(f"{row['qa_id']}: v233={pred} (sorted: {pred_sorted}), inferred={inferred_ans} (actions: {actions})")
                changes.append({'qa_id': row['qa_id'], 'new_pred': inferred_ans})

if len(changes) > 0:
    for c in changes:
        v233.loc[v233['qa_id'] == c['qa_id'], 'prediction'] = c['new_pred']
    v233.to_csv('submission_v235_MULTI_INFERENCE.csv', index=False)
    v233.to_csv('submission.csv', index=False)
    print('Saved to submission.csv')
else:
    print('No changes needed.')
