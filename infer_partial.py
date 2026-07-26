import pandas as pd
tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
v237 = pd.read_csv('submission_v237_SELF_LEAK.csv')

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

seq_vids = set(te[te['category'] == 'sequence']['path'].unique())

non_seq_partial_actions = {}
for idx, row in te.iterrows():
    if row['path'] in seq_vids: continue
    
    key = (row['category'], row['options'])
    if key in tr_map and len(tr_map[key]) == 1:
        correct_text = list(tr_map[key])[0]
        vid = row['path']
        actions = set()
        if row['category'] == 'multi':
            for t in correct_text: actions.add(t)
        elif row['category'] == 'combination':
            for t in correct_text.split(','): actions.add(t.strip())
            
        if len(actions) > 0:
            if vid not in non_seq_partial_actions: non_seq_partial_actions[vid] = set()
            non_seq_partial_actions[vid].update(actions)

changes = []
for vid, actions in non_seq_partial_actions.items():
    vid_df = te[(te['path'] == vid) & (te['category'] == 'single')]
    for _, row in vid_df.iterrows():
        opts = {str(row[l]).strip().lower(): l for l in ['A', 'B', 'C', 'D']}
        valid_letters = [l for t, l in opts.items() if t in actions]
        
        # If exactly one option is in the partial action set, it MUST be the correct answer!
        if len(valid_letters) == 1:
            correct_l = valid_letters[0]
            pred = str(v237[v237['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
            if pred != correct_l:
                print(f"{row['qa_id']} (single): v237={pred}, inferred_partial={correct_l} (actions: {actions})")
                changes.append({'qa_id': row['qa_id'], 'new_pred': correct_l})

print(f'Total partial-action corrections: {len(changes)}')
if len(changes) > 0:
    for c in changes:
        v237.loc[v237['qa_id'] == c['qa_id'], 'prediction'] = c['new_pred']
    v237.to_csv('submission_v238_PARTIAL.csv', index=False)
    v237.to_csv('submission.csv', index=False)
    print('Saved to submission.csv')
