import pandas as pd
tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
v237 = pd.read_csv('submission_v237_SELF_LEAK.csv')

tr['options'] = tr.apply(lambda r: frozenset([str(r['A']).strip().lower(), str(r['B']).strip().lower(), str(r['C']).strip().lower(), str(r['D']).strip().lower()]), axis=1)
te['options'] = te.apply(lambda r: frozenset([str(r['A']).strip().lower(), str(r['B']).strip().lower(), str(r['C']).strip().lower(), str(r['D']).strip().lower()]), axis=1)

tr_map = {}
for idx, row in tr.iterrows():
    if row['category'] != 'single': continue
    key = row['options']
    ans_letters = str(row['answer']).strip()
    try:
        ans_texts = str(row[ans_letters]).strip().lower()
        if key not in tr_map: tr_map[key] = set()
        tr_map[key].add(ans_texts)
    except: pass

changes = []
for idx, row in te.iterrows():
    if row['category'] != 'single': continue
    key = row['options']
    if key in tr_map and len(tr_map[key]) == 1:
        correct_text = list(tr_map[key])[0]
        te_opt_map = {str(row[l]).strip().lower(): l for l in ['A', 'B', 'C', 'D']}
        if correct_text in te_opt_map:
            correct_l = te_opt_map[correct_text]
            pred = str(v237[v237['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
            if pred != correct_l:
                print(f"{row['qa_id']}: v237={pred}, exact_match_inferred={correct_l}")
                changes.append({'qa_id': row['qa_id'], 'new_pred': correct_l})

print(f'Total single exact match corrections: {len(changes)}')
if len(changes) > 0:
    for c in changes:
        v237.loc[v237['qa_id'] == c['qa_id'], 'prediction'] = c['new_pred']
    v237.to_csv('submission_v238_SINGLE_EXACT_MATCH.csv', index=False)
    v237.to_csv('submission.csv', index=False)
    print('Saved to submission.csv')
else:
    print('No changes needed.')
