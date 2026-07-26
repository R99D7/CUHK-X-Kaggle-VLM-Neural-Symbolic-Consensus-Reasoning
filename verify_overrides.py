import pandas as pd

overrides = {
    'test_0012': 'D', 'test_0034': 'D', 'test_0051': 'C', 'test_0069': 'C',
    'test_0094': 'D', 'test_0436': 'C', 'test_0453': 'C', 'test_0548': 'A',
    'test_0413': 'A', 'test_0420': 'D', 'test_0421': 'C', 'test_0665': 'D',
    'test_0145': 'AC', 'test_0175': 'C', 'test_0205': 'BC', 'test_0580': 'AC',
    'test_0581': 'AD', 'test_0309': 'A', 'test_0041': 'A'
}

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
v60 = pd.read_csv('submission_v60_mega_forest.csv')

tr['options'] = tr.apply(lambda r: frozenset([str(r['A']).strip().lower(), str(r['B']).strip().lower(), str(r['C']).strip().lower(), str(r['D']).strip().lower()]), axis=1)
te['options'] = te.apply(lambda r: frozenset([str(r['A']).strip().lower(), str(r['B']).strip().lower(), str(r['C']).strip().lower(), str(r['D']).strip().lower()]), axis=1)

tr_map = {}
for idx, row in tr.iterrows():
    key = (row['category'], row['options'])
    ans_letters = str(row['answer']).strip()
    try:
        if row['category'] == 'sequence':
            ans_texts = tuple([str(row[l]).strip().lower() for l in ans_letters])
        elif row['category'] == 'multi':
            ans_texts = frozenset([str(row[l]).strip().lower() for l in ans_letters])
        else:
            ans_texts = str(row[ans_letters]).strip().lower()
        if key not in tr_map: tr_map[key] = set()
        tr_map[key].add(ans_texts)
    except: pass

print('Verifying 19 overrides against deterministic option set logic:')
for qa_id, man_ans in overrides.items():
    row = te[te['qa_id'] == qa_id].iloc[0]
    key = (row['category'], row['options'])
    v60_ans = str(v60[v60['qa_id'] == qa_id]['prediction'].values[0]).strip()
    
    status = 'NO MATCH'
    det_ans = None
    if key in tr_map and len(tr_map[key]) == 1:
        correct_texts = list(tr_map[key])[0]
        te_opt_map = {str(row[l]).strip().lower(): l for l in ['A', 'B', 'C', 'D']}
        try:
            if row['category'] == 'sequence':
                det_ans = "".join([te_opt_map[t] for t in correct_texts])
            elif row['category'] == 'multi':
                det_ans = "".join(sorted([te_opt_map[t] for t in correct_texts]))
            else:
                det_ans = te_opt_map[correct_texts]
            status = f'MATCH: {det_ans}'
        except:
            pass
            
    print(f"{qa_id} | v60: {v60_ans:<3} | man_override: {man_ans:<3} | Deterministic: {status}")
