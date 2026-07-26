import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
v231 = pd.read_csv('submission.csv') # The current best

tr['options'] = tr.apply(lambda r: frozenset([str(r['A']).strip().lower(), str(r['B']).strip().lower(), str(r['C']).strip().lower(), str(r['D']).strip().lower()]), axis=1)
te['options'] = te.apply(lambda r: frozenset([str(r['A']).strip().lower(), str(r['B']).strip().lower(), str(r['C']).strip().lower(), str(r['D']).strip().lower()]), axis=1)

# Group train by category and options
tr_map = {}
for idx, row in tr.iterrows():
    key = (row['category'], row['options'])
    # The answer can be something like 'A', 'BC', 'CBDA'
    ans_letters = str(row['answer']).strip()
    
    # We map the letters to their actual text from the options
    try:
        if row['category'] == 'sequence':
            # Order matters!
            ans_texts = tuple([str(row[l]).strip().lower() for l in ans_letters])
        elif row['category'] == 'multi':
            # Order doesn't matter, but it's multiple
            ans_texts = frozenset([str(row[l]).strip().lower() for l in ans_letters])
        else:
            # single, combination, emotion (only 1 letter)
            ans_texts = str(row[ans_letters]).strip().lower()
            
        if key not in tr_map:
            tr_map[key] = set()
        tr_map[key].add(ans_texts)
    except:
        pass

matches = 0
disagreements = 0
changes = []

for idx, row in te.iterrows():
    key = (row['category'], row['options'])
    if key in tr_map:
        # If all train questions with these options have the SAME correct answer text(s)
        if len(tr_map[key]) == 1:
            correct_texts = list(tr_map[key])[0]
            
            te_opt_map = {str(row[l]).strip().lower(): l for l in ['A', 'B', 'C', 'D']}
            te_ans = None
            
            try:
                if row['category'] == 'sequence':
                    # Reconstruct the sequence string
                    te_ans = "".join([te_opt_map[t] for t in correct_texts])
                elif row['category'] == 'multi':
                    # Reconstruct the multi string, usually sorted alphabetically
                    te_ans = "".join(sorted([te_opt_map[t] for t in correct_texts]))
                else:
                    te_ans = te_opt_map[correct_texts]
            except Exception as e:
                pass
            
            if te_ans:
                matches += 1
                curr_pred = str(v231[v231['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
                if curr_pred != te_ans:
                    # BUT WAIT: for multi, sometimes the string is sorted. We sorted it above.
                    # Let's check if they are the same set of letters for multi
                    if row['category'] == 'multi':
                        if set(curr_pred) == set(te_ans):
                            continue # They match
                    
                    disagreements += 1
                    print(f"{row['qa_id']} ({row['category']}): v231={curr_pred}, leak={te_ans}")
                    changes.append({'qa_id': row['qa_id'], 'new_pred': te_ans})

print(f'\nTotal exact option set matches: {matches}')
print(f'Total disagreements with v231: {disagreements}')

if disagreements > 0:
    for c in changes:
        v231.loc[v231['qa_id'] == c['qa_id'], 'prediction'] = c['new_pred']
    v231.to_csv('submission_v232_OPTION_SET_LEAK.csv', index=False)
    v231.to_csv('submission.csv', index=False)
    print('Saved to submission.csv')
