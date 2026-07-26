import pandas as pd
train = pd.read_csv('training_qa.csv')
test = pd.read_csv('test_qa.csv')

def get_sig(row):
    q = str(row['question']).strip()
    opts = sorted([str(row['A']).strip(), str(row['B']).strip(), str(row['C']).strip(), str(row['D']).strip()])
    return q + '||' + '||'.join(opts)

train['sig'] = train.apply(get_sig, axis=1)
test['sig'] = test.apply(get_sig, axis=1)

overlap_sigs = set(train['sig']).intersection(set(test['sig']))
print(f"Permuted Overlaps: {len(overlap_sigs)}")

for sig in overlap_sigs:
    t_rows = train[train['sig'] == sig]
    te_rows = test[test['sig'] == sig]
    
    for _, t_row in t_rows.iterrows():
        t_ans_letter = t_row['answer']
        # For sequence tasks, answer might be 'BCAD'
        if len(t_ans_letter) == 1:
            t_ans_text = str(t_row[t_ans_letter]).strip()
            
            for _, te_row in te_rows.iterrows():
                # Find which letter in te_row matches t_ans_text
                for letter in ['A', 'B', 'C', 'D']:
                    if str(te_row[letter]).strip() == t_ans_text:
                        print(f"Mapped {t_row['qa_id']} ({t_ans_letter}) -> {te_row['qa_id']} ({letter})")
