"""
Check exact leaks for all categories.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

def get_sig(row):
    opts = frozenset([str(row['A']).strip().lower(), str(row['B']).strip().lower(), 
                      str(row['C']).strip().lower(), str(row['D']).strip().lower()])
    return (row['category'], opts)

tr_sigs = {}
for idx, row in tr.iterrows():
    ans = str(row['answer'])
    if len(ans) > 1: continue # only process single letter answers
    
    sig = get_sig(row)
    if sig not in tr_sigs:
        tr_sigs[sig] = []
    
    ans_text = str(row[ans]).strip().lower()
    tr_sigs[sig].append(ans_text)

sub_dict = dict(zip(sub['qa_id'], sub['prediction']))

for cat in ['single', 'object_interaction', 'combination']:
    sub_correct = 0
    total_leaks = 0
    
    for idx, row in te.iterrows():
        if row['category'] != cat: continue
        
        sig = get_sig(row)
        if sig in tr_sigs:
            train_ans_texts = set(tr_sigs[sig])
            if len(train_ans_texts) == 1:
                correct_text = list(train_ans_texts)[0]
                
                correct_letter = None
                for letter in ['A', 'B', 'C', 'D']:
                    if str(row[letter]).strip().lower() == correct_text:
                        correct_letter = letter
                        break
                
                if correct_letter:
                    total_leaks += 1
                    if sub_dict.get(row['qa_id']) == correct_letter: sub_correct += 1
    
    print(f"Category {cat} leaks: {total_leaks}")
    print(f"Submission correct: {sub_correct}/{total_leaks}")
