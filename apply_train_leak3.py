"""
Apply unordered exact training set matches to submission.
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
    sig = get_sig(row)
    if sig not in tr_sigs:
        tr_sigs[sig] = []
    
    ans = str(row['answer'])
    
    # map answer letter(s) to texts
    ans_texts = []
    for char in ans:
        if char in 'ABCD':
            ans_texts.append(str(row[char]).strip().lower())
    
    # for sequence, order matters, so join with |
    tr_sigs[sig].append("|".join(ans_texts))

sub_dict = dict(zip(sub['qa_id'], sub['prediction']))
changes = 0

for idx, row in te.iterrows():
    sig = get_sig(row)
    if sig in tr_sigs:
        train_ans_texts = set(tr_sigs[sig])
        if len(train_ans_texts) == 1:
            correct_text_str = list(train_ans_texts)[0]
            correct_texts = correct_text_str.split("|")
            
            # Reconstruct the letter prediction for this test row
            pred_letters = []
            for txt in correct_texts:
                for letter in ['A', 'B', 'C', 'D']:
                    if str(row[letter]).strip().lower() == txt:
                        pred_letters.append(letter)
                        break
            
            if len(pred_letters) == len(correct_texts):
                # We successfully mapped all answer components
                correct_ans = "".join(pred_letters)
                # For multi, sort letters
                if row['category'] == 'multi' or row['category'] == 'combination':
                    correct_ans = "".join(sorted(correct_ans))
                    
                current_pred = sub_dict.get(row['qa_id'])
                
                # Check for combination vs multi sorting matching
                if row['category'] in ['multi', 'combination']:
                    current_pred = "".join(sorted(current_pred)) if current_pred else current_pred
                
                if current_pred != correct_ans:
                    print(f"{row['qa_id']} ({row['category']}): {current_pred} -> {correct_ans} (unordered training leak)")
                    sub.loc[sub['qa_id'] == row['qa_id'], 'prediction'] = correct_ans
                    changes += 1

print(f"\nTotal changes from training set leak: {changes}")
sub.to_csv('submission.csv', index=False)
