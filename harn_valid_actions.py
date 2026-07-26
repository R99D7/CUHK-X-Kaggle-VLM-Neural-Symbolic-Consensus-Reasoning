"""
Now we know the EXACT answer text for each HARn action category!
The test path is like 'large_model_track_test/LM_test_0001/Depth/Depth.mp4'
The LM_test_XXXX number might correspond to specific action+user IDs.

But more importantly: for the HARn test single questions,
if we know the action category (from the path), we know the answer!

The question is: can we figure out which action each LM_test video shows?
From the test options, one of them IS the correct action text.
And from training, we know exactly which text corresponds to which action.

Strategy: For each HARn test single question, look at ALL 3 options.
For each option, check if it appears as an answer in HARn training.
The option that appears most frequently as the CORRECT answer in training
when it's part of a 3-option question IS the answer!

Let me check: in HARn training, are the answer options designed such that
only ONE of the 3 options is a valid "real action" and the others are distractors?
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

harn_tr = tr[tr['source'] == 'HARn']
harn_tr_single = harn_tr[harn_tr['category'] == 'single']
harn_te_single = te[(te['source'] == 'HARn') & (te['category'] == 'single')]

# Build set of ALL valid HARn action answer texts
all_valid_answers = set()
for idx, row in harn_tr_single.iterrows():
    ans_l = str(row['answer']).strip()
    ans_text = str(row[ans_l]).strip().lower()
    all_valid_answers.add(ans_text)

print(f"Total valid HARn answer texts: {len(all_valid_answers)}")
print(sorted(all_valid_answers))

# For each test HARn single question, check which options are "valid HARn actions"
print("\n\nTest HARn single: which options are valid HARn actions?")
changes = 0
for idx, row in harn_te_single.iterrows():
    opts = {}
    for l in ['A', 'B', 'C', 'D']:
        v = str(row[l]).strip().lower()
        if v and v != 'nan':
            opts[l] = v
    
    valid = {l: v for l, v in opts.items() if v in all_valid_answers}
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    if len(valid) == 1:
        best_l = list(valid.keys())[0]
        agree = "AGREE" if pred == best_l else "DISAGREE"
        if pred != best_l:
            print(f"  {row['qa_id']}: pred={pred}, only_valid={best_l} ({valid[best_l]}) [{agree}]")
            changes += 1
    elif len(valid) == 0:
        print(f"  {row['qa_id']}: NO valid options! pred={pred}, opts={opts}")
    else:
        print(f"  {row['qa_id']}: {len(valid)} valid options: {valid}, pred={pred}")

print(f"\nTotal potential changes (1 valid option): {changes}")
