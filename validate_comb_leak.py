"""
For combination questions on the SAME video as sequence:
We know the 4 actions. Each combination option is a pair of actions.
The correct answer is the option whose pair is BOTH in the known actions.
Validate this on training first.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

# Training: video path -> 4 actions
tr_video_actions = {}
for idx, row in tr[tr['category'] == 'sequence'].iterrows():
    acts = frozenset([str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']])
    tr_video_actions[row['path']] = acts

# Validate in training
comb_checks = 0
comb_correct = 0
comb_multiple_valid = 0
comb_no_valid = 0

for idx, row in tr[tr['category'] == 'combination'].iterrows():
    vid = row['path']
    if vid not in tr_video_actions:
        continue
    known_acts = tr_video_actions[vid]
    ans_l = str(row['answer']).strip()
    
    # Find which options are valid subsets
    valid_opts = []
    for l in ['A', 'B', 'C', 'D']:
        opt_text = str(row[l]).strip().lower()
        opt_acts = set([x.strip() for x in opt_text.split(',')])
        if opt_acts.issubset(known_acts):
            valid_opts.append(l)
    
    comb_checks += 1
    if len(valid_opts) == 1:
        if valid_opts[0] == ans_l:
            comb_correct += 1
    elif len(valid_opts) > 1:
        comb_multiple_valid += 1
    else:
        comb_no_valid += 1

print(f"Train combination questions with sequence: {comb_checks}")
print(f"Where exactly 1 valid option (subset of seq): accuracy = {comb_correct}")
print(f"Multiple valid options: {comb_multiple_valid}")
print(f"No valid options: {comb_no_valid}")

# Now apply to test
te_video_actions = {}
for idx, row in te[te['category'] == 'sequence'].iterrows():
    acts = frozenset([str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']])
    te_video_actions[row['path']] = acts

print("\nTest combination + sequence cases:")
changes = 0
for idx, row in te[te['category'] == 'combination'].iterrows():
    vid = row['path']
    if vid not in te_video_actions:
        continue
    known_acts = te_video_actions[vid]
    
    valid_opts = []
    for l in ['A', 'B', 'C', 'D']:
        opt_text = str(row[l]).strip().lower()
        opt_acts = set([x.strip() for x in opt_text.split(',')])
        if opt_acts.issubset(known_acts):
            valid_opts.append(l)
    
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    agree = "AGREE" if (len(valid_opts) == 1 and valid_opts[0] == pred) else "DISAGREE"
    
    if len(valid_opts) == 1:
        print(f"{row['qa_id']}: valid_opts={valid_opts}, pred={pred} [{agree}]")
        if valid_opts[0] != pred:
            changes += 1

print(f"\nChanges needed: {changes}")
