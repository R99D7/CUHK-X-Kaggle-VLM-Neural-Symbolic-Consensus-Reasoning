"""
Now let's look at the HARn training structure more carefully.
The HARn training path is 'HARn/ACTION_NAME/user/clip'
For HARn test paths like 'large_model_track_test/LM_test_0001/Depth/Depth.mp4',
we can see LM_test_0001 -> 0001. 

Key insight: The HARn training data has 44 action categories.
Each test video shows ONE of these actions.
The single question asks "which action is performed?"
3 options are given (A, B, C), one is correct, D is NaN.

The training path contains the action name! So for any HARn training question,
the answer's text should match (approximately) the action folder name.

Let's map training option texts to the action folder names:
"""
import pandas as pd
from difflib import SequenceMatcher

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

harn_tr = tr[tr['source'] == 'HARn']
harn_te = te[te['source'] == 'HARn']
harn_tr_single = harn_tr[harn_tr['category'] == 'single']

# Build mapping: action folder -> answer texts seen in training
action_to_answers = {}
for idx, row in harn_tr_single.iterrows():
    parts = row['path'].split('/')
    action_folder = parts[1] if len(parts) > 1 else ''
    ans_l = str(row['answer']).strip()
    ans_text = str(row[ans_l]).strip().lower()
    if action_folder not in action_to_answers:
        action_to_answers[action_folder] = {}
    action_to_answers[action_folder][ans_text] = action_to_answers[action_folder].get(ans_text, 0) + 1

print("Action folder -> most common answer text:")
for folder, answers in sorted(action_to_answers.items()):
    best = max(answers, key=answers.get)
    print(f"  {folder}: '{best}' ({answers})")
