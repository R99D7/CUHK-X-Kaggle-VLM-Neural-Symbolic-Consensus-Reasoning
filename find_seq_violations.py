"""
Find sequence violations in submission.csv.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

rules = [
    ('combing hair', 'getting dressed'),
    ('wiping hands', 'getting dressed'),
    ('walking', 'sitting down'),
    ('walking', 'typing on a keyboard'),
    ('sitting down', 'typing on a keyboard'),
    ('sitting down', 'reading'),
    ('sitting down', 'turning a page'),
    ('walking', 'reading'),
    ('sitting down', 'massaging oneself'),
    ('walking', 'stretching'),
    ('walking', 'lying down'),
    ('sitting down', 'listening to the music with headphones'),
    ('sitting down', 'writing'),
    ('walking', 'listening to the music with headphones'),
    ('walking', 'turning a page'),
    ('walking', 'checking the time'),
    ('undressing', 'getting dressed'),
    ('sweeping', 'mopping'),
    ('walking', 'mopping'),
    ('brushing teeth', 'wiping hands'),
    ('brushing teeth', 'combing hair'),
    ('brushing teeth', 'getting dressed'),
    ('getting dressed', 'stretching'),
    ('washing face', 'brushing teeth'),
    ('sitting down', 'playing games'),
    ('grabbing utensils', 'eating'),
    ('pouring', 'drinking'),
    ('pouring', 'eating'),
    ('stirring', 'eating'),
    ('stirring', 'drinking'),
    ('taking medicine', 'drinking')
]

violations = 0
for idx, row in te[te['category'] == 'sequence'].iterrows():
    vid = row['path']
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    if len(pred) != 4: continue
    
    for i in range(4):
        for j in range(i+1, 4):
            act1 = opts.get(pred[i]) # act1 happens BEFORE act2 in our prediction
            act2 = opts.get(pred[j])
            
            # Check if this ordering VIOLATES a rule
            # A rule is (A, B) meaning A MUST happen before B.
            # If we predict act1 before act2, and there is a rule (act2, act1), that's a violation!
            if (act2, act1) in rules:
                print(f"SEQ {row['qa_id']}: predicted {pred[i]} ({act1}) BEFORE {pred[j]} ({act2}) -- VIOLATES {act2} -> {act1}")
                violations += 1

print(f"\nTotal sequence rule violations: {violations}")
