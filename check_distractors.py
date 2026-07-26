import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
v231 = pd.read_csv('submission.csv') # Current best

tr_distractors = {}
for idx, row in tr.iterrows():
    if row['category'] in ['sequence', 'multi']: continue
    
    ans_letters = str(row['answer']).strip()
    try:
        ans_texts = str(row[ans_letters]).strip().lower()
        opts = [str(row['A']).strip().lower(), str(row['B']).strip().lower(), str(row['C']).strip().lower(), str(row['D']).strip().lower()]
        distractors = frozenset([o for o in opts if o != ans_texts])
        
        key = (row['category'], distractors)
        if key not in tr_distractors:
            tr_distractors[key] = set()
        tr_distractors[key].add(ans_texts)
    except:
        pass

matches = 0
disagreements = 0
changes = []

for idx, row in te.iterrows():
    if row['category'] in ['sequence', 'multi']: continue
    
    opts = [str(row['A']).strip().lower(), str(row['B']).strip().lower(), str(row['C']).strip().lower(), str(row['D']).strip().lower()]
    te_opt_map = {str(row[l]).strip().lower(): l for l in ['A', 'B', 'C', 'D']}
    
    for i in range(4):
        # Assume opts[i] is the correct answer
        test_distractors = frozenset([opts[j] for j in range(4) if j != i])
        key = (row['category'], test_distractors)
        
        if key in tr_distractors:
            # Found matching 3 distractors!
            if len(tr_distractors[key]) == 1:
                train_correct = list(tr_distractors[key])[0]
                # If the test option WE ASSUMED was correct (opts[i]) IS the SAME as the train correct answer!
                if opts[i] == train_correct:
                    # Then this is an EXACT option set match! Which we already handled.
                    pass
                else:
                    # The test options have 3 identical distractors to train, BUT the 4th option is DIFFERENT from the train correct answer!
                    # This means if train_correct is NOT in the test options, then train_correct wasn't chosen.
                    # Does the generator generate the same 3 distractors for a DIFFERENT correct answer?
                    # Or is opts[i] the NEW correct answer, and it coincidentally got the same 3 distractors?
                    # Since distractors are generated FROM the correct answer, two DIFFERENT correct answers shouldn't yield the EXACT SAME 3 distractors.
                    # UNLESS the 3 distractors are just a fixed pool for a category!
                    pass

print('Done')
