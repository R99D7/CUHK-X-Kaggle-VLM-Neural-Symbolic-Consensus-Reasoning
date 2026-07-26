import pandas as pd

v211 = pd.read_csv('submission_v211_CRACKED_657.csv')
test = pd.read_csv('test_qa.csv')
merged = pd.merge(test, v211, on='qa_id')

grouped = merged.groupby('path')

inconsistencies = 0

for path, group in grouped:
    # Get the combination or sequence answers
    comb_row = group[group['category'] == 'combination']
    seq_row = group[group['category'] == 'sequence']
    single_row = group[group['category'] == 'single']
    multi_row = group[group['category'] == 'multi']
    
    actions_in_video = set()
    
    # Extract actions from comb
    if len(comb_row) > 0:
        pred = comb_row.iloc[0]['prediction']
        # For combination, the answer is a single letter (e.g. 'B')
        if pd.notna(pred) and len(pred) == 1:
            ans_text = comb_row.iloc[0][pred]
            if pd.notna(ans_text):
                actions = [x.strip().lower() for x in str(ans_text).split(',')]
                actions_in_video.update(actions)
                
    # Extract actions from seq
    if len(seq_row) > 0:
        pred = seq_row.iloc[0]['prediction']
        # Sequence answer is e.g. 'BCAD'
        if pd.notna(pred):
            for char in pred:
                if char in ['A', 'B', 'C', 'D']:
                    ans_text = seq_row.iloc[0][char]
                    if pd.notna(ans_text):
                        actions_in_video.add(str(ans_text).strip().lower())
                        
    # Now check single
    if len(single_row) > 0 and len(actions_in_video) > 0:
        pred = single_row.iloc[0]['prediction']
        if pd.notna(pred) and len(pred) == 1:
            ans_text = str(single_row.iloc[0][pred]).strip().lower()
            if ans_text not in actions_in_video:
                print(f"Inconsistency in {path} (Single): Pred is '{ans_text}', but video actions are {actions_in_video}")
                inconsistencies += 1
                
    # Check multi
    if len(multi_row) > 0 and len(actions_in_video) > 0:
        pred = multi_row.iloc[0]['prediction']
        if pd.notna(pred):
            for char in pred:
                if char in ['A', 'B', 'C', 'D']:
                    ans_text = str(multi_row.iloc[0][char]).strip().lower()
                    if ans_text not in actions_in_video:
                        print(f"Inconsistency in {path} (Multi): Pred includes '{ans_text}', but video actions are {actions_in_video}")
                        inconsistencies += 1

print(f'Total inconsistencies found: {inconsistencies}')
