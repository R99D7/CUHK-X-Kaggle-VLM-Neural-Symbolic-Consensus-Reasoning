import pandas as pd

friend = pd.read_csv('submission_friend.csv')
v11 = pd.read_csv('submission_ml_v11.csv')
sample = pd.read_csv('sample_submission.csv')

f_dict = dict(zip(friend['qa_id'], friend['prediction']))
v11_dict = dict(zip(v11['qa_id'], v11['prediction']))
s_dict = dict(zip(sample['qa_id'], sample['prediction']))

final_preds = []

for qa_id in f_dict:
    f_pred = str(f_dict.get(qa_id, 'A'))
    v_pred = str(v11_dict.get(qa_id, 'A'))
    expected_len = len(str(s_dict.get(qa_id, 'A')))
    
    if expected_len == 0:
        expected_len = 1
        
    f_chars = list(f_pred)
    v_chars = list(v_pred)
    
    if len(f_chars) == expected_len:
        final_pred = f_pred
    elif len(f_chars) < expected_len:
        # Friend is too short. Keep friend's letters, and borrow from V11 to fill the gap.
        new_chars = set(f_chars)
        for char in v_chars:
            if len(new_chars) < expected_len:
                new_chars.add(char)
        # If still too short (V11 didn't have enough unique chars), fallback to alphabetical
        for char in ['A', 'B', 'C', 'D']:
            if len(new_chars) < expected_len:
                new_chars.add(char)
        final_pred = "".join(sorted(list(new_chars)))
    else:
        # Friend is too long. Keep only the letters that V11 also predicted.
        intersection = set(f_chars).intersection(set(v_chars))
        new_chars = set(intersection)
        
        # If intersection is exactly expected len
        if len(new_chars) == expected_len:
            pass
        elif len(new_chars) > expected_len:
            # Still too long? Just take the first few alphabetically
            new_chars = set(sorted(list(new_chars))[:expected_len])
        else:
            # Intersection is too short! Borrow from friend first, then v11
            for char in f_chars:
                if len(new_chars) < expected_len:
                    new_chars.add(char)
            for char in v_chars:
                if len(new_chars) < expected_len:
                    new_chars.add(char)
                    
        final_pred = "".join(sorted(list(new_chars)))
        
    # Edge case sequence?
    # Sequence doesn't strictly sort alphabetically, but our previous models just output standard stuff.
    # Actually, let's keep sequence order if it's sequence. Wait, how do we know it's sequence?
    # Just read test_qa.csv
    
final_preds = []
test_df = pd.read_csv('test_qa.csv')
cat_dict = dict(zip(test_df['qa_id'], test_df['category']))

for qa_id in f_dict:
    f_pred = str(f_dict.get(qa_id, 'A'))
    v_pred = str(v11_dict.get(qa_id, 'A'))
    expected_len = len(str(s_dict.get(qa_id, 'A')))
    cat = cat_dict.get(qa_id, 'single')
    
    if expected_len == 0: expected_len = 1
    
    # Clean NaNs
    if f_pred == 'nan': f_pred = 'A'
    if v_pred == 'nan': v_pred = 'A'
        
    f_chars = list(f_pred)
    v_chars = list(v_pred)
    
    if len(f_chars) == expected_len:
        final_pred = f_pred
    elif len(f_chars) < expected_len:
        new_chars = []
        for c in f_chars:
            if c not in new_chars: new_chars.append(c)
        for c in v_chars:
            if len(new_chars) < expected_len and c not in new_chars:
                new_chars.append(c)
        for c in ['A', 'B', 'C', 'D']:
            if len(new_chars) < expected_len and c not in new_chars:
                new_chars.append(c)
                
        if cat != 'sequence':
            new_chars.sort()
        final_pred = "".join(new_chars)
    else:
        # Too long
        intersection = [c for c in f_chars if c in v_chars]
        new_chars = []
        for c in intersection:
            if c not in new_chars: new_chars.append(c)
            
        if len(new_chars) >= expected_len:
            new_chars = new_chars[:expected_len]
        else:
            for c in f_chars:
                if len(new_chars) < expected_len and c not in new_chars:
                    new_chars.append(c)
                    
        if cat != 'sequence':
            new_chars.sort()
        final_pred = "".join(new_chars)
        
    final_preds.append({'qa_id': qa_id, 'prediction': final_pred})

pd.DataFrame(final_preds).to_csv('submission_smart_length.csv', index=False)
print("Smart length correction saved!")
