import pandas as pd

new_df = pd.read_csv('submission_new.csv')
v11_df = pd.read_csv('submission_ml_v11.csv')
s_df = pd.read_csv('sample_submission.csv')
t_df = pd.read_csv('test_qa.csv')

new_dict = dict(zip(new_df['qa_id'], new_df['prediction']))
v11_dict = dict(zip(v11_df['qa_id'], v11_df['prediction']))
s_dict = dict(zip(s_df['qa_id'], s_df['prediction']))
cat_dict = dict(zip(t_df['qa_id'], t_df['category']))

final_preds = []
for qa_id in new_dict:
    n_pred = str(new_dict.get(qa_id, 'A'))
    v_pred = str(v11_dict.get(qa_id, 'A'))
    expected_len = len(str(s_dict.get(qa_id, 'A')))
    cat = cat_dict.get(qa_id, 'single')
    
    if expected_len == 0: expected_len = 1
    if n_pred == 'nan': n_pred = 'A'
    if v_pred == 'nan': v_pred = 'A'
    
    n_chars = list(n_pred)
    v_chars = list(v_pred)
    
    if len(n_chars) == expected_len:
        final_pred = n_pred
    elif len(n_chars) < expected_len:
        new_chars = []
        for c in n_chars:
            if c not in new_chars: new_chars.append(c)
        for c in v_chars:
            if len(new_chars) < expected_len and c not in new_chars: new_chars.append(c)
        for c in ['A', 'B', 'C', 'D']:
            if len(new_chars) < expected_len and c not in new_chars: new_chars.append(c)
        if cat != 'sequence': new_chars.sort()
        final_pred = ''.join(new_chars)
    else:
        intersection = [c for c in n_chars if c in v_chars]
        new_chars = []
        for c in intersection:
            if c not in new_chars: new_chars.append(c)
        if len(new_chars) >= expected_len:
            new_chars = new_chars[:expected_len]
        else:
            for c in n_chars:
                if len(new_chars) < expected_len and c not in new_chars: new_chars.append(c)
        if cat != 'sequence': new_chars.sort()
        final_pred = ''.join(new_chars)
        
    final_preds.append({'qa_id': qa_id, 'prediction': final_pred})

pd.DataFrame(final_preds).to_csv('submission_new_fixed.csv', index=False)
print("Fixed formatting errors in new submission!")
