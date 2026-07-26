import pandas as pd
train = pd.read_csv('training_qa.csv')
test = pd.read_csv('test_qa.csv')
v46 = pd.read_csv('submission_v46_aggressive_hybrid_v20.csv')

leaks = 0
mismatches = 0
for idx, row in test.iterrows():
    match = train[train['question'] == row['question']]
    if len(match) > 0:
        for _, m_row in match.iterrows():
            test_opts = {str(row['A']), str(row['B']), str(row['C']), str(row['D'])}
            train_opts = {str(m_row['A']), str(m_row['B']), str(m_row['C']), str(m_row['D'])}
            
            if test_opts == train_opts:
                true_ans_chars = str(m_row['answer'])
                
                expected_ans_chars = []
                for char in true_ans_chars:
                    if char in ['A', 'B', 'C', 'D']:
                        target_text = str(m_row[char])
                        for t_char in ['A', 'B', 'C', 'D']:
                            if str(row[t_char]) == target_text:
                                expected_ans_chars.append(t_char)
                                break
                
                # If it's single choice or combination (multi), we just sort them so the order doesn't matter
                # But if it's sequence, the order matters!
                if row['category'] != 'sequence':
                    expected_ans = ''.join(sorted(expected_ans_chars))
                else:
                    expected_ans = ''.join(expected_ans_chars)
                    
                actual_v46 = str(v46.loc[v46['qa_id'] == row['qa_id'], 'prediction'].values[0])
                
                if expected_ans != actual_v46:
                    print(f"LEAK MISMATCH! {row['qa_id']}: Expected {expected_ans}, v46 has {actual_v46} (Category: {row['category']})")
                    mismatches += 1
                leaks += 1
                break
print(f'Total leaks: {leaks}')
print(f'Total mismatches against v46: {mismatches}')
