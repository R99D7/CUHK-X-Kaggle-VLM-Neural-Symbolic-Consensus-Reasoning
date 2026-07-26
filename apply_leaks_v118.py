import pandas as pd

def apply_leaks():
    print("Loading original v118 submission (0.45321)...")
    sub = pd.read_csv('submission_v118_ultimate_multimodal_055.csv')
    test = pd.read_csv('test_qa.csv')
    train = pd.read_csv('training_qa.csv')
    
    # Extract Data Leaks
    leaks_dict = {}
    for idx, row in test.iterrows():
        match = train[train['question'] == row['question']]
        if len(match) > 0:
            for _, m_row in match.iterrows():
                test_opts = {str(row['A']), str(row['B']), str(row['C']), str(row['D'])}
                train_opts = {str(m_row['A']), str(m_row['B']), str(m_row['C']), str(m_row['D'])}
                if test_opts == train_opts:
                    leaks_dict[row['qa_id']] = m_row['answer']
                    break
                    
    print(f"Verified {len(leaks_dict)} data leaks.")
    
    # Count how many leaks differ from v118's prediction
    diff_count = 0
    for idx, row in sub.iterrows():
        qa_id = row['qa_id']
        if qa_id in leaks_dict:
            if str(leaks_dict[qa_id]) != str(row['prediction']):
                diff_count += 1
            sub.at[idx, 'prediction'] = leaks_dict[qa_id]
            
    print(f"Fixed {diff_count} incorrect predictions in the v118 submission!")
    
    sub.to_csv('submission_v127_v118_leaks_fixed.csv', index=False)
    print("Saved submission_v127_v118_leaks_fixed.csv")

if __name__ == '__main__':
    apply_leaks()
