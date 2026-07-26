import pandas as pd

train = pd.read_csv('training_qa.csv')
test = pd.read_csv('test_qa.csv')
v46 = pd.read_csv('submission_v46_aggressive_hybrid_v20.csv')
cnn = pd.read_csv('transformer_fixed_raw_predictions.csv')
cross = pd.read_csv('crossencoder_raw_predictions.csv')

leaks_dict = {}
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
                if row['category'] != 'sequence':
                    expected_ans = ''.join(sorted(expected_ans_chars))
                else:
                    expected_ans = ''.join(expected_ans_chars)
                leaks_dict[row['qa_id']] = expected_ans
                break

merged = pd.merge(cnn, cross, on='qa_id', suffixes=('_cnn', '_cross'))
merged = pd.merge(merged, v46, on='qa_id')

intersections = []
for idx, row in merged.iterrows():
    qa_id = row['qa_id']
    v46_pred = str(row['prediction'])
    if len(v46_pred) != 1: continue
        
    probs_cnn = {'A': row['raw_prob_A_cnn'], 'B': row['raw_prob_B_cnn'], 'C': row['raw_prob_C_cnn'], 'D': row['raw_prob_D_cnn']}
    best_opt_cnn = max(probs_cnn, key=probs_cnn.get)
    best_prob_cnn = probs_cnn[best_opt_cnn]
    
    probs_cross = {'A': row['raw_prob_A_cross'], 'B': row['raw_prob_B_cross'], 'C': row['raw_prob_C_cross'], 'D': row['raw_prob_D_cross']}
    best_opt_cross = max(probs_cross, key=probs_cross.get)
    best_prob_cross = probs_cross[best_opt_cross]
    
    if best_opt_cnn == best_opt_cross and best_opt_cnn != v46_pred:
        if best_prob_cnn > 0.50 and best_prob_cross > 0.50:
            if qa_id in leaks_dict:
                print(f"REJECTED LEAK OVERRIDE: {qa_id}. v46={v46_pred}, Override={best_opt_cnn}, True Leak={leaks_dict[qa_id]}")
            else:
                intersections.append({'qa_id': qa_id, 'v46': v46_pred, 'override': best_opt_cnn, 'cnn': best_prob_cnn, 'cross': best_prob_cross})

print(f'Total safe overrides: {len(intersections)}')
intersections.sort(key=lambda x: x['cnn'] + x['cross'], reverse=True)
for o in intersections:
    print(f"{o['qa_id']}: v46={o['v46']} -> Override={o['override']} (CNN: {o['cnn']:.3f}, Cross: {o['cross']:.3f})")

# Write the final safe file
final_preds = []
for idx, row in merged.iterrows():
    qa_id = row['qa_id']
    v46_pred = str(row['prediction'])
    
    override_dict = {x['qa_id']: x['override'] for x in intersections}
    
    if qa_id in override_dict:
        final_preds.append({'qa_id': qa_id, 'prediction': override_dict[qa_id]})
    else:
        final_preds.append({'qa_id': qa_id, 'prediction': v46_pred})

pd.DataFrame(final_preds).to_csv('submission_v114_ultimate_safe_dual_agreement.csv', index=False)
