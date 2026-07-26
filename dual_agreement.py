import pandas as pd

cnn = pd.read_csv('transformer_fixed_raw_predictions.csv')
cross = pd.read_csv('crossencoder_raw_predictions.csv')
v46 = pd.read_csv('submission_v46_aggressive_hybrid_v20.csv')

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
            intersections.append({
                'qa_id': qa_id,
                'v46': v46_pred,
                'override': best_opt_cnn,
                'cnn_prob': best_prob_cnn,
                'cross_prob': best_prob_cross
            })

print(f'Total Dual-Agreement Overrides: {len(intersections)}')
intersections.sort(key=lambda x: x['cnn_prob'] + x['cross_prob'], reverse=True)
for o in intersections[:10]:
    print(f"{o['qa_id']}: v46={o['v46']} -> Override={o['override']} (CNN: {o['cnn_prob']:.3f}, Cross: {o['cross_prob']:.3f})")

# Generate the actual submission file!
final_preds = []
for idx, row in merged.iterrows():
    qa_id = row['qa_id']
    v46_pred = str(row['prediction'])
    
    if len(v46_pred) != 1:
        final_preds.append({'qa_id': qa_id, 'prediction': v46_pred})
        continue
        
    probs_cnn = {'A': row['raw_prob_A_cnn'], 'B': row['raw_prob_B_cnn'], 'C': row['raw_prob_C_cnn'], 'D': row['raw_prob_D_cnn']}
    best_opt_cnn = max(probs_cnn, key=probs_cnn.get)
    best_prob_cnn = probs_cnn[best_opt_cnn]
    
    probs_cross = {'A': row['raw_prob_A_cross'], 'B': row['raw_prob_B_cross'], 'C': row['raw_prob_C_cross'], 'D': row['raw_prob_D_cross']}
    best_opt_cross = max(probs_cross, key=probs_cross.get)
    best_prob_cross = probs_cross[best_opt_cross]
    
    if best_opt_cnn == best_opt_cross and best_opt_cnn != v46_pred and best_prob_cnn > 0.50 and best_prob_cross > 0.50:
        final_preds.append({'qa_id': qa_id, 'prediction': best_opt_cnn})
    else:
        final_preds.append({'qa_id': qa_id, 'prediction': v46_pred})

pd.DataFrame(final_preds).to_csv('submission_v113_dual_agreement_override.csv', index=False)
