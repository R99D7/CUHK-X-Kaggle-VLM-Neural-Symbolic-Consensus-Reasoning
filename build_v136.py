import pandas as pd

# Load the two submissions
print("Loading submissions...")
qwen = pd.read_csv('submission_v135_qwen2vl_full.csv').set_index('qa_id')
v117 = pd.read_csv('submission_v117_ultimate_multimodal.csv').set_index('qa_id')
test_df = pd.read_csv('test_qa.csv').set_index('qa_id')

def onehot(pred):
    pred = str(pred).strip()
    d = {'A': 0.05, 'B': 0.05, 'C': 0.05, 'D': 0.05}
    valid_chars = [x for x in pred if x in 'ABCD']
    if not valid_chars:
        return d
    val = 0.85 / len(valid_chars)
    for c in valid_chars:
        d[c] = val
    return d

def decode_pred(scores, category, original_qwen, original_v117):
    """Decode probability scores to valid prediction string."""
    sorted_opts = sorted('ABCD', key=lambda k: scores[k], reverse=True)
    
    if category in ['single', 'emotion', 'object_interaction', 'combination']:
        return sorted_opts[0]
    elif category == 'sequence':
        # Soft probabilities for sequences are all 0.2125, so scores tie.
        # Fallback to the 60% model (Qwen2-VL)
        return original_qwen
    elif category == 'multi':
        selected = [k for k in 'ABCD' if scores[k] > 0.25]
        if not selected:
            selected = [sorted_opts[0]]
        if len(selected) > 3:
            selected = sorted_opts[:2]
        return ''.join(sorted(selected))
    return sorted_opts[0]

results = []
diff_count = 0
for qa_id in qwen.index:
    category = test_df.loc[qa_id, 'category']
    q_pred = str(qwen.loc[qa_id, 'prediction'])
    v_pred = str(v117.loc[qa_id, 'prediction'])
    
    p_qwen = onehot(q_pred)
    p_v117 = onehot(v_pred)
    
    avg_scores = {}
    for k in 'ABCD':
        avg_scores[k] = 0.6 * p_qwen[k] + 0.4 * p_v117[k]
        
    pred = decode_pred(avg_scores, category, q_pred, v_pred)
    
    if pred != q_pred:
        diff_count += 1
    
    results.append({'qa_id': qa_id, 'prediction': pred})

out = pd.DataFrame(results)
out.to_csv('submission_v136_qwen2vl_v117_blend.csv', index=False)
print(f"Saved submission_v136_qwen2vl_v117_blend.csv with {diff_count} modified answers from Qwen2-VL due to blending.")
