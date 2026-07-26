"""
Simple and Clean Ensemble - v137
Learned from previous mistakes - no Unicode characters, direct execution
"""
import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SIMPLE CLEAN ENSEMBLE - v137")
print("=" * 80)

# Load prediction sources
print("\n[1/6] Loading prediction sources...")
v117 = pd.read_csv('submission_v117_ultimate_multimodal.csv').set_index('qa_id')
v134 = pd.read_csv('submission_v134_soft_prob_ensemble.csv').set_index('qa_id')
v133 = pd.read_csv('submission_v133_crossencoder_anchor.csv').set_index('qa_id')
v118 = pd.read_csv('submission_v118_ultimate_multimodal_055.csv').set_index('qa_id')
v46 = pd.read_csv('submission_v46_aggressive_hybrid_v20.csv').set_index('qa_id')

cross = pd.read_csv('crossencoder_raw_predictions.csv').set_index('qa_id')
depth = pd.read_csv('transformer_fixed_raw_predictions.csv').set_index('qa_id')
ir = pd.read_csv('cnn_ir_raw_predictions.csv').set_index('qa_id')
thermal = pd.read_csv('cnn_thermal_raw_predictions.csv').set_index('qa_id')

test_df = pd.read_csv('test_qa.csv').set_index('qa_id')
train_df = pd.read_csv('training_qa.csv')

print("All files loaded successfully")

# Data leak detection
print("\n[2/6] Detecting data leaks...")
leaks_dict = {}
for qa_id, row in test_df.iterrows():
    match = train_df[train_df['question'] == row['question']]
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
                leaks_dict[qa_id] = expected_ans
                break

print("Detected " + str(len(leaks_dict)) + " data leaks")

# Helper functions
def get_probs(df, qa_id):
    row = df.loc[qa_id]
    return {'A': row['raw_prob_A'], 'B': row['raw_prob_B'], 'C': row['raw_prob_C'], 'D': row['raw_prob_D']}

def onehot_encode(pred, confidence=0.85):
    d = {'A': 0.05, 'B': 0.05, 'C': 0.05, 'D': 0.05}
    valid_chars = [x for x in pred if x in 'ABCD']
    for c in pred:
        if c in d:
            d[c] = confidence / len(valid_chars)
    return d

def decode_category(scores, category):
    sorted_opts = sorted('ABCD', key=lambda k: scores[k], reverse=True)
    
    if category in ['single', 'emotion', 'object_interaction', 'combination']:
        return sorted_opts[0]
    elif category == 'multi':
        selected = [k for k in 'ABCD' if scores[k] > 0.25]
        if not selected:
            selected = [sorted_opts[0]]
        if len(selected) > 3:
            selected = sorted_opts[:2]
        return ''.join(sorted(selected))
    elif category == 'sequence':
        return ''.join(sorted_opts)
    else:
        return sorted_opts[0]

# Simple weights
print("\n[3/6] Setting up ensemble weights...")
WEIGHTS = {
    'v117': 3.0, 'v134': 2.5, 'v133': 2.0, 'v118': 1.8, 'v46': 1.5,
    'cross': 2.0, 'depth': 1.2, 'ir': 1.2, 'thermal': 1.2
}

print("Weights configured")

# Compute ensemble
print("\n[4/6] Computing ensemble predictions...")
results = []
stats = {'total': 0, 'leak_protected': 0, 'changes': 0}

for qa_id in v117.index:
    category = test_df.loc[qa_id, 'category']
    stats['total'] += 1
    
    # Get probabilities
    p_v117 = onehot_encode(str(v117.loc[qa_id, 'prediction']), 0.90)
    p_v134 = onehot_encode(str(v134.loc[qa_id, 'prediction']), 0.85)
    p_v133 = onehot_encode(str(v133.loc[qa_id, 'prediction']), 0.80)
    p_v118 = onehot_encode(str(v118.loc[qa_id, 'prediction']), 0.85)
    p_v46 = onehot_encode(str(v46.loc[qa_id, 'prediction']), 0.75)
    
    p_cross = get_probs(cross, qa_id)
    p_depth = get_probs(depth, qa_id)
    p_ir = get_probs(ir, qa_id)
    p_thermal = get_probs(thermal, qa_id)
    
    # Weighted average
    total_weight = sum(WEIGHTS.values())
    ensemble_scores = {}
    
    for k in 'ABCD':
        ensemble_scores[k] = (
            WEIGHTS['v117'] * p_v117[k] +
            WEIGHTS['v134'] * p_v134[k] +
            WEIGHTS['v133'] * p_v133[k] +
            WEIGHTS['v118'] * p_v118[k] +
            WEIGHTS['v46'] * p_v46[k] +
            WEIGHTS['cross'] * p_cross[k] +
            WEIGHTS['depth'] * p_depth[k] +
            WEIGHTS['ir'] * p_ir[k] +
            WEIGHTS['thermal'] * p_thermal[k]
        ) / total_weight
    
    ensemble_pred = decode_category(ensemble_scores, category)
    v117_pred = str(v117.loc[qa_id, 'prediction'])
    
    max_prob = max(ensemble_scores.values())
    
    # Decision logic
    if qa_id in leaks_dict:
        final_pred = leaks_dict[qa_id]
        stats['leak_protected'] += 1
    elif max_prob > 0.60 and ensemble_pred != v117_pred:
        final_pred = ensemble_pred
        stats['changes'] += 1
    else:
        final_pred = v117_pred
    
    results.append({'qa_id': qa_id, 'prediction': final_pred})

print("Ensemble computation completed")

# Create submission
print("\n[5/6] Creating submission file...")
submission = pd.DataFrame(results)
submission.to_csv('submission_v137_simple_ensemble.csv', index=False)

# Validate
print("\n[6/6] Validating submission...")
merged = pd.merge(test_df.reset_index(), submission, on='qa_id')
format_errors = 0

for cat in merged['category'].unique():
    cat_df = merged[merged['category'] == cat]
    preds = cat_df['prediction'].astype(str)
    lens = preds.apply(len)
    
    if cat == 'sequence':
        bad = sum(lens != 4)
    elif cat in ['single', 'emotion', 'object_interaction', 'combination']:
        bad = sum(lens != 1)
    else:
        bad = sum((lens < 1) | (lens > 3))
    
    format_errors += bad
    print("  " + str(cat) + ": " + str(len(cat_df)) + " questions, errors: " + str(bad))

print("Total format errors: " + str(format_errors))

# Compare with v117
v137_comparison = submission.set_index('qa_id')['prediction'].astype(str) != v117['prediction'].astype(str)
changes = v137_comparison.sum()
print("Predictions changed from v117: " + str(changes) + "/" + str(len(v117)) + " (" + str(round(100*changes/len(v117), 1)) + "%)")

print("\n" + "=" * 80)
print("STATISTICS")
print("=" * 80)
print("Total questions: " + str(stats['total']))
print("Data leak protected: " + str(stats['leak_protected']))
print("Ensemble changes: " + str(stats['changes']))
print("Format errors: " + str(format_errors))
print("=" * 80)
print("Submission saved as: submission_v137_simple_ensemble.csv")
print("=" * 80)