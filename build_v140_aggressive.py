"""
Aggressive Optimization Ensemble - v140
More aggressive optimization to push beyond current limits
"""
import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("AGGRESSIVE OPTIMIZATION ENSEMBLE - v140")
print("Pushing beyond current limits")
print("=" * 80)

# Load all prediction sources
print("\n[1/8] Loading all prediction sources...")
v117 = pd.read_csv('submission_v117_ultimate_multimodal.csv').set_index('qa_id')
v137 = pd.read_csv('submission_v137_simple_ensemble.csv').set_index('qa_id')
v138 = pd.read_csv('submission_v138_deep_ensemble.csv').set_index('qa_id')
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
print("\n[2/8] Enhanced data leak detection...")
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

# More aggressive weights
print("\n[3/8] Setting up aggressive weights...")
AGGRESSIVE_WEIGHTS = {
    'single': {
        'v138': 3.5, 'v137': 3.0, 'v117': 2.5, 'v134': 2.0, 'v133': 1.5, 'v118': 1.2,
        'cross': 2.5, 'depth': 1.5, 'ir': 1.5, 'thermal': 1.5
    },
    'multi': {
        'v134': 3.5, 'v138': 3.0, 'v137': 2.5, 'v117': 2.0, 'v133': 1.5, 'v118': 1.0,
        'cross': 3.0, 'depth': 1.2, 'ir': 1.2, 'thermal': 1.2
    },
    'sequence': {
        'v133': 3.5, 'v134': 3.0, 'cross': 3.5, 'v138': 2.5, 'v137': 2.0,
        'v117': 1.5, 'v118': 1.0, 'depth': 1.0, 'ir': 1.0, 'thermal': 1.0
    },
    'emotion': {
        'v138': 3.5, 'v117': 3.0, 'v137': 2.5, 'v134': 1.8, 'v133': 1.5, 'v118': 1.2,
        'cross': 1.5, 'depth': 1.8, 'ir': 1.8, 'thermal': 1.8
    },
    'object_interaction': {
        'v138': 3.5, 'v137': 3.0, 'depth': 2.0, 'v117': 2.0, 'v134': 1.8, 'v118': 1.0,
        'cross': 1.5, 'ir': 2.0, 'thermal': 2.0
    },
    'combination': {
        'v138': 3.5, 'v134': 3.0, 'v137': 2.5, 'v117': 2.0, 'v118': 1.2, 'cross': 2.5,
        'depth': 1.3, 'ir': 1.3, 'thermal': 1.3
    }
}

print("Aggressive weights configured")

# Lower thresholds for more changes
print("\n[4/8] Setting up aggressive thresholds...")
THRESHOLDS = {
    'single': 0.55,
    'multi': 0.50,
    'sequence': 0.45,
    'emotion': 0.55,
    'object_interaction': 0.50,
    'combination': 0.52
}

print("Aggressive thresholds configured")

# Compute aggressive ensemble
print("\n[5/8] Computing aggressive ensemble predictions...")
results = []
stats = {'total': 0, 'leak_protected': 0, 'aggressive_overrides': 0, 'moderate_overrides': 0, 'changes': 0}

for qa_id in v117.index:
    category = test_df.loc[qa_id, 'category']
    weights = AGGRESSIVE_WEIGHTS[category]
    threshold = THRESHOLDS[category]
    stats['total'] += 1
    
    # Get all predictions
    preds = {
        'v117': str(v117.loc[qa_id, 'prediction']),
        'v137': str(v137.loc[qa_id, 'prediction']),
        'v138': str(v138.loc[qa_id, 'prediction']),
        'v134': str(v134.loc[qa_id, 'prediction']),
        'v133': str(v133.loc[qa_id, 'prediction']),
        'v118': str(v118.loc[qa_id, 'prediction']),
        'v46': str(v46.loc[qa_id, 'prediction'])
    }
    
    # Get probability distributions
    p_v117 = onehot_encode(preds['v117'], 0.90)
    p_v137 = onehot_encode(preds['v137'], 0.92)
    p_v138 = onehot_encode(preds['v138'], 0.88)
    p_v134 = onehot_encode(preds['v134'], 0.85)
    p_v133 = onehot_encode(preds['v133'], 0.80)
    p_v118 = onehot_encode(preds['v118'], 0.85)
    p_v46 = onehot_encode(preds['v46'], 0.75)
    
    p_cross = get_probs(cross, qa_id)
    p_depth = get_probs(depth, qa_id)
    p_ir = get_probs(ir, qa_id)
    p_thermal = get_probs(thermal, qa_id)
    
    # Calculate weighted average
    total_weight = sum(weights.values())
    ensemble_scores = {}
    
    for k in 'ABCD':
        ensemble_scores[k] = (
            weights['v138'] * p_v138[k] +
            weights['v137'] * p_v137[k] +
            weights['v117'] * p_v117[k] +
            weights['v134'] * p_v134[k] +
            weights.get('v133', 0) * p_v133[k] +
            weights['cross'] * p_cross[k] +
            weights['depth'] * p_depth[k] +
            weights['ir'] * p_ir[k] +
            weights['thermal'] * p_thermal[k]
        ) / total_weight
    
    # Get ensemble prediction
    ensemble_pred = decode_category(ensemble_scores, category)
    
    # Confidence metrics
    max_prob = max(ensemble_scores.values())
    second_max = sorted(ensemble_scores.values())[-2]
    confidence_margin = max_prob - second_max
    entropy = -sum(p * np.log(p + 1e-10) for p in ensemble_scores.values())
    
    # Count model agreements
    all_model_preds = list(preds.values())
    pred_counts = Counter(all_model_preds)
    most_common = pred_counts.most_common(1)[0]
    majority_pred = most_common[0]
    majority_count = most_common[1]
    
    # Aggressive decision logic
    if qa_id in leaks_dict:
        final_pred = leaks_dict[qa_id]
        stats['leak_protected'] += 1
    elif majority_count >= 5 and majority_pred != preds['v138']:
        # Strong majority agreement
        final_pred = majority_pred
        stats['aggressive_overrides'] += 1
        stats['changes'] += 1
    elif max_prob > threshold and confidence_margin > 0.10:
        # High confidence with good margin
        final_pred = ensemble_pred
        stats['aggressive_overrides'] += 1
        stats['changes'] += 1
    elif max_prob > (threshold - 0.05) and entropy < 1.3:
        # Moderate confidence with low entropy
        final_pred = ensemble_pred
        stats['moderate_overrides'] += 1
        stats['changes'] += 1
    else:
        # Default to v138
        final_pred = preds['v138']
    
    results.append({'qa_id': qa_id, 'prediction': final_pred})

print("Aggressive ensemble computation completed")

# Create submission
print("\n[6/8] Creating submission file...")
submission = pd.DataFrame(results)
submission.to_csv('submission_v140_aggressive.csv', index=False)

# Validate
print("\n[7/8] Validating submission...")
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

# Compare with previous versions
v140_vs_v117 = submission.set_index('qa_id')['prediction'].astype(str) != v117['prediction'].astype(str)
v140_vs_v137 = submission.set_index('qa_id')['prediction'].astype(str) != v137['prediction'].astype(str)
v140_vs_v138 = submission.set_index('qa_id')['prediction'].astype(str) != v138['prediction'].astype(str)

changes_v117 = v140_vs_v117.sum()
changes_v137 = v140_vs_v137.sum()
changes_v138 = v140_vs_v138.sum()

print("Predictions changed from v117: " + str(changes_v117) + "/" + str(len(v117)) + " (" + str(round(100*changes_v117/len(v117), 1)) + "%)")
print("Predictions changed from v137: " + str(changes_v137) + "/" + str(len(v137)) + " (" + str(round(100*changes_v137/len(v137), 1)) + "%)")
print("Predictions changed from v138: " + str(changes_v138) + "/" + str(len(v138)) + " (" + str(round(100*changes_v138/len(v138), 1)) + "%)")

# Category-wise analysis
print("\n[8/8] Category-wise analysis:")
for cat in merged['category'].unique():
    cat_df = merged[merged['category'] == cat]
    v140_cat = submission.set_index('qa_id').loc[cat_df['qa_id']]
    v138_cat = v138.loc[cat_df['qa_id']]
    changes = (v140_cat['prediction'].astype(str) != v138_cat['prediction'].astype(str)).sum()
    print("  " + str(cat) + ": " + str(changes) + "/" + str(len(cat_df)) + " changed (" + str(round(100*changes/len(cat_df), 1)) + "%)")

print("\n" + "=" * 80)
print("STATISTICS")
print("=" * 80)
print("Total questions: " + str(stats['total']))
print("Data leak protected: " + str(stats['leak_protected']))
print("Aggressive overrides: " + str(stats['aggressive_overrides']))
print("Moderate overrides: " + str(stats['moderate_overrides']))
print("Total changes: " + str(stats['changes']))
print("Format errors: " + str(format_errors))
print("=" * 80)
print("Submission saved as: submission_v140_aggressive.csv")
print("=" * 80)