"""
Soft probability ensemble - v134
Strategy: Average raw softmax probabilities from 5 raw prediction sources,
then decode best answer per category.
This is mathematically superior to hard voting because it preserves confidence.
"""
import pandas as pd
import numpy as np
import re

# Load all raw probability files
print("Loading raw predictions...")
cross = pd.read_csv('crossencoder_raw_predictions.csv').set_index('qa_id')
depth = pd.read_csv('transformer_fixed_raw_predictions.csv').set_index('qa_id')
ir    = pd.read_csv('cnn_ir_raw_predictions.csv').set_index('qa_id')
thermal = pd.read_csv('cnn_thermal_raw_predictions.csv').set_index('qa_id')
trans = pd.read_csv('transformer_raw_predictions.csv').set_index('qa_id')

# Load v117 as a soft anchor (convert hard labels to one-hot)
v117 = pd.read_csv('submission_v117_ultimate_multimodal.csv').set_index('qa_id')
test_df = pd.read_csv('test_qa.csv').set_index('qa_id')

def get_probs(df, qa_id):
    row = df.loc[qa_id]
    return {k: row[f'raw_prob_{k}'] for k in 'ABCD'}

def onehot_v117(qa_id):
    pred = str(v117.loc[qa_id, 'prediction'])
    # Use soft one-hot: 0.85 for predicted, 0.05 for others
    d = {'A': 0.05, 'B': 0.05, 'C': 0.05, 'D': 0.05}
    for c in pred:
        if c in d:
            d[c] = 0.85 / len([x for x in pred if x in 'ABCD'])
    return d

def decode_pred(scores, category):
    """Decode probability scores to valid prediction string per category."""
    sorted_opts = sorted('ABCD', key=lambda k: scores[k], reverse=True)
    
    if category in ['single', 'emotion', 'object_interaction', 'combination']:
        return sorted_opts[0]
    elif category == 'multi':
        # Top 1-3 by probability, but use threshold approach
        # Include options above 0.25 threshold
        selected = [k for k in 'ABCD' if scores[k] > 0.25]
        if not selected:
            selected = [sorted_opts[0]]
        if len(selected) > 3:
            selected = sorted_opts[:2]
        return ''.join(sorted(selected))
    elif category == 'sequence':
        return ''.join(sorted_opts)  # All 4 in ranked order
    return sorted_opts[0]

print("Computing soft ensemble...")
results = []
# Weights for each source (tuned by quality)
# crossencoder: great on combination/sequence
# depth/ir/thermal: great on visual questions 
# v117 one-hot: our best baseline
WEIGHTS = {
    'cross':   1.8,
    'depth':   1.0,
    'ir':      1.0,
    'thermal': 1.0,
    'trans':   0.8,
    'v117':    3.5,   # Strong anchor on our best submission
}

for qa_id in v117.index:
    category = test_df.loc[qa_id, 'category']
    
    p_cross   = get_probs(cross, qa_id)
    p_depth   = get_probs(depth, qa_id)
    p_ir      = get_probs(ir, qa_id)
    p_thermal = get_probs(thermal, qa_id)
    p_trans   = get_probs(trans, qa_id)
    p_v117    = onehot_v117(qa_id)
    
    # Weighted average of all probability distributions
    total_w = sum(WEIGHTS.values())
    avg_scores = {}
    for k in 'ABCD':
        avg_scores[k] = (
            WEIGHTS['cross']   * p_cross[k]   +
            WEIGHTS['depth']   * p_depth[k]   +
            WEIGHTS['ir']      * p_ir[k]       +
            WEIGHTS['thermal'] * p_thermal[k]  +
            WEIGHTS['trans']   * p_trans[k]    +
            WEIGHTS['v117']    * p_v117[k]
        ) / total_w
    
    pred = decode_pred(avg_scores, category)
    results.append({'qa_id': qa_id, 'prediction': pred})

out = pd.DataFrame(results)
out.to_csv('submission_v134_soft_prob_ensemble.csv', index=False)

# Validate
merged = pd.merge(test_df.reset_index(), out, on='qa_id')
errors = 0
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
    errors += bad
    print(f"  {cat}: min={lens.min()} max={lens.max()} bad={bad}")

v117_diffs = sum(out.set_index('qa_id')['prediction'].astype(str) != v117['prediction'].astype(str))
print(f"\nTotal format errors: {errors}")
print(f"Diffs from v117: {v117_diffs}")
print("Saved: submission_v134_soft_prob_ensemble.csv")
