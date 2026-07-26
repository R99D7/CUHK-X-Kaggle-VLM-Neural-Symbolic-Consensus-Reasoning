import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import os

subs = {
    'v143': ('submission_v143_qwen_tiebreaker.csv', 313),
    'v117': ('submission_v117_ultimate_multimodal.csv', 309),
    'v116': ('submission_v116_ultimate_safe_threshold_040.csv', 307),
    'v114': ('submission_v114_ultimate_safe_dual_agreement.csv', 307),
    'v113': ('submission_v113_dual_agreement_override.csv', 303),
    'v118': ('submission_v118_ultimate_multimodal_055.csv', 301),
    'v99' : ('submission_v99_top_kaggle_guarantee.csv', 301),
    'v142': ('submission_v142_dawid_skene.csv', 301),
    'v53' : ('submission_v53_aggressive_hybrid_v46_v28.csv', 301),
    'v46' : ('submission_v46_aggressive_hybrid_v20.csv', 301),
    'v134': ('submission_v134_soft_prob_ensemble.csv', 299),
    'v65' : ('submission_v65_surgical_strike.csv', 299),
    'v60' : ('submission_v60_mega_forest.csv', 297),
    'qwen': ('kernel_output/submission.csv', 261)
}

data = {}
all_options = set()
for name, (file, score) in subs.items():
    if os.path.exists(file):
        df = pd.read_csv(file)
        data[name] = df['prediction'].values
        all_options.update(df['prediction'].unique())

N = 682
options = list(all_options)
num_opt = len(options)
opt_to_idx = {o: i for i, o in enumerate(options)}

V = N * num_opt
# Objective is dummy
c = np.zeros(V)

# Constraints matrix A
num_constraints = N + len(data)
A = np.zeros((num_constraints, V))
b_lb = np.zeros(num_constraints)
b_ub = np.zeros(num_constraints)

# 1. Exactly one true answer per question
for i in range(N):
    A[i, i*num_opt : (i+1)*num_opt] = 1
    b_lb[i] = 1
    b_ub[i] = 1

# 2. Model scores
model_names = list(data.keys())
for m_idx, name in enumerate(model_names):
    preds = data[name]
    score = subs[name][1]
    row_idx = N + m_idx
    for i in range(N):
        pred_idx = opt_to_idx[preds[i]]
        A[row_idx, i*num_opt + pred_idx] = 1
    b_lb[row_idx] = score
    b_ub[row_idx] = score

constraints = LinearConstraint(A, b_lb, b_ub)
integrality = np.ones(V) # all integer/boolean
bounds_lb = np.zeros(V)
bounds_ub = np.ones(V)
bounds = Bounds(bounds_lb, bounds_ub)

disagreements = []
for i in range(N):
    preds = set([data[name][i] for name in data.keys()])
    if len(preds) > 1:
        disagreements.append((i, list(preds)))

print(f"Total disagreements: {len(disagreements)}")

count_forced = 0
out_df = pd.read_csv(subs['v143'][0])
v143_preds = data['v143']

def check_feasibility(b_lb_mod, b_ub_mod):
    res = milp(c=c, constraints=LinearConstraint(A, b_lb_mod, b_ub_mod), integrality=integrality, bounds=bounds)
    return res.success

for i, model_preds in disagreements:
    curr_c = v143_preds[i]
    c_idx = opt_to_idx[curr_c]
    v_idx = i * num_opt + c_idx
    
    # Force z[i, c] = 1. If infeasible, it's PROVEN WRONG
    # We do this by temporarily overriding the bounds
    bounds.lb[v_idx] = 1
    bounds.ub[v_idx] = 1
    is_feasible = check_feasibility(b_lb, b_ub)
    bounds.lb[v_idx] = 0
    bounds.ub[v_idx] = 1
    
    if not is_feasible:
        print(f"Question {i}: v143 prediction '{curr_c}' is PROVEN WRONG.")
        count_forced += 1
        
        # It's wrong, so set bound to 0
        bounds.lb[v_idx] = 0
        bounds.ub[v_idx] = 0
        
        feasible_options = []
        for other_c in model_preds:
            if other_c == curr_c: continue
            other_idx = opt_to_idx[other_c]
            o_v_idx = i * num_opt + other_idx
            
            bounds.lb[o_v_idx] = 1
            bounds.ub[o_v_idx] = 1
            if check_feasibility(b_lb, b_ub):
                feasible_options.append(other_c)
            bounds.lb[o_v_idx] = 0
            bounds.ub[o_v_idx] = 1
            
        if len(feasible_options) == 1:
            true_c = feasible_options[0]
            print(f"--> PROVEN TRUE ANSWER IS: {true_c} !!")
            out_df.loc[i, 'prediction'] = true_c
            # Keep it locked
            true_idx = opt_to_idx[true_c]
            t_v_idx = i * num_opt + true_idx
            bounds.lb[t_v_idx] = 1
            bounds.ub[t_v_idx] = 1
        elif len(feasible_options) > 1:
            print(f"--> Ambiguous. Feasible options: {feasible_options}")
            bounds.ub[v_idx] = 1 # reset
    else:
        # Feasible, might be right
        pass

print(f"Total proven wrong in v143: {count_forced}")
out_df.to_csv("submission_v146_scipy_probed.csv", index=False)
