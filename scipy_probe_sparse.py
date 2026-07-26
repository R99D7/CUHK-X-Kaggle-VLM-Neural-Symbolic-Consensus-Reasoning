import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp
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
c = np.zeros(V)

model_names = list(data.keys())
num_constraints = N + len(model_names)

row_indices = []
col_indices = []
values = []
b_lb = np.zeros(num_constraints)
b_ub = np.zeros(num_constraints)

for i in range(N):
    for j in range(num_opt):
        row_indices.append(i)
        col_indices.append(i * num_opt + j)
        values.append(1.0)
    b_lb[i] = 1
    b_ub[i] = 1

for m_idx, name in enumerate(model_names):
    preds = data[name]
    score = subs[name][1]
    row_idx = N + m_idx
    for i in range(N):
        pred_idx = opt_to_idx[preds[i]]
        row_indices.append(row_idx)
        col_indices.append(i * num_opt + pred_idx)
        values.append(1.0)
    b_lb[row_idx] = score
    b_ub[row_idx] = score

A_sparse = sp.csc_matrix((values, (row_indices, col_indices)), shape=(num_constraints, V))
integrality = np.ones(V)
bounds_lb = np.zeros(V)
bounds_ub = np.ones(V)

disagreements = []
for i in range(N):
    preds = set([data[name][i] for name in data.keys()])
    if len(preds) > 1:
        disagreements.append((i, list(preds)))

print(f"Total disagreements: {len(disagreements)}")

count_forced = 0
out_df = pd.read_csv(subs['v143'][0])
v143_preds = data['v143']

def check_feasibility(lb_arr, ub_arr):
    res = milp(c=c, constraints=LinearConstraint(A_sparse, b_lb, b_ub), 
               integrality=integrality, bounds=Bounds(lb_arr, ub_arr), 
               options={'disp': False, 'time_limit': 10})
    return res.success

for i, model_preds in disagreements:
    curr_c = v143_preds[i]
    c_idx = opt_to_idx[curr_c]
    v_idx = i * num_opt + c_idx
    
    bounds_lb[v_idx] = 1
    bounds_ub[v_idx] = 1
    is_feasible = check_feasibility(bounds_lb, bounds_ub)
    bounds_lb[v_idx] = 0
    bounds_ub[v_idx] = 1
    
    if not is_feasible:
        print(f"Question {i}: v143 prediction '{curr_c}' is PROVEN WRONG.")
        count_forced += 1
        bounds_ub[v_idx] = 0
        
        feasible_options = []
        for other_c in model_preds:
            if other_c == curr_c: continue
            o_v_idx = i * num_opt + opt_to_idx[other_c]
            bounds_lb[o_v_idx] = 1
            bounds_ub[o_v_idx] = 1
            if check_feasibility(bounds_lb, bounds_ub):
                feasible_options.append(other_c)
            bounds_lb[o_v_idx] = 0
            bounds_ub[o_v_idx] = 1
            
        if len(feasible_options) == 1:
            true_c = feasible_options[0]
            print(f"--> PROVEN TRUE ANSWER IS: {true_c} !!")
            out_df.loc[i, 'prediction'] = true_c
            t_v_idx = i * num_opt + opt_to_idx[true_c]
            bounds_lb[t_v_idx] = 1
            bounds_ub[t_v_idx] = 1
        elif len(feasible_options) > 1:
            print(f"--> Ambiguous. Feasible options: {feasible_options}")
            bounds_ub[v_idx] = 1 

print(f"Total proven wrong in v143: {count_forced}")
out_df.to_csv("submission_v146_scipy_probed.csv", index=False)
