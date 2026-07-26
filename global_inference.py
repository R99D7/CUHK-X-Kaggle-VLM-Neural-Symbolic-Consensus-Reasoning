import pandas as pd
import math
from itertools import combinations

te = pd.read_csv('test_qa.csv')
raw = pd.read_csv('transformer_fixed_raw_predictions.csv')
sub = pd.read_csv('submission.csv')

# Group by video
vids = te['path'].unique()

new_preds = {}

# Keep track of changes
changes_single = 0
changes_multi = 0
changes_comb = 0

for vid in vids:
    q_df = te[te['path'] == vid]
    
    cats = {}
    for _, row in q_df.iterrows():
        cats[row['category']] = row
        
    # We only optimize single, multi, and combination jointly.
    if 'multi' not in cats or 'combination' not in cats or 'single' not in cats:
        continue
        
    s_q = cats['single']
    m_q = cats['multi']
    c_q = cats['combination']
    
    # Sequence acts
    seq_acts = set()
    if 'sequence' in cats:
        seq_q = cats['sequence']
        seq_pred = str(sub[sub['qa_id'] == seq_q['qa_id']]['prediction'].values[0]).strip()
        if len(seq_pred) == 4:
            opts = {l: str(seq_q[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
            seq_acts = set([opts.get(l, "") for l in seq_pred])
            
    # Load probs
    s_r = raw[raw['qa_id'] == s_q['qa_id']]
    c_r = raw[raw['qa_id'] == c_q['qa_id']]
    m_r = raw[raw['qa_id'] == m_q['qa_id']]
    
    if s_r.empty or c_r.empty or m_r.empty: continue
    s_r = s_r.iloc[0]
    c_r = c_r.iloc[0]
    m_r = m_r.iloc[0]
    
    # Build options
    s_opts = {l: str(s_q[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    c_opts = {l: [a.strip() for a in str(c_q[l]).strip().lower().split(',')] for l in ['A', 'B', 'C', 'D']}
    m_opts = {l: str(m_q[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    
    best_score = -float('inf')
    best_config = None
    
    best_unconstrained_score = -float('inf')
    best_unconstrained_config = None
    
    # 4 (single) * 4 (comb) * 16 (multi) = 256
    # For multi subsets
    subsets = []
    for r in range(5):
        for combo in combinations(['A', 'B', 'C', 'D'], r):
            subsets.append("".join(combo))
            
    for s_choice in ['A', 'B', 'C', 'D']:
        p_s = s_r[f'raw_prob_{s_choice}']
        log_ps = math.log(p_s) if p_s > 0 else -1e9
        
        for c_choice in ['A', 'B', 'C', 'D']:
            p_c = c_r[f'raw_prob_{c_choice}']
            log_pc = math.log(p_c) if p_c > 0 else -1e9
            
            for m_choice in subsets:
                log_pm = 0
                for l in ['A', 'B', 'C', 'D']:
                    p_m = m_r[f'raw_prob_{l}']
                    if l in m_choice:
                        log_pm += math.log(p_m) if p_m > 0 else -1e9
                    else:
                        log_pm += math.log(1 - p_m) if (1 - p_m) > 0 else -1e9
                        
                total_score = log_ps + log_pc + log_pm
                
                # Update unconstrained
                if total_score > best_unconstrained_score:
                    best_unconstrained_score = total_score
                    best_unconstrained_config = (s_choice, c_choice, m_choice)
                
                # Check constraints
                # T = S U C U M U Seq
                T = set()
                if s_opts[s_choice]: T.add(s_opts[s_choice])
                T.update(c_opts[c_choice])
                for l in m_choice:
                    if m_opts[l]: T.add(m_opts[l])
                T.update(seq_acts)
                
                # Constraint 1: Multi completeness
                valid_multi = True
                for l, act in m_opts.items():
                    if act in T and l not in m_choice:
                        valid_multi = False
                        break
                if not valid_multi: continue
                
                # Constraint 2: Comb exhaustiveness
                # Find all valid comb options
                valid_c_opts = []
                for l, acts in c_opts.items():
                    if set(acts).issubset(T):
                        valid_c_opts.append(l)
                        
                if c_choice not in valid_c_opts: continue # Should be in valid since C is in T
                
                max_len = max([len(c_opts[l]) for l in valid_c_opts])
                if len(c_opts[c_choice]) < max_len:
                    continue # Not maximal
                    
                # Passed all constraints!
                if total_score > best_score:
                    best_score = total_score
                    best_config = (s_choice, c_choice, m_choice)
                    
    # Decide final configuration
    final_config = best_config if best_config else best_unconstrained_config
    if final_config:
        s_choice, c_choice, m_choice = final_config
        new_preds[s_q['qa_id']] = s_choice
        new_preds[c_q['qa_id']] = c_choice
        new_preds[m_q['qa_id']] = m_choice

# Apply to sub
for qa_id, new_pred in new_preds.items():
    old_pred = str(sub[sub['qa_id'] == qa_id]['prediction'].values[0]).strip()
    if old_pred != new_pred:
        cat = te[te['qa_id'] == qa_id]['category'].values[0]
        print(f"Changed {cat} {qa_id}: {old_pred} -> {new_pred}")
        if cat == 'single': changes_single += 1
        elif cat == 'multi': changes_multi += 1
        elif cat == 'combination': changes_comb += 1
        
        sub.loc[sub['qa_id'] == qa_id, 'prediction'] = new_pred

print(f"\nTotal changes: Single {changes_single}, Multi {changes_multi}, Comb {changes_comb}")
sub.to_csv('submission_global.csv', index=False)
print("Saved to submission_global.csv")
