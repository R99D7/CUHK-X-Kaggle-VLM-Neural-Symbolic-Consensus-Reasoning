"""
Apply highly confident COMBINATION (> 0.4 prob) to MULTI and SINGLE.
"""
import pandas as pd

sub = pd.read_csv('submission.csv')
te = pd.read_csv('test_qa.csv')
raw = pd.read_csv('transformer_fixed_raw_predictions.csv')

def get_opts(df, qid):
    row = df[df['qa_id'] == qid].iloc[0]
    return {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}

def get_pred(qid):
    return str(sub[sub['qa_id'] == qid]['prediction'].values[0]).strip()

def get_prob(qid, pred_l):
    r = raw[raw['qa_id'] == qid]
    if r.empty: return 0.0
    return r.iloc[0][f'raw_prob_{pred_l}']

changes = 0
for vid in te['path'].unique():
    q_df = te[te['path'] == vid]
    cats = {row['category']: row['qa_id'] for _, row in q_df.iterrows()}
    
    if 'combination' not in cats: continue
    c_q = cats['combination']
    c_pred = get_pred(c_q)
    c_prob = get_prob(c_q, c_pred)
    
    # We trust COMB if prob > 0.4 (or 0.35, let's say 0.38 to be safe)
    if c_prob < 0.38: continue
    
    c_acts = [a.strip() for a in get_opts(te, c_q).get(c_pred, "").split(',')]
    
    # Check MULTI
    if 'multi' in cats:
        m_q = cats['multi']
        m_pred = get_pred(m_q)
        m_opts = get_opts(te, m_q)
        
        missing = []
        for l, txt in m_opts.items():
            if txt in c_acts and l not in m_pred:
                missing.append(l)
                
        if missing:
            new_m_pred = "".join(sorted(set(m_pred) | set(missing)))
            print(f"MULTI {m_q} (vid {vid}): {m_pred} -> {new_m_pred} (COMB prob {c_prob:.3f})")
            sub.loc[sub['qa_id'] == m_q, 'prediction'] = new_m_pred
            changes += 1
            # Update m_pred for SINGLE check
            m_pred = new_m_pred
            
    # Check SINGLE
    if 'single' in cats:
        s_q = cats['single']
        s_pred = get_pred(s_q)
        s_opts = get_opts(te, s_q)
        
        if s_opts.get(s_pred, "") not in c_acts:
            # Current single is invalid!
            valid_s = [l for l, txt in s_opts.items() if txt in c_acts]
            if len(valid_s) == 1:
                print(f"SINGLE {s_q} (vid {vid}): {s_pred} -> {valid_s[0]} (COMB prob {c_prob:.3f})")
                sub.loc[sub['qa_id'] == s_q, 'prediction'] = valid_s[0]
                changes += 1

print(f"\nTotal changes from high conf COMB: {changes}")
sub.to_csv('submission.csv', index=False)
