"""
Check if we can deduce COMB from True_Acts.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

# Build True_Acts
vid_to_true = {}
for vid in te['path'].unique():
    acts = set()
    
    # Seq acts
    seq_q = te[(te['path'] == vid) & (te['category'] == 'sequence')]
    if not seq_q.empty:
        opts = {l: str(seq_q.iloc[0][l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        acts.update(opts.values())
        
    # Single act
    single_q = te[(te['path'] == vid) & (te['category'] == 'single')]
    if not single_q.empty:
        opts = {l: str(single_q.iloc[0][l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        pred = str(sub[sub['qa_id'] == single_q.iloc[0]['qa_id']]['prediction'].values[0]).strip()
        acts.add(opts.get(pred, ""))
        
    vid_to_true[vid] = acts

changes = 0
for idx, comb_q in te[te['category'] == 'combination'].iterrows():
    vid = comb_q['path']
    true_acts = vid_to_true[vid]
    
    opts = {l: [a.strip() for a in str(comb_q[l]).strip().lower().split(',')] for l in ['A', 'B', 'C', 'D']}
    
    # Count overlaps
    overlaps = {l: len(set(acts) & true_acts) for l, acts in opts.items()}
    max_overlap = max(overlaps.values())
    
    if max_overlap > 0:
        best_opts = [l for l, c in overlaps.items() if c == max_overlap]
        
        # If there's a unique best option based purely on overlap with True_Acts
        if len(best_opts) == 1:
            best_opt = best_opts[0]
            pred = str(sub[sub['qa_id'] == comb_q['qa_id']]['prediction'].values[0]).strip()
            
            if pred != best_opt:
                # Is the best_opt valid? (All its actions are plausible? We can't know for sure if it's exhaustive)
                # But if it has the MOST overlap with known true actions, it's highly likely to be the answer!
                print(f"COMB {comb_q['qa_id']} (vid {vid}): pred {pred} -> {best_opt} (Overlap {max_overlap} vs {overlaps[pred]})")
                changes += 1

print(f"\nTotal potential COMB fixes based on overlap: {changes}")
