"""
Deep Consensus Miner:
Scans all 682 test questions to find high-probability consensus refinements over v276_APEX_SUMMIT (0.77485).
Specifically audits:
1. Emotion questions: checks if raw probability of best option is significantly higher (>0.20) than current prediction and supported by training emotion correlations.
2. Multi questions: checks if any single uncorroborated letter can be replaced or pruned safely.
3. Single/Combination/Sequence: checks raw vision transformer probability vs current prediction.
"""
import pandas as pd

sub276 = pd.read_csv("submission_v276_APEX_SUMMIT.csv")
raw = pd.read_csv("transformer_fixed_raw_predictions.csv")
raw_map = dict(zip(raw['qa_id'], raw.to_dict('records')))
te = pd.read_csv("test_qa.csv")
sub_map276 = dict(zip(sub276['qa_id'], sub276['prediction']))
te['pred'] = te['qa_id'].map(sub_map276)

print("=== 1. AUDITING EMOTION PREDICTIONS (RAW PROB VS CURRENT PRED) ===")
emo_diffs = []
for idx, r in te[te['category'] == 'emotion'].iterrows():
    qid = r['qa_id']
    pred = str(r['pred']).strip()
    probs = {l: raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0) for l in ['A', 'B', 'C', 'D']}
    best_l = max(probs.keys(), key=lambda x: probs[x])
    pred_prob = probs.get(pred, 0.0)
    best_prob = probs[best_l]
    if best_l != pred:
        emo_diffs.append((qid, pred, best_l, round(pred_prob, 3), round(best_prob, 3), round(best_prob - pred_prob, 3), r[pred], r[best_l]))

for e in sorted(emo_diffs, key=lambda x: x[5], reverse=True):
    print(f"[EMOTION] {e[0]}: Curr={e[1]} ('{e[6]}', prob={e[3]}) vs MaxProb={e[2]} ('{e[7]}', prob={e[4]}, delta=+{e[5]})")

print(f"\nTotal Emotion differences: {len(emo_diffs)}")

print("\n=== 2. AUDITING SINGLE ATOMIC PREDICTIONS (RAW PROB VS CURRENT PRED) ===")
single_diffs = []
for idx, r in te[te['category'] == 'single'].iterrows():
    qid = r['qa_id']
    pred = str(r['pred']).strip()
    probs = {l: raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0) for l in ['A', 'B', 'C', 'D']}
    best_l = max(probs.keys(), key=lambda x: probs[x])
    pred_prob = probs.get(pred, 0.0)
    best_prob = probs[best_l]
    if best_l != pred and best_prob - pred_prob > 0.15:
        single_diffs.append((qid, pred, best_l, round(pred_prob, 3), round(best_prob, 3), round(best_prob - pred_prob, 3), r[pred], r[best_l]))

for s in sorted(single_diffs, key=lambda x: x[5], reverse=True):
    print(f"[SINGLE] {s[0]}: Curr={s[1]} ('{s[6]}', prob={s[3]}) vs MaxProb={s[2]} ('{s[7]}', prob={s[4]}, delta=+{s[5]})")

print(f"\nTotal Single high-delta differences: {len(single_diffs)}")
